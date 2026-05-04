"""
mesh_delaunay.py — Maillage triangulaire Delaunay depuis masque de segmentation .npy

Stratégie géométrique (v2) :
  - Les contours pixel sont extraits via cv2 puis simplifiés avec l'algorithme
    de Douglas-Peucker (epsilon configurable). On obtient des polygones fidèles
    aux formes réelles sans forcer d'ellipse.
  - Chaque polygone est inséré dans OCC via addSpline (ou addPolyline selon
    l'option --spline). Les intersections entre contours proches sont gérées
    par un shrink pixel avant extraction, puis BooleanFragments OCC assure
    la conformité topologique.
  - Option --refine : raffinement adaptatif local aux interfaces fibres/pores
    sans exploser le nombre total d'éléments.

Groupes physiques : 1=Matrix  2=Fiber  3=Pore
Compatible : Gmsh 2.2, meshReader.cpp (P1/T3)

Usage :
  python mesh_Delauney.py img_seg.npy [options]

Options clés :
  --pixel-size FLOAT      Taille d'un pixel en µm (défaut: 1.0)
  --downscale INT         Facteur de sous-échantillonnage (défaut: 1)
  --epsilon FLOAT         Tolérance Douglas-Peucker en pixels (défaut: 1.5)
  --shrink INT            Érosion en pixels avant extraction des contours (défaut: 1)
  --mesh-size FLOAT       Taille cible des éléments en µm (défaut: 3×pixel_size)
  --mesh-size-min FLOAT   Taille min (défaut: mesh_size/4)
  --mesh-size-max FLOAT   Taille max (défaut: mesh_size×3)
  --refine                Raffinement local aux interfaces (BoundaryLayer-like)
  --refine-factor FLOAT   Facteur de réduction taille aux interfaces (défaut: 0.3)
  --refine-layers INT     Nombre de couches raffinées (défaut: 3)
  --spline                Utilise B-splines OCC au lieu de polylignes droites
  --min-area INT          Aire minimale blob en pixels² (défaut: 100)
  --no-pores              Ignore les pores
  --out PATH              Chemin de sortie .msh
  --gui                   Ouvre Gmsh en mode interactif après génération
"""

import os, sys, argparse, math
import numpy as np
import cv2
from skimage import morphology

try:
    import gmsh
except ImportError:
    print("Erreur : gmsh non installé.  pip install gmsh")
    sys.exit(1)

PHYS_NAME  = {1: "Matrix", 2: "Fiber", 3: "Pore"}
LABEL_MATRIX, LABEL_FIBER, LABEL_PORE = 0, 1, 2


# ─────────────────────────────────────────────────────────────────────────────
# Nettoyage morphologique
# ─────────────────────────────────────────────────────────────────────────────
def clean_mask(seg: np.ndarray, min_size: int = 100) -> np.ndarray:
    seg = seg.copy()
    for label in (LABEL_FIBER, LABEL_PORE):
        mask = seg == label
        try:
            mask = morphology.remove_small_objects(mask, min_size=min_size)
            mask = morphology.remove_small_holes(mask, area_threshold=min_size)
        except TypeError:
            mask = morphology.remove_small_objects(mask, max_size=min_size)
            mask = morphology.remove_small_holes(mask, max_size=min_size)
        seg[seg == label] = LABEL_MATRIX
        seg[mask] = label
    return seg


# ─────────────────────────────────────────────────────────────────────────────
# Extraction des contours simplifiés (Douglas-Peucker)
# ─────────────────────────────────────────────────────────────────────────────
def extract_contours(seg: np.ndarray, label: int, min_area: int = 100,
                     epsilon: float = 1.5, shrink: int = 1) -> list[np.ndarray]:
    """
    Extrait les contours de chaque blob de la phase `label`.
    - Érosion `shrink` pixels pour éviter les chevauchements entre phases proches.
    - Simplification Douglas-Peucker avec tolérance `epsilon` pixels.
    Retourne une liste de tableaux (N, 2) [col, row] en coordonnées pixel.
    """
    binary = (seg == label).astype(np.uint8)

    # Érosion pour séparer les fibres proches et éviter les intersections OCC
    if shrink > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * shrink + 1, 2 * shrink + 1))
        binary = cv2.erode(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    result = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        # Simplification Douglas-Peucker
        approx = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=True)
        pts = approx.reshape(-1, 2).astype(float)  # (N, 2) [x, y] pixel

        if len(pts) < 3:
            continue

        # Fermeture explicite (premier == dernier point pour OCC)
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])

        result.append(pts)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Insertion d'un contour polygonal dans OCC (coordonnées physiques)
# ─────────────────────────────────────────────────────────────────────────────
def add_occ_contour(pts_px: np.ndarray, ps: float, H: int,
                    mesh_size: float, use_spline: bool = False,
                    margin: float = 1.0) -> int | None:
    """
    Insère un contour fermé dans le modèle OCC courant.
    pts_px : tableau (N, 2) [col_pixel, row_pixel], premier == dernier point.
    Retourne le tag de surface, ou None en cas d'échec.
    """
    # Conversion pixel → coordonnées physiques avec inversion Y
    xs = pts_px[:, 0] * ps
    ys = (H - pts_px[:, 1]) * ps

    # Vérification que le contour est dans le domaine
    m = margin * ps
    W_phys = pts_px[:, 0].max() * ps  # sera comparé au bord plus bas
    if xs.min() < m or ys.min() < m:
        return None

    try:
        # Créer les points OCC (on déduplique le dernier point = premier)
        n = len(pts_px) - 1  # nombre de segments (pts[0]==pts[-1])
        point_tags = []
        for i in range(n):
            pt = gmsh.model.occ.addPoint(float(xs[i]), float(ys[i]), 0.0, mesh_size)
            point_tags.append(pt)

        if len(point_tags) < 3:
            return None

        # Créer les courbes
        curve_tags = []
        if use_spline and len(point_tags) >= 4:
            # B-spline fermée : OCC addBSpline prend la liste des points + fermeture
            spline_pts = point_tags + [point_tags[0]]
            curve = gmsh.model.occ.addSpline(spline_pts)
            curve_tags.append(curve)
        else:
            # Polylignes segment par segment
            for i in range(n):
                p1 = point_tags[i]
                p2 = point_tags[(i + 1) % n]
                line = gmsh.model.occ.addLine(p1, p2)
                curve_tags.append(line)

        loop = gmsh.model.occ.addCurveLoop(curve_tags)
        surf = gmsh.model.occ.addPlaneSurface([loop])
        return surf

    except Exception as ex:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Programme principal
# ─────────────────────────────────────────────────────────────────────────────
def generate_mesh_delaunay(seg: np.ndarray, out: str,
                           pixel_size: float = 1.0,
                           downscale: int = 1,
                           epsilon: float = 1.5,
                           shrink: int = 1,
                           min_area: int = 100,
                           no_pores: bool = False,
                           spline: bool = False,
                           mesh_size: float = None,
                           mesh_size_min: float = None,
                           mesh_size_max: float = None,
                           refine: bool = False,
                           refine_factor: float = 0.3,
                           refine_layers: int = 3,
                           optimize: bool = False,
                           algo: int = 5,
                           gui: bool = False) -> tuple[int, int, int]:
    """
    Génère un maillage Delaunay non structuré depuis un masque de segmentation numpy.

    Paramètres
    ----------
    seg          : masque numpy (int) — 0=Matrix, 1=Fiber, 2=Pore
    out          : chemin de sortie .msh
    pixel_size   : taille physique d'un pixel en µm
    downscale    : facteur de sous-échantillonnage (1 = pas de réduction)
    epsilon      : tolérance Douglas-Peucker en pixels (plus grand = moins de points)
    shrink       : érosion en pixels avant extraction contours (évite intersections OCC)
    min_area     : aire minimale blob en pixels²
    no_pores     : ignorer la phase pore
    spline       : utiliser des B-splines OCC au lieu de polylignes
    mesh_size    : taille cible éléments µm (défaut: 3×pixel_size)
    mesh_size_min: taille min éléments µm
    mesh_size_max: taille max éléments µm
    refine       : raffinement local aux interfaces
    refine_factor: facteur multiplicatif taille aux interfaces
    refine_layers: nombre de couches raffinées
    optimize     : optimisation Netgen post-génération
    algo         : algorithme Gmsh (5=Delaunay, 6=Frontal-Delaunay, 7=BAMG)
    gui          : ouvrir l'interface Gmsh interactive

    Retourne (H, W, n_elems) comme generate_mesh() de mesh.py.
    """
    seg = np.array(seg, copy=True)

    if downscale > 1:
        h, w = seg.shape
        seg = cv2.resize(seg.astype(np.uint8),
                         (w // downscale, h // downscale),
                         interpolation=cv2.INTER_NEAREST).astype(np.int64)
        print(f"    Sous-échantillonnage ×{downscale} → {seg.shape[0]}×{seg.shape[1]}")

    seg = clean_mask(seg, min_size=min_area)
    H, W = seg.shape
    ps = pixel_size * max(downscale, 1)

    _mesh_size     = mesh_size     or (3.0 * ps)
    _mesh_size_min = mesh_size_min or (_mesh_size / 4.0)
    _mesh_size_max = mesh_size_max or (_mesh_size * 3.0)

    print(f"    Grille  : {H}×{W} px  |  pixel = {ps} µm")
    print(f"    Maille  : cible={_mesh_size} µm  min={_mesh_size_min:.2f}  max={_mesh_size_max:.2f}")
    if refine:
        print(f"    Raffinement local : facteur={refine_factor}  couches={refine_layers}")

    # ── Extraction des contours ───────────────────────────────────────────────
    print("[2/5] Extraction des contours (Douglas-Peucker)...")
    fiber_contours = extract_contours(seg, LABEL_FIBER,
                                      min_area=min_area,
                                      epsilon=epsilon,
                                      shrink=shrink)
    pore_contours  = [] if no_pores else                      extract_contours(seg, LABEL_PORE,
                                      min_area=max(10, min_area // 5),
                                      epsilon=max(0.5, epsilon / 2),
                                      shrink=max(0, shrink - 1))

    print(f"    Fibres : {len(fiber_contours)} contours")
    print(f"    Pores  : {len(pore_contours)} contours")
    if fiber_contours:
        npts = [len(c) for c in fiber_contours]
        print(f"    Points/contour : min={min(npts)}  max={max(npts)}  moy={np.mean(npts):.0f}")

    # ── Construction OCC ──────────────────────────────────────────────────────
    print("[3/5] Construction géométrie OCC...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("composite_delaunay")

    bbox = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, W * ps, H * ps)

    fiber_surfs = []
    for pts in fiber_contours:
        s = add_occ_contour(pts, ps, H, _mesh_size_min,
                            use_spline=spline, margin=0.5)
        if s is not None:
            fiber_surfs.append(s)

    pore_surfs = []
    for pts in pore_contours:
        s = add_occ_contour(pts, ps, H, _mesh_size_min,
                            use_spline=spline, margin=0.0)
        if s is not None:
            pore_surfs.append(s)

    print(f"    Surfaces fibre OCC : {len(fiber_surfs)}")
    print(f"    Surfaces pore  OCC : {len(pore_surfs)}")

    gmsh.model.occ.synchronize()

    # ── BooleanFragments ──────────────────────────────────────────────────────
    all_inclusions = [(2, s) for s in fiber_surfs + pore_surfs]

    if all_inclusions:
        try:
            out_map, _ = gmsh.model.occ.fragment([(2, bbox)], all_inclusions)
            gmsh.model.occ.synchronize()
        except Exception as e:
            print(f"    AVERTISSEMENT BooleanFragments : {e}")
            print("    Tentative sans les pores...")
            gmsh.finalize()
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("composite_delaunay")
            bbox = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, W * ps, H * ps)
            fiber_surfs = []
            for pts in fiber_contours:
                s = add_occ_contour(pts, ps, H, _mesh_size_min,
                                    use_spline=spline, margin=0.5)
                if s is not None:
                    fiber_surfs.append(s)
            pore_surfs = []
            all_inclusions = [(2, s) for s in fiber_surfs]
            gmsh.model.occ.synchronize()
            out_map, _ = gmsh.model.occ.fragment([(2, bbox)], all_inclusions)
            gmsh.model.occ.synchronize()

        matrix_result, fiber_result, pore_result = [], [], []
        all_result_surfs = [tag for dim, tag in out_map if dim == 2]

        for stag in all_result_surfs:
            try:
                com = gmsh.model.occ.getCenterOfMass(2, stag)
                cx_px = int(np.clip(com[0] / ps,            0, W - 1))
                cy_px = int(np.clip((H * ps - com[1]) / ps, 0, H - 1))
                label = int(seg[cy_px, cx_px])
            except Exception:
                label = LABEL_MATRIX

            if label == LABEL_FIBER:
                fiber_result.append(stag)
            elif label == LABEL_PORE:
                pore_result.append(stag)
            else:
                matrix_result.append(stag)
    else:
        gmsh.model.occ.synchronize()
        matrix_result = [bbox]
        fiber_result  = []
        pore_result   = []

    print(f"    Fragments : matrice={len(matrix_result)}  "
          f"fibre={len(fiber_result)}  pore={len(pore_result)}")

    # ── Groupes physiques ─────────────────────────────────────────────────────
    if matrix_result:
        gmsh.model.addPhysicalGroup(2, matrix_result, tag=1, name="Matrix")
    if fiber_result:
        gmsh.model.addPhysicalGroup(2, fiber_result,  tag=2, name="Fiber")
    if pore_result:
        gmsh.model.addPhysicalGroup(2, pore_result,   tag=3, name="Pore")

    # ── Raffinement local ─────────────────────────────────────────────────────
    if refine and (fiber_result or pore_result):
        print("    Raffinement local aux interfaces...")
        inclusion_curves = []
        for stag in fiber_result + pore_result:
            boundary = gmsh.model.getBoundary([(2, stag)], oriented=False)
            inclusion_curves.extend([abs(b[1]) for b in boundary])
        inclusion_curves = list(set(inclusion_curves))

        if inclusion_curves:
            f_dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", inclusion_curves)
            gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 50)

            interface_size  = _mesh_size * refine_factor
            transition_dist = refine_layers * _mesh_size

            f_thresh = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_thresh, "InField",  f_dist)
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin",  interface_size)
            gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax",  _mesh_size_max)
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMin",  0.0)
            gmsh.model.mesh.field.setNumber(f_thresh, "DistMax",  transition_dist)

            gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)
            gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints",         0)
            gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature",      0)
        else:
            refine = False

    # ── Paramètres maillage ───────────────────────────────────────────────────
    gmsh.option.setNumber("Mesh.Algorithm",    algo)
    gmsh.option.setNumber("Mesh.Smoothing",    5)
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    if not refine:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", _mesh_size_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", _mesh_size_max)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)

    # ── Génération ────────────────────────────────────────────────────────────
    print(f"[4/5] Génération Delaunay (algo={algo})...")
    gmsh.model.mesh.generate(2)

    if optimize:
        print("    Optimisation Netgen...")
        gmsh.model.mesh.optimize("Netgen")

    # ── Export ────────────────────────────────────────────────────────────────
    print(f"[5/5] Export → {out}")
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.write(out)

    node_tags, _, _ = gmsh.model.mesh.getNodes()
    _, elem_tags, _ = gmsh.model.mesh.getElements(dim=2)
    n_elems = sum(len(et) for et in elem_tags)
    print(f"\n    ✓ Noeuds    : {len(node_tags)}")
    print(f"    ✓ Eléments  : {n_elems}  (T3 Delaunay)")
    print(f"    ✓ Fichier   : {out}")

    total = seg.size
    print("\n    Fractions volumiques (masque) :")
    for lbl, name in [(LABEL_MATRIX, "Matrix"), (LABEL_FIBER, "Fiber"), (LABEL_PORE, "Pore")]:
        frac = (seg == lbl).sum() / total * 100
        print(f"      {name:8s} : {frac:.2f}%")

    if gui:
        gmsh.fltk.run()

    gmsh.finalize()
    print("\nTerminé.")
    return H, W, n_elems


def main():
    parser = argparse.ArgumentParser(
        description="Maillage Delaunay non structuré depuis masque .npy (contours fidèles)"
    )
    parser.add_argument("seg",
        help="Chemin vers le masque .npy (valeurs entières 0=matrice 1=fibre 2=pore)")

    # Géométrie
    parser.add_argument("--pixel-size",    type=float, default=1.0,
        help="Taille d'un pixel en µm (défaut: 1.0)")
    parser.add_argument("--downscale",     type=int,   default=1,
        help="Sous-échantillonnage entier avant traitement (défaut: 1)")
    parser.add_argument("--epsilon",       type=float, default=1.5,
        help="Tolérance Douglas-Peucker en pixels (défaut: 1.5 — plus grand = moins de points)")
    parser.add_argument("--shrink",        type=int,   default=1,
        help="Érosion en pixels avant extraction contours (défaut: 1)")
    parser.add_argument("--min-area",      type=int,   default=100,
        help="Aire minimale blob en pixels² (défaut: 100)")
    parser.add_argument("--no-pores",      action="store_true",
        help="Ignore la phase pores")
    parser.add_argument("--spline",        action="store_true",
        help="Utilise des B-splines OCC (plus lisse, mais plus lent)")

    # Maillage
    parser.add_argument("--mesh-size",     type=float, default=None,
        help="Taille cible éléments en µm (défaut: 3×pixel_size)")
    parser.add_argument("--mesh-size-min", type=float, default=None,
        help="Taille min éléments en µm")
    parser.add_argument("--mesh-size-max", type=float, default=None,
        help="Taille max éléments en µm")

    # Raffinement local
    parser.add_argument("--refine",        action="store_true",
        help="Raffinement local aux interfaces fibre/matrice et pore/matrice")
    parser.add_argument("--refine-factor", type=float, default=0.3,
        help="Facteur multiplicatif mesh-size aux interfaces (défaut: 0.3)")
    parser.add_argument("--refine-layers", type=int,   default=3,
        help="Nombre de couches de points aux interfaces (défaut: 3)")

    # Sortie
    parser.add_argument("--optimize",      action="store_true", default=False,
        help="Optimisation Netgen post-génération (désactivée par défaut — peut geler)")
    parser.add_argument("--algo",          type=int, default=5,
        help="Algorithme de maillage Gmsh : 5=Delaunay (défaut, robuste), 6=Frontal-Delaunay (meilleure qualité mais peut geler), 7=BAMG")
    parser.add_argument("--out",           default=None,
        help="Chemin de sortie .msh (défaut: results/<stem>_delaunay.msh)")
    parser.add_argument("--gui",           action="store_true",
        help="Ouvre Gmsh en mode interactif après génération")

    args = parser.parse_args()

    # ── Chargement ───────────────────────────────────────────────────────────
    print(f"[1/5] Chargement : {args.seg}")
    seg = np.load(args.seg, allow_pickle=False)

    if args.downscale > 1:
        h, w = seg.shape
        seg = cv2.resize(seg.astype(np.uint8),
                         (w // args.downscale, h // args.downscale),
                         interpolation=cv2.INTER_NEAREST).astype(np.int64)
        print(f"    Sous-échantillonnage ×{args.downscale} → {seg.shape[0]}×{seg.shape[1]}")

    if args.out is None:
        os.makedirs("results", exist_ok=True)
        stem     = os.path.splitext(os.path.basename(args.seg))[0].replace("_seg", "")
        args.out = f"results/{stem}_delaunay.msh"

    generate_mesh_delaunay(
        seg=seg,
        out=args.out,
        pixel_size=args.pixel_size,
        downscale=args.downscale,
        epsilon=args.epsilon,
        shrink=args.shrink,
        min_area=args.min_area,
        no_pores=args.no_pores,
        spline=args.spline,
        mesh_size=args.mesh_size,
        mesh_size_min=args.mesh_size_min,
        mesh_size_max=args.mesh_size_max,
        refine=args.refine,
        refine_factor=args.refine_factor,
        refine_layers=args.refine_layers,
        optimize=args.optimize,
        algo=args.algo,
        gui=args.gui,
    )


if __name__ == "__main__":
    main()