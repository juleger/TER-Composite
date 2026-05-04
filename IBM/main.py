import argparse
import numpy as np
from pathlib import Path
import os

os.environ["MPLBACKEND"] = "Agg"

from mesh import generate_mesh
from mesh_Delauney import generate_mesh_delaunay
from segmentation import predict_image

"""
Point d'entrée unique du pipeline segmentation → maillage.

Utilisation :
  # Maillage structuré quadrangles (défaut)
  python main.py image.png --mesh-type quad

  # Maillage structuré triangles
  python main.py image.png --mesh-type tri

  # Maillage Delaunay non structuré
  python main.py image.png --mesh-type delaunay

  # Delaunay avec options avancées
  python main.py image.png --mesh-type delaunay \\
      --mesh-size 5.0 --epsilon 2.0 --shrink 2 --refine

Tous les types de maillage produisent un fichier .msh (Gmsh 2.2)
compatible avec meshReader.cpp.
"""

if __name__ == "__main__":
    base_dir    = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Pipeline segmentation → maillage pour composites"
    )

    # ── Arguments communs ────────────────────────────────────────────────────
    parser.add_argument("img_path",
        help="Chemin vers l'image du composite (.png, .tif, ...)")
    parser.add_argument("--weights-path", default="unet_weights.pth",
        help="Chemin vers les poids U-Net (défaut: unet_weights.pth)")
    parser.add_argument("--mesh-type",
        choices=["quad", "tri", "delaunay"], default="quad",
        help="Type de maillage : quad (Q1 structuré), tri (T3 structuré), "
             "delaunay (T3 non structuré Gmsh) — défaut: quad")
    parser.add_argument("--out", default=None,
        help="Chemin de sortie .msh (défaut: results/composite<N>_vf<VF>.msh)")

    # ── Options maillages structurés (quad / tri) ────────────────────────────
    parser.add_argument("--min-size", type=int, default=1000,
        help="[quad/tri] Taille minimale des objets en pixels² (défaut: 1000)")
    parser.add_argument("--resolution", type=int, default=4,
        help="[quad/tri] Facteur de sous-échantillonnage (défaut: 4)")

    # ── Options maillage Delaunay ─────────────────────────────────────────────
    parser.add_argument("--pixel-size", type=float, default=1.0,
        help="[delaunay] Taille physique d'un pixel en µm (défaut: 1.0)")
    parser.add_argument("--downscale", type=int, default=1,
        help="[delaunay] Facteur de sous-échantillonnage avant maillage (défaut: 1)")
    parser.add_argument("--epsilon", type=float, default=1.5,
        help="[delaunay] Tolérance Douglas-Peucker en pixels (défaut: 1.5)")
    parser.add_argument("--shrink", type=int, default=1,
        help="[delaunay] Érosion en pixels avant extraction contours (défaut: 1)")
    parser.add_argument("--mesh-size", type=float, default=None,
        help="[delaunay] Taille cible des éléments en µm (défaut: 3×pixel_size)")
    parser.add_argument("--mesh-size-min", type=float, default=None,
        help="[delaunay] Taille minimale des éléments en µm")
    parser.add_argument("--mesh-size-max", type=float, default=None,
        help="[delaunay] Taille maximale des éléments en µm")
    parser.add_argument("--refine", action="store_true",
        help="[delaunay] Raffinement local aux interfaces fibre/matrice")
    parser.add_argument("--refine-factor", type=float, default=0.3,
        help="[delaunay] Facteur de réduction de taille aux interfaces (défaut: 0.3)")
    parser.add_argument("--refine-layers", type=int, default=3,
        help="[delaunay] Nombre de couches raffinées (défaut: 3)")
    parser.add_argument("--spline", action="store_true",
        help="[delaunay] Utiliser des B-splines OCC au lieu de polylignes")
    parser.add_argument("--no-pores", action="store_true",
        help="[delaunay] Ignorer les porosités")
    parser.add_argument("--optimize", action="store_true", default=False,
        help="[delaunay] Optimisation Netgen post-génération (défaut: désactivé)")
    parser.add_argument("--algo", type=int, default=5,
        help="[delaunay] Algorithme Gmsh : 5=Delaunay (défaut), "
             "6=Frontal-Delaunay, 7=BAMG")
    parser.add_argument("--gui", action="store_true",
        help="[delaunay] Ouvrir l'interface Gmsh après génération")

    args = parser.parse_args()

    # ── Segmentation ─────────────────────────────────────────────────────────
    image_path   = Path(args.img_path)
    weights_path = Path(args.weights_path)
    stem_num     = image_path.stem.split("_")[0]
    seg_path     = results_dir / f"{stem_num}_seg.npy"

    if seg_path.exists():
        print(f"Segmentation déjà existante : {seg_path}")
        seg = np.load(str(seg_path))
    else:
        seg = predict_image(str(image_path), weights_path=str(weights_path), save=True)

    vf = np.sum(seg == 1) / seg.size  # fraction volumique fibre

    # ── Chemin de sortie ──────────────────────────────────────────────────────
    if args.out is None:
        suffix = {"quad": "q1", "tri": "tri", "delaunay": "delaunay"}[args.mesh_type]
        mesh_path = results_dir / f"composite{stem_num}_vf{vf:.3f}_{suffix}.msh"
    else:
        mesh_path = results_dir / Path(args.out).name

    # ── Génération du maillage ────────────────────────────────────────────────
    if args.mesh_type in ("quad", "tri"):
        print(f"Maillage structuré ({args.mesh_type.upper()})...")
        h, w, ne = generate_mesh(
            seg,
            str(mesh_path),
            downscale=args.resolution,
            min_size=args.min_size,
            tri=(args.mesh_type == "tri"),
        )

    else:  # delaunay
        print("Maillage Delaunay non structuré...")
        h, w, ne = generate_mesh_delaunay(
            seg=seg,
            out=str(mesh_path),
            pixel_size=args.pixel_size,
            downscale=args.downscale,
            epsilon=args.epsilon,
            shrink=args.shrink,
            min_area=args.min_size,
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

    # ── Résumé ────────────────────────────────────────────────────────────────
    h_carac = (1.0 / ne) ** 0.5 if ne > 0 else 0
    print(f"\nSegmentation       : {seg_path}")
    print(f"Maillage           : {mesh_path}")
    print(f"Type               : {args.mesh_type}")
    print(f"Dimensions         : {h}×{w} px")
    print(f"Éléments           : {ne}")
    print(f"Fraction volumique : {vf:.3f}")
    print(f"h caractéristique  : {h_carac:.4f}")