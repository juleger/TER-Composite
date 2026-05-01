import argparse
import numpy as np
from pathlib import Path
import os

os.environ["MPLBACKEND"] = "Agg"

from mesh import generate_mesh
from segmentation import predict_image

"""Utilisation type :
python main.py path/to/image.png --weights-path path/to/unet_weights.pth --mesh-type tri --min-size 1000 --resolution 4 --out results/mesh.msh"""

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("img_path")
    parser.add_argument("--weights-path", default="unet_weights.pth")
    parser.add_argument("--mesh-type", choices=["quad", "tri"], default="quad")
    parser.add_argument("--min-size", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    image_path = Path(args.img_path)
    weights_path = Path(args.weights_path)
    stem_num = image_path.stem.split('_')[0]
    segmentation_path = results_dir / f"{stem_num}_seg.npy"
    if segmentation_path.exists():
        print(f"Segmentation déjà existante : {segmentation_path}")
        seg = np.load(str(segmentation_path))
    else:
        seg = predict_image(str(image_path), weights_path=str(weights_path), save=True)
    vf = np.sum(seg == 1) / seg.size
    if args.out is None:
        mesh_path = results_dir / f"composite{stem_num}_vf{vf:.3f}.msh"
    else:
        mesh_path = results_dir / Path(args.out).name
    h, w, ne = generate_mesh(seg, str(mesh_path), downscale=args.resolution, min_size=args.min_size, tri=args.mesh_type == "tri")
    h_carac = (1.0 / ne) ** 0.5 if ne > 0 else 0  # Taille de maille caractéristique approximative
    print(f"Segmentation : {segmentation_path}")
    print(f"Maillage : {mesh_path}")
    print(f"Dimensions : {h}x{w}")
    print(f"Elements : {ne}")
    print(f"Taille de maille caractéristique h : {h_carac:.4f}")
