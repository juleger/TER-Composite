import argparse
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
    parser.add_argument("--out", default="results/composite_quad.msh")
    args = parser.parse_args()

    image_path = Path(args.img_path)
    weights_path = Path(args.weights_path)
    mesh_path = Path(args.out)
    mesh_path = results_dir / mesh_path.name
    segmentation_path = results_dir / f"{image_path.stem}_seg.npy"
    seg = predict_image(str(image_path), weights_path=str(weights_path), save=True)
    h, w, ne = generate_mesh(seg, str(mesh_path), downscale=args.resolution, min_size=args.min_size, tri=args.mesh_type == "tri")

    print(f"Segmentation : {segmentation_path}")
    print(f"Maillage : {mesh_path}")
    print(f"Dimensions : {h}x{w}")
    print(f"Elements : {ne}")
