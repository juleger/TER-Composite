import os, argparse
import numpy as np
import cv2
from skimage import morphology

PHYS = {0: 1, 1: 2, 2: 3}
NAMES = {1: "Matrix", 2: "Fiber", 3: "Pore"}


def clean_mask(seg, min_size=200):
    for label in (1, 2):
        mask = seg == label
        mask = morphology.remove_small_objects(mask, min_size=min_size)
        mask = morphology.remove_small_holes(mask, area_threshold=min_size)
        seg[seg == label] = 0
        seg[mask] = label


parser = argparse.ArgumentParser()
parser.add_argument("seg")
parser.add_argument("--pixel-size", type=float, default=1.0)
parser.add_argument("--downscale", type=int, default=5)
parser.add_argument("--tri", action="store_true")
parser.add_argument("--out", default=None)
args = parser.parse_args()

seg = np.load(args.seg)
if args.downscale > 1:
    h, w = seg.shape
    seg = cv2.resize(seg.astype(np.uint8), (w // args.downscale, h // args.downscale), interpolation=cv2.INTER_NEAREST).astype(np.int64)

clean_mask(seg)

H, W = seg.shape
ps = args.pixel_size * max(args.downscale, 1)
if args.out is None:
    os.makedirs("results", exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.seg))[0].replace("_seg", "")
    args.out = f"results/{stem}_{'tri' if args.tri else 'q1'}.msh"

with open(args.out, "w") as f:
    f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
    phys_used = sorted({PHYS[int(v)] for v in np.unique(seg)})
    f.write(f"$PhysicalNames\n{len(phys_used)}\n")
    for tag in phys_used:
        f.write(f'2 {tag} "{NAMES[tag]}"\n')
    f.write("$EndPhysicalNames\n")
    f.write(f"$Nodes\n{(H + 1) * (W + 1)}\n")
    for i in range(H + 1):
        for j in range(W + 1):
            f.write(f"{i * (W + 1) + j + 1} {j * ps:.6g} {i * ps:.6g} 0\n")
    f.write("$EndNodes\n")
    ne = H * W * (2 if args.tri else 1)
    f.write(f"$Elements\n{ne}\n")
    eid = 1
    for i in range(H):
        for j in range(W):
            p = PHYS[int(seg[i, j])]
            n0, n1 = i * (W + 1) + j + 1, i * (W + 1) + j + 2
            n2, n3 = (i + 1) * (W + 1) + j + 2, (i + 1) * (W + 1) + j + 1
            if args.tri:
                if (i + j) % 2 == 0:
                    f.write(f"{eid} 2 2 {p} {p} {n0} {n1} {n2}\n"); eid += 1
                    f.write(f"{eid} 2 2 {p} {p} {n0} {n2} {n3}\n"); eid += 1
                else:
                    f.write(f"{eid} 2 2 {p} {p} {n0} {n1} {n3}\n"); eid += 1
                    f.write(f"{eid} 2 2 {p} {p} {n1} {n2} {n3}\n"); eid += 1
            else:
                f.write(f"{eid} 3 2 {p} {p} {n0} {n1} {n2} {n3}\n"); eid += 1
    f.write("$EndElements\n")

print(f" ->{args.out}  ({H}x{W} px, {ne} elements {'tri' if args.tri else 'Q1'})")

