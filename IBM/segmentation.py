import os, glob, random, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
from scipy.ndimage import map_coordinates, gaussian_filter

"""
Ce script implémente la segmentation d'images de matériaux composites (Fibre, matrice, porosités) en utilisant un réseau de neurones de type U-Net. Le principe est de découper les images en patchs, d'entraîner le modèle sur ces patchs, puis d'utiliser une approche de "sliding window" pour faire des prédictions sur des images de taille arbitraire. Le réseau est entraîné à minimiser la perte de cross-entropie, et la performance est évaluée à l'aide de l'IoU moyen sur les classes présentes dans le batch. 
"""

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "unet_weights.pth")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PATCH_SIZE = 256
PATCHES_PER_IMAGE = 128
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 8
LR = 1e-3
NUM_CLASSES = 3
VAL_SPLIT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_COLORS = np.array([[0, 0, 0], [180, 100, 255], [255, 80, 80]], dtype=np.uint8)
CLASS_NAMES = ["Matrix", "Fiber", "Pore"]

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=NUM_CLASSES, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs, self.ups, self.pool = nn.ModuleList(), nn.ModuleList(), nn.MaxPool2d(2)
        ch = in_ch
        for f in features:
            self.downs.append(DoubleConv(ch, f)); ch = f
        self.bottleneck = DoubleConv(ch, ch * 2); ch = ch * 2
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.ups.append(DoubleConv(f * 2, f)); ch = f
        self.head = nn.Conv2d(ch, num_classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x); s = skips[-(i//2 + 1)]
            if x.shape != s.shape: x = F.interpolate(x, size=s.shape[2:])
            x = self.ups[i + 1](torch.cat([s, x], dim=1))
        return self.head(x)

class PatchDataset(Dataset):
    def __init__(self, patch_size=PATCH_SIZE, patches_per_image=PATCHES_PER_IMAGE, augment=True):
        self.patch_size, self.augment, self.patches = patch_size, augment, []
        for d in sorted(glob.glob(os.path.join(DATASET_DIR, "*"))):
            img = np.array(Image.open(os.path.join(d, "img.png")).convert("L"), dtype=np.float32) / 255.0
            lbl = np.array(Image.open(os.path.join(d, "label.png")), dtype=np.int64)
            H, W = img.shape
            for _ in range(patches_per_image):
                y, x = random.randint(0, H - patch_size), random.randint(0, W - patch_size)
                self.patches.append((img[y:y+patch_size, x:x+patch_size], lbl[y:y+patch_size, x:x+patch_size]))

    def __len__(self): return len(self.patches)

    def __getitem__(self, idx):
        img, lbl = self.patches[idx]
        img = img.copy()

        # Création d'images artificielles par augmentation de données (flip, rotation, déformation, filtre, etc...)
        if self.augment:
            if random.random() > 0.5: img, lbl = img[:, ::-1].copy(), lbl[:, ::-1].copy()
            if random.random() > 0.5: img, lbl = img[::-1].copy(),    lbl[::-1].copy()
            k = random.randint(0, 3)
            img, lbl = np.rot90(img, k).copy(), np.rot90(lbl, k).copy()

            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                ip = Image.fromarray((img*255).astype(np.uint8))
                lp = Image.fromarray(lbl.astype(np.uint8))
                img = np.array(TF.rotate(ip, angle, TF.InterpolationMode.BILINEAR), dtype=np.float32) / 255.0
                lbl = np.array(TF.rotate(lp, angle, TF.InterpolationMode.NEAREST),  dtype=np.int64)

            if random.random() > 0.6:
                alpha, sigma, sh = random.uniform(20, 60), random.uniform(4, 6), img.shape
                dx = gaussian_filter(np.random.randn(*sh), sigma) * alpha
                dy = gaussian_filter(np.random.randn(*sh), sigma) * alpha
                gx, gy = np.meshgrid(np.arange(sh[1]), np.arange(sh[0]))
                cx, cy = np.clip(gx+dx, 0, sh[1]-1), np.clip(gy+dy, 0, sh[0]-1)
                img = map_coordinates(img,               [cy, cx], order=1, mode='reflect').astype(np.float32)
                lbl = map_coordinates(lbl.astype(float), [cy, cx], order=0, mode='reflect').astype(np.int64)

            if random.random() > 0.5:
                img = np.clip(img.mean() + random.uniform(0.6, 1.6) * (img - img.mean()), 0, 1).astype(np.float32)
            if random.random() > 0.5:
                img = np.clip(img + random.uniform(-0.15, 0.15), 0, 1).astype(np.float32)
            if random.random() > 0.5:
                img = np.power(np.clip(img, 1e-6, 1.0), random.uniform(0.6, 1.6)).astype(np.float32)
            if random.random() > 0.6:
                img = gaussian_filter(img, sigma=random.uniform(0.3, 1.2)).astype(np.float32)
            if random.random() > 0.5:
                img = np.clip(img + np.random.normal(0, random.uniform(0.01, 0.04), img.shape), 0, 1).astype(np.float32)

        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(lbl.copy())

def mean_iou(preds, masks):
    # Méthode de calcul de l'IoU moyen sur les classes présentes dans le batch
    ious = []
    for c in range(NUM_CLASSES):
        inter = ((preds == c) & (masks == c)).sum().item()
        union = ((preds == c) | (masks == c)).sum().item()
        if union > 0: ious.append(inter / union)
    return np.mean(ious)

def sliding_window_predict(model, img_np, patch_size=PATCH_SIZE, stride=192):
    # Prédiction pour image de taille arbitraire (sliding window avec vote)
    H, W = img_np.shape
    votes = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    model.eval()
    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))
    if ys[-1] + patch_size < H: ys.append(H - patch_size)
    if xs[-1] + patch_size < W: xs.append(W - patch_size)
    with torch.no_grad():
        for y in ys:
            for x in xs:
                t = torch.from_numpy(img_np[y:y+patch_size, x:x+patch_size]).unsqueeze(0).unsqueeze(0).to(DEVICE)
                prob = torch.softmax(model(t), dim=1).squeeze(0).cpu().numpy()
                votes[:, y:y+patch_size, x:x+patch_size] += prob
                count[y:y+patch_size, x:x+patch_size]    += 1
    return (votes / np.maximum(count, 1)).argmax(0).astype(np.int64)

def train():
    dataset = PatchDataset()
    n_val   = max(1, int(len(dataset) * VAL_SPLIT))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    val_ds.dataset.augment = False

    loader_kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader = DataLoader(val_ds,   shuffle=False, **loader_kw)

    model = UNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(DEVICE)
    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            t_loss, t_iou = 0.0, 0.0
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(DEVICE, non_blocking=True), masks.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                with torch.amp.autocast(DEVICE):
                    logits = model(imgs); loss = criterion(logits, masks)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                t_loss += loss.item(); t_iou += mean_iou(logits.argmax(1), masks)

            model.eval()
            v_loss, v_iou = 0.0, 0.0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(DEVICE, non_blocking=True), masks.to(DEVICE, non_blocking=True)
                    with torch.amp.autocast(DEVICE):
                        logits = model(imgs); v_loss += criterion(logits, masks).item()
                    v_iou += mean_iou(logits.argmax(1), masks)

            t_loss /= len(train_loader); t_iou /= len(train_loader)
            v_loss /= len(val_loader);   v_iou /= len(val_loader)
            scheduler.step()
            for k, v in zip(history, [t_loss, v_loss, t_iou, v_iou]): history[k].append(v)
            print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  train loss={t_loss:.4f} iou={t_iou:.4f}  val loss={v_loss:.4f} iou={v_iou:.4f}")

    except KeyboardInterrupt:
        print("\nInterrompu manuellement.")

    torch.save(model.state_dict(), WEIGHTS_PATH)

    # Courbes
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, k, title in [(ax1, "loss", "Cross-Entropy Loss"), (ax2, "iou", "Mean IoU")]:
        ax.plot(epochs, history[f"train_{k}"], label="Train")
        ax.plot(epochs, history[f"val_{k}"],   label="Val")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "training_curves.png")
    plt.savefig(out, dpi=150); plt.show(); print(f"Curves saved : {out}")

    predict_examples(model)

def predict_examples(model, n_samples=3):
    dirs = sorted(glob.glob(os.path.join(DATASET_DIR, "*")))[:n_samples]
    fig, axes = plt.subplots(n_samples, 3, figsize=(14, 4 * n_samples))
    if n_samples == 1: axes = [axes]
    for row, d in enumerate(dirs):
        img_np = np.array(Image.open(os.path.join(d, "img.png")).convert("L"), dtype=np.float32) / 255.0
        lbl_np = np.array(Image.open(os.path.join(d, "label.png")), dtype=np.int64)
        pred   = sliding_window_predict(model, img_np)
        for ax, data, title, kw in zip(axes[row],[img_np, CLASS_COLORS[lbl_np], CLASS_COLORS[pred]],["Image", "Réalité", "Prédiction"],[{"cmap": "gray"}, {}, {}]):
            ax.imshow(data, **kw); ax.set_title(title); ax.axis("off")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "predictions.png")
    plt.savefig(out, dpi=150); plt.show(); print(f"Predictions saved : {out}")


def predict_image(image_path, weights_path=WEIGHTS_PATH, save=True):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stem  = os.path.splitext(os.path.basename(image_path))[0]
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    img_np = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    pred   = sliding_window_predict(model, img_np)

    print("Résultat segmentation :")
    for cls, cnt in zip(*np.unique(pred, return_counts=True)):
        print(f"  {CLASS_NAMES[cls]:12s}: {cnt/pred.size*100:.2f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.imshow(img_np, cmap="gray"); ax1.set_title("Image originale"); ax1.axis("off")
    ax2.imshow(CLASS_COLORS[pred]);  ax2.set_title("Segmentation U-Net"); ax2.axis("off")
    ax2.legend(handles=[Patch(color=CLASS_COLORS[i]/255, label=CLASS_NAMES[i]) for i in range(NUM_CLASSES)],
               loc="lower right", fontsize=9)
    plt.tight_layout()
    if save:
        png_out = os.path.join(RESULTS_DIR, f"{stem}_segmented.png")
        npy_out = os.path.join(RESULTS_DIR, f"{stem}_seg.npy")
        plt.savefig(png_out, dpi=150); np.save(npy_out, pred)
        print(f"Segmentation PNG  : {png_out}\nSegmentation mask : {npy_out}")
    plt.show()
    return pred


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "predict":
        if len(sys.argv) < 3:
            print("Usage: python segmentation.py predict <image_path> [weights_path]"); sys.exit(1)
        predict_image(sys.argv[2], sys.argv[3] if len(sys.argv) >= 4 else WEIGHTS_PATH)
    else:
        train()

