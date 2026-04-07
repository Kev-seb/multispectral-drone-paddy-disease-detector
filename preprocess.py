import os
import shutil
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
RAW_DIR    = Path("data/raw/Agricultural-crops")  # ✅ FIXED PATH
OUTPUT_DIR = Path("data/processed")
IMG_SIZE   = (224, 224)
SPLIT      = (0.70, 0.15, 0.15)
SEED       = 42

SPLITS = ["train", "val", "test"]


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────

def create_output_dirs(class_dirs):
    for split in SPLITS:
        for cls_dir in class_dirs:
            cls_name = cls_dir.name.lower().strip()
            (OUTPUT_DIR / split / cls_name).mkdir(parents=True, exist_ok=True)
    print("[✓] Output directories created dynamically")


def load_and_resize(img_path: Path):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        return img
    except (UnidentifiedImageError, Exception):
        return None


def split_and_copy(class_dir: Path, cls_name: str):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = [p for p in class_dir.iterdir() if p.suffix in exts]
    random.shuffle(images)

    n  = len(images)
    t  = int(n * SPLIT[0])
    v  = int(n * SPLIT[1])

    buckets = {
        "train": images[:t],
        "val":   images[t:t+v],
        "test":  images[t+v:]
    }

    counts = {}
    for split, files in buckets.items():
        ok = 0
        for img_path in tqdm(files, desc=f"{split}/{cls_name}", leave=False):
            img = load_and_resize(img_path)
            if img is None:
                continue

            out_path = OUTPUT_DIR / split / cls_name / img_path.name
            img.save(out_path, quality=95)
            ok += 1

        counts[split] = ok

    return counts


# ─────────────────────────────────────────
#  NDVI SIMULATION
# ─────────────────────────────────────────

def compute_ndvi_rgb(img_array):
    img = img_array.astype(np.float32)
    R = img[:, :, 0]
    G = img[:, :, 1]

    denom = G + R
    denom[denom == 0] = 1

    ndvi = (G - R) / denom
    return ndvi


def save_ndvi_sample():
    sample_path = None

    for split in SPLITS:
        candidates = list((OUTPUT_DIR / split).rglob("*.jpg"))
        if candidates:
            sample_path = candidates[0]
            break

    if sample_path is None:
        print("[!] No image found for NDVI sample — skipping")
        return

    img_array = np.array(Image.open(sample_path))
    ndvi = compute_ndvi_rgb(img_array)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(img_array)
    axes[0].set_title("Original RGB")
    axes[0].axis("off")

    im = axes[1].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1].set_title("Simulated NDVI")
    axes[1].axis("off")

    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()

    out = OUTPUT_DIR / "ndvi_sample.png"
    plt.savefig(out)
    plt.close()

    print(f"[✓] NDVI sample saved → {out}")


# ─────────────────────────────────────────
#  CLASS DISTRIBUTION
# ─────────────────────────────────────────

def plot_class_distribution(summary):
    classes = list(summary.keys())

    train_counts = [summary[c].get("train", 0) for c in classes]
    val_counts   = [summary[c].get("val", 0) for c in classes]
    test_counts  = [summary[c].get("test", 0) for c in classes]

    x = np.arange(len(classes))
    w = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar(x - w, train_counts, width=w, label="Train")
    plt.bar(x, val_counts, width=w, label="Val")
    plt.bar(x + w, test_counts, width=w, label="Test")

    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.title("Class Distribution")

    out = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(out)
    plt.close()

    print(f"[✓] Class distribution saved → {out}")


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────

def main():
    random.seed(SEED)

    print("\n==== MULTISPECTRAL DRONE AI — PREPROCESSING ====\n")

    if not RAW_DIR.exists():
        print(f"[ERROR] Raw data folder not found: {RAW_DIR}")
        return

    class_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]

    if not class_dirs:
        print("[ERROR] No class folders found")
        return

    create_output_dirs(class_dirs)

    summary = {}

    print(f"\nFound {len(class_dirs)} classes\n")

    for cls_dir in class_dirs:
        cls_name = cls_dir.name.lower().strip()

        print(f"Processing: {cls_name}")
        counts = split_and_copy(cls_dir, cls_name)

        summary[cls_name] = counts

        print(f"train={counts['train']} val={counts['val']} test={counts['test']}")

    json.dump(summary, open(OUTPUT_DIR / "split_summary.json", "w"), indent=2)

    save_ndvi_sample()
    plot_class_distribution(summary)

    print("\n==== DONE ====")


if __name__ == "__main__":
    main()