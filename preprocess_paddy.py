import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_DIR = "data/raw/paddy_doctor"
OUTPUT_DIR = "data/processed/paddy"
IMG_SIZE = (224, 224)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# LOAD CSV
# ─────────────────────────────────────────
csv_path = os.path.join(DATA_DIR, "train.csv")

if not os.path.exists(csv_path):
    print(f"[ERROR] CSV not found at {csv_path}")
    exit()

df = pd.read_csv(csv_path)

print("\n==== PADDY DATA PREPROCESSING ====\n")
print("CSV loaded successfully")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())
print("\nSample data:\n", df.head())

# Detect correct column name
if "image_id" in df.columns:
    IMAGE_COL = "image_id"
elif "image" in df.columns:
    IMAGE_COL = "image"
else:
    print("[ERROR] No valid image column found!")
    exit()

LABEL_COL = "label"

# ─────────────────────────────────────────
# CREATE CLASS FOLDERS
# ─────────────────────────────────────────
classes = df[LABEL_COL].unique()
print("\nClasses found:", classes)

for label in classes:
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

# ─────────────────────────────────────────
# PROCESS IMAGES
# ─────────────────────────────────────────
success = 0
failed = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_name = row[IMAGE_COL]
    label = row[LABEL_COL]

    img_path = os.path.join(DATA_DIR, "train_images", img_name)

    if not os.path.exists(img_path):
        failed += 1
        continue

    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMG_SIZE)

        save_path = os.path.join(OUTPUT_DIR, label, img_name)
        img.save(save_path)

        success += 1

    except Exception as e:
        failed += 1

# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────
print("\n==== DONE ====")
print(f"Images processed: {success}")
print(f"Failed/skipped: {failed}")
print(f"Output folder: {OUTPUT_DIR}")