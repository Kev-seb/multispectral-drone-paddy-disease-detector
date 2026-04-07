import os
import numpy as np
import tifffile as tiff
from PIL import Image

# Paths
image_dir = "data/raw/D1/train"
mask_dir = "data/raw/D1/train_labels"

healthy_dir = "data/blb/train/healthy"
disease_dir = "data/blb/train/leaf_blight"

os.makedirs(healthy_dir, exist_ok=True)
os.makedirs(disease_dir, exist_ok=True)

count = 0

for img_name in os.listdir(image_dir):
    if not img_name.endswith(".tif"):
        continue

    img_path = os.path.join(image_dir, img_name)

    # Match label
    mask_name = img_name.replace("image", "label")
    mask_path = os.path.join(mask_dir, mask_name)

    if not os.path.exists(mask_path):
        print(f"❌ Mask not found: {mask_name}")
        continue

    try:
        # Read mask using tifffile
        mask = tiff.imread(mask_path)

        # Check disease
        if np.any(mask > 0):
            save_folder = disease_dir
        else:
            save_folder = healthy_dir

        # Read image
        img = tiff.imread(img_path)

        # Normalize image (important!)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = (img * 255).astype(np.uint8)

        # Convert to RGB if needed
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[-1] > 3:
            img = img[:, :, :3]

        # Save as JPG
        new_name = img_name.replace(".tif", ".jpg")
        save_path = os.path.join(save_folder, new_name)

        Image.fromarray(img).save(save_path)

        count += 1

    except Exception as e:
        print(f"❌ Error processing {img_name}: {e}")

print(f"\n✅ Done! Converted {count} images.")