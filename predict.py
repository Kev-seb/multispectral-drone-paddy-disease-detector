import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

# -------------------------
# Load Crop Class Map
# -------------------------
class_map = json.load(open("models/class_map.json"))
idx_to_class = {v: k for k, v in class_map.items()}

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load Crop Model
# -------------------------
crop_model = models.efficientnet_b0(weights=None)

in_features = crop_model.classifier[1].in_features
crop_model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Linear(256, len(class_map))
)

crop_model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
crop_model.to(device)
crop_model.eval()

# -------------------------
# Load BLB Disease Model
# -------------------------
blb_model = models.efficientnet_b0(weights=None)

in_features = blb_model.classifier[1].in_features
blb_model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Linear(256, 2)  # healthy, leaf_blight
)

blb_model.load_state_dict(torch.load("models/blb_model.pth", map_location=device))
blb_model.to(device)
blb_model.eval()

# BLB classes
blb_classes = ["Healthy", "Leaf Blight"]

# -------------------------
# Image Preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def preprocess(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

# -------------------------
# Prediction Function
# -------------------------
def predict_image(img_path):
    img = preprocess(img_path).to(device)

    # ---- Crop Prediction ----
    with torch.no_grad():
        crop_outputs = crop_model(img)
        crop_probs = torch.softmax(crop_outputs, dim=1)
        crop_conf, crop_idx = torch.max(crop_probs, dim=1)

    crop_name = idx_to_class[crop_idx.item()]
    crop_confidence = crop_conf.item() * 100

    print(f"[DEBUG] Detected Crop: {crop_name}")

    # ---- Disease only if Paddy/Rice ----
    if crop_name.lower() in ["rice", "paddy"]:
        with torch.no_grad():
            blb_outputs = blb_model(img)
            blb_probs = torch.softmax(blb_outputs, dim=1)
            disease_conf, disease_idx = torch.max(blb_probs, dim=1)

        disease_name = blb_classes[disease_idx.item()]
        disease_confidence = disease_conf.item() * 100

        return f"""
Crop: Paddy ({crop_confidence:.2f}%)
Disease: {disease_name} ({disease_confidence:.2f}%)
"""

    # ---- If not rice ----
    return f"""
Crop: {crop_name} ({crop_confidence:.2f}%)
Disease: Not Applicable (Non-paddy crop)
"""

# -------------------------
# Run Test
# -------------------------
if __name__ == "__main__":
    img_path = "data/blb/train/leaf_blight/image_patch_43.jpg" # change to your test image
    result = predict_image(img_path)
    print(result)