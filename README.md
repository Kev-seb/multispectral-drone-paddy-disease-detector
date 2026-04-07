## ⚙️ How This Project Works (Detailed Pipeline)

This project implements a **two-stage deep learning pipeline** for crop classification and disease detection using UAV imagery.

The system is designed to mimic a **real-world smart agriculture workflow**.

---

## 🔹 Stage 1: Image Input

The system accepts an image from various sources:

- UAV (Drone) imagery
- Field-level RGB images
- Dataset images (.jpg / .png)
- Multispectral images (.tif)

Example:

test.jpg


---

## 🔹 Stage 2: Image Preprocessing

Before feeding the image into the model, preprocessing is applied:

- Resize image to **224 × 224 pixels**
- Convert to **RGB format**
- Normalize pixel values
- Convert to tensor for PyTorch

This ensures compatibility with EfficientNet architecture.

---

## 🔹 Stage 3: Crop Classification Model

The preprocessed image is passed into the **Crop Classification Model**.

### Model Details:
- Architecture: EfficientNet-B0
- Framework: PyTorch
- Task: Binary Classification

### Classes:
- Rice (Paddy)
- Others (All non-rice crops)

### Output:

Crop: Rice (92.3%)


---

## 🔹 Stage 4: Decision Logic (Pipeline Control)

This is a critical step in the system.

The output from the crop model is evaluated:

- If crop = **Rice** → proceed to disease detection
- If crop = **Others** → stop pipeline

### Why this step?

- Avoids unnecessary computation
- Improves accuracy
- Reduces false disease predictions

---

## 🔹 Stage 5: Disease Detection Model (BLB)

If the crop is identified as rice, the image is passed into the **Disease Detection Model**.

### Model Details:
- Architecture: EfficientNet-B0
- Task: Binary Classification

### Classes:
- Healthy
- Leaf Blight

### Output:

Disease: Leaf Blight (88.5%)


---

## 🔹 Stage 6: Final Output Generation

The system combines both outputs:


Crop: Rice (92.3%)
Disease: Leaf Blight (88.5%)


If crop is not rice:


Crop: Wheat (95.1%)
Disease: Not Applicable


---

## 🔄 Complete Workflow

      Input Image
           │
           ▼
    Preprocessing
           │
           ▼

Crop Classification Model
(Rice / Others)
│
┌─────────┴─────────┐
│ │
▼ ▼
Rice Others
│ │
▼ ▼
Disease Detection STOP
(BLB Model)
│
▼
Final Output


---

## 🧠 Why Two-Stage Architecture?

This system uses **two separate models instead of one**, which provides:

- Better accuracy
- Reduced class confusion
- Modular design
- Easier scalability

This design is commonly used in **real-world AI pipelines**.

---

## 🔄 Multispectral Data Handling

The original dataset contains **.tif (multispectral) images**, which are not directly usable.

### Conversion Process:

- Read `.tif` using `tifffile`
- Extract usable channels
- Normalize pixel values
- Convert to `.jpg`

Script used:

python convert_blb.py


---

## 🧪 Training Pipeline

### Crop Model:
- Dataset: Rice vs Others
- Loss: CrossEntropyLoss
- Optimizer: Adam

### Disease Model:
- Dataset: Healthy vs Leaf Blight

Training commands:

python train.py
python train_blb.py


---

## ⚡ Inference Pipeline

Prediction is done using:


python predict.py


Steps:
1. Load trained models
2. Preprocess input image
3. Predict crop
4. If rice → predict disease
5. Display result

---

## 🌍 Real-World Application Flow

1. Drone captures field images
2. Images are processed by AI model
3. Crop type is identified
4. Disease is detected early
5. Farmer takes action
