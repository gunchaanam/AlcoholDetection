import torch
import cv2
import numpy as np
from model import CapsuleNetwork

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CapsuleNetwork().to(device)

# ✅ Load existing trained model (ignoring missing layers)
state_dict = torch.load("../models/alcohol_capsnet_epoch_3.pth", map_location=device)
filtered_state_dict = {k: v for k, v in state_dict.items() if "fc1" not in k}  # Ignore fc1 if missing
model.load_state_dict(filtered_state_dict, strict=False)
model.eval()

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Error: Could not read image {image_path}")
        return None

    img = cv2.resize(img, (128, 128))  # Resize to match model input
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
    return img

# Test on a sample image
image_path = "../test_loader/E_0_0_0_L_M_N_N_1981_0_2017.bmp"  # ✅ Replace with actual test image path
img = preprocess_image(image_path)

if img is not None:
    output = model(img)
    prediction = torch.argmax(output).item()
    print(f"🔹 Prediction: {'Alcohol' if prediction == 1 else 'No Alcohol'}")
else:
    print("⚠️ No valid image found. Please check the file path.")
