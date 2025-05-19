import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from model import CapsuleNetwork
from data_loader import test_loader  # ✅ Using test_loader

# ✅ Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CapsuleNetwork().to(device)

# ✅ Load checkpoint with strict=False to ignore mismatches
checkpoint_path = "../models/alcohol_capsnet_epoch_3.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

# 🔍 Debugging: Print saved model keys
print("🔍 Saved Model Keys:", checkpoint.keys())

# ✅ Load model state with strict=False to bypass errors
model.load_state_dict(checkpoint, strict=False)
model.eval()

# ✅ Check dataset class distribution
label_counts = Counter([label for _, label in test_loader.dataset])
print(f"🔹 Dataset Label Distribution: {label_counts}")

# ✅ Evaluate on 30% dataset
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:  # ✅ Testing on 30% dataset
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ✅ Compute Accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"✅ Model Accuracy on 30% dataset: {accuracy:.4f}")

# ✅ Generate Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", conf_matrix)

# ✅ Generate Classification Report
print("Classification Report:\n", classification_report(all_labels, all_preds))

# 🔹 Identify Misclassified Images
misclassified_indices = [i for i in range(len(all_labels)
                                          ) if all_preds[i] != all_labels[i]]

# 🔹 Show First 5 Misclassified Images for Debugging
for idx in misclassified_indices[:5]:
    # ✅ Get the original dataset reference
    original_dataset = test_loader.dataset.dataset

    # ✅ Get correct image path
    img_path = original_dataset.image_paths[test_loader.dataset.indices[idx]]

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    plt.imshow(img, cmap="gray")
    plt.title(f"Misclassified as: {all_preds[idx]} (Actual: {all_labels[idx]})")
    plt.show()
