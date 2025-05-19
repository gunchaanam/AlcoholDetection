import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter

# Define dataset path (Adjust if needed)
DATASET_PATH = os.path.abspath("../dataset")

# Define transformations for preprocessing images
transform = A.Compose([
    A.Resize(128, 128),  # Resize images to 128x128

    # 🔹 Stronger Data Augmentation (Helps No-Alcohol Class)
    A.HorizontalFlip(p=0.8),  # Flip more often
    A.RandomBrightnessContrast(p=0.5),  # More contrast variation
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), rotate=(-30, 30), p=0.7),
    A.GaussianBlur(p=0.3),  # Adds slight blur to generalize better

    A.Normalize(mean=[0.5], std=[0.5]),  # Normalize grayscale images
    ToTensorV2(),  # Convert to PyTorch tensor
])

# Custom Dataset Class
class AlcoholDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Iterate over dataset folders (Grupo_0, Grupo_1, etc.)
        for group in os.listdir(dataset_path):
            group_path = os.path.join(dataset_path, group)
            if not os.path.isdir(group_path):
                continue

            for subject in os.listdir(group_path):
                subject_path = os.path.join(group_path, subject)
                if not os.path.isdir(subject_path):
                    continue

                for sensor in os.listdir(subject_path):  # LG, IriTech
                    sensor_path = os.path.join(subject_path, sensor)
                    if not os.path.isdir(sensor_path):
                        continue

                    for interval in os.listdir(sensor_path):  # 0, 15, 30, etc.
                        interval_path = os.path.join(sensor_path, interval)
                        if not os.path.isdir(interval_path):
                            continue

                        for folder in os.listdir(interval_path):  # E_x_x_x_R_x
                            folder_path = os.path.join(interval_path, folder)
                            if not os.path.isdir(folder_path):
                                continue

                            for file in os.listdir(folder_path):
                                if file.endswith(".bmp"):  # ✅ Only process images
                                    image_path = os.path.join(folder_path, file)
                                    self.image_paths.append(image_path)

                                    # ✅ Improved Labeling: 0 min → No Alcohol, 15-60 min → Alcohol
                                    label = 0 if interval == "0" else 1
                                    self.labels.append(label)

        # ✅ Debugging: Check dataset balance
        self._check_balance()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 🔹 Try to read the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 🔥 If image is None (corrupted), return a black image instead of infinite recursion
        if img is None:
            print(f"⚠️ Skipping corrupted image: {img_path}")
            img = np.zeros((128, 128), dtype=np.uint8)  # Return black image

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, torch.tensor(label, dtype=torch.long)

    def _check_balance(self):
        """Checks class distribution in the dataset"""
        label_counts = Counter(self.labels)
        print(f"🔹 Dataset Label Distribution: {label_counts}")

# ✅ Load full dataset
full_dataset = AlcoholDataset(DATASET_PATH, transform)

# ✅ Print Dataset Stats Before Splitting
print(f"🔹 Total Dataset Size: {len(full_dataset)}")
label_counts = Counter(full_dataset.labels)
print(f"🔹 Dataset Label Distribution Before Split: {label_counts}")

# ✅ Split dataset: 70% train, 30% test
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# ✅ Use Weighted Sampling to Balance the Training Dataset
train_labels = [full_dataset.labels[i] for i in train_dataset.indices]
class_counts = Counter(train_labels)
num_samples = sum(class_counts.values())

# 🔹 Compute Weights (Higher weight for smaller class)
class_weights = {label: num_samples / count for label, count in class_counts.items()}
weights = [class_weights[label] for label in train_labels]

# 🔹 Create Weighted Sampler
sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)

# ✅ Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)  # Use weighted sampler
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"✅ Dataset split: {train_size} training images, {test_size} testing images")
