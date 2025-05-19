import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
transform = transforms.Compose([
    transforms.Resize((64, 64), interpolation=torchvision.transforms.InterpolationMode.LANCZOS),  # Efficient resizing
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

processed_dataset = r"C:\Users\Guncha Anam\Downloads\Alcohol Detection (2)\Alcohol Detection\scripts\processed_dataset"

dataset = datasets.ImageFolder(root=processed_dataset, transform=transform)

# Class Mapping
print("Class Mapping:", dataset.class_to_idx)  # {'Alcohol': 0, 'Non-Alcohol': 1}


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")


batch_size = 16
num_workers = 0
pin_memory = False

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

print("✅ Data loaders created successfully!")

def imshow(img):
    img = img * 0.5 + 0.5
    npimg = img.numpy().transpose((1, 2, 0))
    plt.imshow(npimg)
    plt.axis("off")
    plt.show()

dataiter = iter(train_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images[:8]))
print("Labels:", labels[:8].numpy())
