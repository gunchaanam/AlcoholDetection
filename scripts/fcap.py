import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

data_path = r"C:\Users\Guncha Anam\Downloads\Alcohol Detection (2)\Alcohol Detection\processed_dataset1"
dataset = datasets.ImageFolder(root=data_path, transform=transform)
print("Class Mapping:", dataset.class_to_idx)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + 1e-8)

class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, out_capsules, capsule_dim, kernel_size=9, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_capsules * capsule_dim, kernel_size, stride)
        self.out_capsules = out_capsules
        self.capsule_dim = capsule_dim

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        _, _, h, w = x.size()
        self.num_capsules = h * w * self.out_capsules
        x = x.view(batch_size, self.out_capsules, self.capsule_dim, h * w)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, -1, self.capsule_dim)
        return squash(x)

class DigitCapsules(nn.Module):
    def __init__(self, in_dim, out_capsules, out_dim, num_routes=3):
        super().__init__()
        self.num_routes = num_routes
        self.out_capsules = out_capsules
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.W = None

    def forward(self, x):
        batch_size, in_capsules, _ = x.size()
        if self.W is None or self.W.size(1) != in_capsules:
            self.W = nn.Parameter(0.01 * torch.randn(1, in_capsules, self.out_capsules, self.out_dim, self.in_dim, device=x.device))

        x = x.unsqueeze(2).unsqueeze(4)
        W = self.W.expand(batch_size, -1, -1, -1, -1)
        u_hat = torch.matmul(W, x).squeeze(-1)
        b_ij = torch.zeros(batch_size, in_capsules, self.out_capsules, device=x.device)
        for _ in range(self.num_routes):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)
            v_j = squash(s_j, dim=-1)
            if _ < self.num_routes - 1:
                b_ij = b_ij + (u_hat * v_j.unsqueeze(1)).sum(dim=-1)

        return v_j

class FusedCapsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.primary_capsules = PrimaryCapsules(in_channels=128, out_capsules=8, capsule_dim=8)
        self.digit_capsules = DigitCapsules(in_dim=8, out_capsules=2, out_dim=16)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        x = torch.norm(x, dim=-1)
        return F.log_softmax(x, dim=-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusedCapsNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

def train_model(model, train_loader, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

train_model(model, train_loader)

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(f"Accuracy: {accuracy_score(all_labels, all_preds) * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

evaluate_model(model, test_loader)
output_dir = "results"  # or some valid path
os.makedirs(output_dir, exist_ok=True)

torch.save(model.state_dict(), "saved_models/fused_capsnet.pth")
print("Model saved to 'saved_models/fused_capsnet.pth'")
