# ==========================================================================
# 📦 1. Import Libraries
# ==========================================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import time

# ==========================================================================
# 📎 2. Set Device
# ==========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================================
# 🛉 3. Data Loading and Preprocessing
# ==========================================================================
transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=torchvision.transforms.InterpolationMode.LANCZOS),
    transforms.Grayscale(),  # Convert to single channel
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

data_path = r"C:\\PycharmProjects\\Alcohol Detection\\processed_dataset"
dataset = datasets.ImageFolder(root=data_path, transform=transform)
print("Class Mapping:", dataset.class_to_idx)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("✅ Data loaders created successfully!")


# ==========================================================================
# 🌀 4. Squash Activation Function (for Capsule Vectors)
# ==========================================================================
def squash(tensor, dim=-1):
    norm = torch.norm(tensor, dim=dim, keepdim=True)
    scale = (norm ** 2) / (1 + norm ** 2)
    return scale * tensor / (norm + 1e-8)


# ==========================================================================
# 🧱 5. Capsule Network Architecture
# ==========================================================================

# 5.1 Primary Capsules Layer
class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, out_capsules, capsule_dim, kernel_size, stride):
        super(PrimaryCapsules, self).__init__()
        self.out_capsules = out_capsules
        self.capsule_dim = capsule_dim
        self.capsules = nn.Conv2d(in_channels, out_capsules * capsule_dim, kernel_size, stride)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.capsules(x)
        out = out.view(batch_size, self.out_capsules, self.capsule_dim, -1)
        out = out.permute(0, 3, 1, 2).contiguous()
        out = out.view(batch_size, -1, self.capsule_dim)
        return squash(out)


# 5.2 Digit Capsules Layer (Dynamic Routing)
class DigitCapsules(nn.Module):
    def __init__(self, num_capsules, in_dim, out_dim, num_routes, routing_iters=3):
        super(DigitCapsules, self).__init__()
        self.num_capsules = num_capsules
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_routes = num_routes
        self.routing_iters = routing_iters
        self.route_weights = nn.Parameter(torch.randn(num_routes, num_capsules, in_dim, out_dim))

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(2).unsqueeze(3)
        weights = self.route_weights.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        u_hat = torch.matmul(x, weights).squeeze(3)
        b_ij = torch.zeros(batch_size, self.num_routes, self.num_capsules, 1, device=x.device)

        for _ in range(self.routing_iters):
            c_ij = torch.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j, dim=-1)
            b_ij = b_ij + (u_hat * v_j).sum(dim=-1, keepdim=True)

        v_j = v_j.squeeze(1)
        return v_j


# 5.3 Full Capsule Network (F-CapsNet)
class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.primary_capsules = PrimaryCapsules(in_channels=256, out_capsules=32, capsule_dim=8, kernel_size=9, stride=2)
        self.digit_capsules = DigitCapsules(num_capsules=2, in_dim=8, out_dim=16, num_routes=100352, routing_iters=3)
        self.decoder = nn.Sequential(
            nn.Linear(16 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        x = x.view(x.size(0), -1)
        out = self.decoder(x)
        return out


# ==========================================================================
# 🌟 6. Training and Evaluation Functions
# ==========================================================================
def train_model(model, loader, epochs=10, save_every=5, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Time: {time.time() - epoch_start:.2f}s")

        # Save model checkpoint every few epochs
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"capsnet_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"📦 Model checkpoint saved at: {checkpoint_path}")

    print(f"Total Training Time: {time.time() - start_time:.2f}s")


def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"🔍 Accuracy: {acc * 100:.2f}%")
    print("🧾 Confusion Matrix:\n", cm)
    return acc, cm


# ==========================================================================
# 🚀 7. Execution
# ==========================================================================
model = CapsuleNet().to(device)
print("🚦 Starting training...")
train_model(model, train_loader, epochs=50, save_every=5)
print("✅ Training completed!")
print("📊 Evaluating model on test set...")
evaluate_model(model, test_loader)
# ==========================================================================
# 💾 8. Save Final Model
# ==========================================================================
final_model_path = os.path.join("checkpoints", "capsnet_final.pt")
torch.save(model.state_dict(), final_model_path)
print(f"✅ Final model saved at: {final_model_path}")

# ==========================================================================
# ♻️ 9. Load Model (Optional for Inference or Resume Training)
# ==========================================================================
def load_model(path, device):
    model = CapsuleNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"📂 Loaded model from: {path}")
    return model

# ==========================================================================
# 🧪 10. Inference on a Single Image
# ==========================================================================
from PIL import Image

def predict_image(image_path, model, transform):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    pred_class = output.argmax(dim=1).item()
    class_map = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_map.items()}
    print(f"🔍 Prediction: {idx_to_class[pred_class]}")
    return idx_to_class[pred_class]

# Example usage:
# model = load_model("checkpoints/capsnet_final.pt", device)
# prediction = predict_image("path_to_image.bmp", model, transform)
