import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((64, 64), interpolation=torchvision.transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

data_path = r"C:\Users\Guncha Anam\Downloads\Alcohol Detection (2)\Alcohol Detection\processed_dataset"
dataset = datasets.ImageFolder(root=data_path, transform=transform)

# Class Mapping (e.g., {'Alcohol': 0, 'Non-Alcohol': 1})
print("Class Mapping:", dataset.class_to_idx)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

batch_size = 16  # Adjust for memory efficiency
num_workers = 0  # Set to 0 for CPU
pin_memory = False  # Only needed for GPU

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                         pin_memory=pin_memory)
print("✅ Data loaders created successfully!")

class AlcoholClassifier(nn.Module):
    def __init__(self):
        super(AlcoholClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # After 3 pooling layers, 64x64 becomes 8x8.
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Initialize model, loss function, and optimizer
model = AlcoholClassifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



def train_model(model, train_loader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")


train_model(model, train_loader, num_epochs=5)



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=list(dataset.class_to_idx.keys()))

    print(f"✅ Test Accuracy: {accuracy * 100:.2f}% on {len(all_labels)} test images")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)


evaluate_model(model, test_loader)

save_dir = r"C:\PycharmProjects\Alcohol Detection\saved_models"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "alcohol_classifier.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved at: {model_path}")



def imshow(img_tensor):
    # Denormalize the image
    img = img_tensor * 0.5 + 0.5
    npimg = img.numpy().transpose((1, 2, 0))
    plt.imshow(npimg, cmap='gray')
    plt.axis("off")
    plt.show()


# Display a few images from the training set
dataiter = iter(train_loader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images[:8]))
print("Labels:", labels[:8].numpy())
