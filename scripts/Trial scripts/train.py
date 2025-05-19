import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys
from model import CapsuleNetwork
from data_loader import train_loader

# 🔹 Force PyTorch to use GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8)  # Use all CPU cores for faster training
print(f"🚀 Using device: {device}")  # Should print "cuda:0" if GPU is in use

# Load model and move to device
model = CapsuleNetwork().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🔹 Check if a previously trained model exists
last_epoch = 0
model_path = "../models/last_epoch.txt"

if os.path.exists(model_path):
    with open(model_path, "r") as f:
        last_epoch = int(f.read().strip())

    checkpoint_file = f"../models/alcohol_capsnet_epoch_{last_epoch}.pth"

    if os.path.exists(checkpoint_file):
        model.load_state_dict(torch.load(checkpoint_file, map_location=device))
        print(f"🔄 Resuming training from Epoch {last_epoch + 1}", flush=True)
    else:
        print("⚠️ Checkpoint file not found, starting from scratch.", flush=True)
else:
    print("🆕 No previous training found, starting from scratch.", flush=True)

# Resume training from last completed epoch
num_epochs = 3  # ✅ Training for 3 epochs
checkpoint_interval = 2  # Save checkpoint every 2 epochs

for epoch in range(last_epoch, num_epochs):  # Start from last epoch
    total_loss = 0
    correct = 0
    total = 0

    print(f"\n🚀 Starting Epoch {epoch + 1}/{num_epochs}...", flush=True)

    for batch_idx, (images, labels) in enumerate(train_loader):
        # 🔹 Move data to CPU/GPU
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # ✅ Print progress every 10 batches & force immediate output
        if batch_idx % 10 == 0:
            print(f"✅ Epoch {epoch + 1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss:.4f}",
                  flush=True)
            sys.stdout.flush()

    # Save last completed epoch
    with open(model_path, "w") as f:
        f.write(str(last_epoch + 1))

    # ✅ Correctly save the model as a .pth file (instead of .txt)
    checkpoint_file = f"../models/alcohol_capsnet_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), checkpoint_file)
    print(f"✅ Model saved: {checkpoint_file}", flush=True)

print("\n🎉 Training Complete! Model is saved.", flush=True)
