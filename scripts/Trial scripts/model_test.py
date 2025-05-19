import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleNetwork(nn.Module):
    def __init__(self):
        super(CapsuleNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 32 * 8, kernel_size=9, stride=2, padding=0)

        # Fully Connected Layers (Fixed size)
        self.fc1 = nn.Linear(4608, 512)  # ✅ Fixed input size
        self.fc2 = nn.Linear(512, 2)  # 2 classes (Alcohol / No Alcohol)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ✅ Test Model
if __name__ == "__main__":
    model = CapsuleNetwork()
    test_input = torch.randn(1, 1, 128, 128)  # Example input
    output = model(test_input)
    print("Output Shape:", output.shape)
