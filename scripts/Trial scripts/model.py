import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleNetwork(nn.Module):
    def __init__(self):
        super(CapsuleNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 32 * 8, kernel_size=9, stride=2, padding=0)

        # Placeholder for fc1 input size
        self.fc1_input_features = None
        self.fc1 = None  # This will be defined properly in forward()
        self.fc2 = nn.Linear(512, 2)  # Output layer (2 classes: Alcohol / No Alcohol)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Get shape before flattening
        if self.fc1 is None:
            self.fc1_input_features = x.view(x.size(0), -1).shape[1]
            print(f"📌 Shape before Flatten: {x.shape}")
            print(f"📌 Flattened Shape: {self.fc1_input_features}")

            # ✅ Register fc1 properly so it gets saved & loaded correctly
            self.fc1 = nn.Linear(self.fc1_input_features, 512).to(x.device)
            self.add_module("fc1", self.fc1)  # Ensures `fc1.weight` & `fc1.bias` are saved

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ✅ Test Model
if __name__ == "__main__":
    model = CapsuleNetwork()
    test_input = torch.randn(1, 1, 128, 128)  # Example input
    output = model(test_input)
    print("Output Shape:", output.shape)  # Should be [1, 2]
