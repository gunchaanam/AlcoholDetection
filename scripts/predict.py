import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
import os

def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * x / torch.sqrt(squared_norm + 1e-8)

class PrimaryCapsules(torch.nn.Module):
    def __init__(self, in_channels, out_capsules, capsule_dim, kernel_size=9, stride=2):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_capsules * capsule_dim, kernel_size, stride)
        self.out_capsules = out_capsules
        self.capsule_dim = capsule_dim

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        _, _, h, w = x.size()
        x = x.view(batch_size, self.out_capsules, self.capsule_dim, h * w)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, -1, self.capsule_dim)
        return squash(x)

class DigitCapsules(torch.nn.Module):
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
            self.W = torch.nn.Parameter(0.01 * torch.randn(1, in_capsules, self.out_capsules, self.out_dim, self.in_dim, device=x.device))
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

class FusedCapsNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
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
model.load_state_dict(torch.load("saved_models/fused_capsnet.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

if len(sys.argv) != 2:
    print("Usage: python predict.py path_to_image.jpg")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Image '{image_path}' not found!")
    sys.exit(1)

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

class_mapping = {0: "alcoholic", 1: "non-alcoholic"}  # Adjust this based on your folder structure
print(f"Prediction: {class_mapping[predicted_class]}")
