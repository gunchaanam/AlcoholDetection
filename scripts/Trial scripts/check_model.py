import torch

# ✅ Load the saved model
checkpoint = torch.load("../models/alcohol_capsnet_epoch_3.pth")

# ✅ Print all saved keys (should include 'fc1.weight' and 'fc1.bias')
print("Saved Model Keys:", checkpoint.keys())

# ✅ Check if fc1.weight and fc1.bias exist
if "fc1.weight" in checkpoint and "fc1.bias" in checkpoint:
    print("✅ fc1.weight and fc1.bias are properly saved!")
else:
    print("❌ fc1.weight or fc1.bias is missing!")
