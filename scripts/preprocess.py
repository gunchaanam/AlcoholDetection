import os
import pandas as pd
import re
from torchvision import transforms
from PIL import Image

# Paths
dataset_path = r"E:\Subset_Iris_under_alcohol_ICPRv2 (1)\data"
metadata_csv = r"C:\Users\Guncha Anam\Downloads\Alcohol Detection (2)\Alcohol Detection\scripts\metadata.csv"  # Corrected path
processed_dataset = r"C:\Users\Guncha Anam\Downloads\Alcohol Detection (2)\Alcohol Detection\processed_dataset1"

# Create processed dataset folders if they don't exist
os.makedirs(processed_dataset, exist_ok=True)
for label in ["Alcohol", "Non-Alcohol"]:
    os.makedirs(os.path.join(processed_dataset, label), exist_ok=True)

# Load metadata
metadata = pd.read_csv(metadata_csv)


# Function to clean image names
def clean_image_name(image_name):
    """Fixes file extension issues by keeping only the valid extension."""
    return re.sub(r'(\.bmp|\.jpg|\.png).*', r'\1', image_name)  # Keeps only the correct extension


# Define transformation: Resize images to 64x64 with efficient LANCZOS filter
resize_transform = transforms.Compose([
    transforms.Resize((64, 64), Image.LANCZOS)  # High-quality downscaling
])

# Process images
missing_files = 0  # Counter for missing files
processed_count = 0  # Counter for successfully processed files

for index, row in metadata.iterrows():
    image_name = clean_image_name(row["Image_Name"])  # Clean filename
    label, json_path = row["Label"], row["JSON_Path"]

    # Construct actual image path based on JSON location
    image_path = os.path.join(os.path.dirname(json_path), image_name)

    if os.path.exists(image_path):  # Ensure the image exists
        try:
            # Load and preprocess the image
            img = Image.open(image_path).convert("L")  # 'L' for single-channel grayscale
            img = resize_transform(img)  # Resize to 64x64

            # Destination folder
            dest_folder = os.path.join(processed_dataset, label)
            dest_path = os.path.join(dest_folder, image_name)

            img.save(dest_path)  # Save processed image
            processed_count += 1  # Increment processed counter

        except Exception as e:
            print(f"⚠️ Error processing {image_path}: {e}")  # Handle file read errors

    else:
        print(f"❌ Missing image: {image_path}")  # Debugging message
        missing_files += 1  # Increment missing file counter

print(f"\n✅ Preprocessing complete. {processed_count} images processed.")
if missing_files > 0:
    print(f"⚠️ {missing_files} images were missing and could not be processed.")
