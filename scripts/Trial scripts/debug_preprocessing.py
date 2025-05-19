import os
import json
import cv2

# Update this to your dataset path
DATASET_PATH = "C:/PycharmProjects/Alcohol Detection/dataset"


# ✅ Function to find all `via_region_data.json` files in the dataset
def find_json_files():
    json_files = []
    for root, _, files in os.walk(DATASET_PATH):
        for file in files:
            if file == "via_region_data.json":
                json_files.append(os.path.join(root, file))
    return json_files


# ✅ Function to validate `.bmp` images
def is_valid_image(image_path):
    img = cv2.imread(image_path)
    return img is not None


# ✅ Function to check all `.bmp` images
def check_bmp_images():
    print("\n🔍 Checking sample BMP images...")
    sample_images = []
    for root, _, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith(".bmp"):
                sample_images.append(os.path.join(root, file))
            if len(sample_images) >= 5:  # Limit to 5 for quick check
                break

    for img_path in sample_images:
        status = "✅ Valid" if is_valid_image(img_path) else "❌ Corrupt or Unreadable"
        print(f"🖼 Checking: {img_path} -> {status}")


# ✅ Function to check expected image filenames from all `via_region_data.json` files
def check_expected_filenames():
    json_files = find_json_files()
    if not json_files:
        print("⚠️ No `via_region_data.json` files found!")
        return

    expected_images = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
            expected_images.extend([value["filename"] for value in data.values()])

    print("\n📌 First 10 expected images from JSON:")
    print("\n".join(expected_images[:10]))


# ✅ Function to check for missing images
def check_missing_images():
    json_files = find_json_files()
    if not json_files:
        print("⚠️ No `via_region_data.json` files found!")
        return

    missing_images = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
            for value in data.values():
                img_path = os.path.join(os.path.dirname(json_file), value["filename"])
                if not os.path.exists(img_path):
                    missing_images.append(img_path)

    print("\n🔍 First 10 missing images:")
    print("\n".join(missing_images[:10]) if missing_images else "✅ No missing images found!")


# 🔥 Run all checks
if __name__ == "__main__":
    check_bmp_images()
    check_expected_filenames()
    check_missing_images()
