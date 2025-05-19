import json
import os

# Path to the dataset (modify this path if needed)
dataset_path = "C:/PycharmProjects/Alcohol Detection/dataset"


# Function to check labels for images at 00:00 time
def check_zero_time_labels():
    found = False  # Flag to track if 00:00 images are found

    # Walk through the dataset directories
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file == "via_region_data.json":  # Annotation file
                json_path = os.path.join(root, file)

                # Load JSON data
                with open(json_path, "r") as f:
                    data = json.load(f)

                # Check each image annotation
                for key, value in data.items():
                    if "filename" in value and "regions" in value:
                        filename = value["filename"]

                        # Check if timestamp 00:00 exists in the filename
                        if "_0_0_0_" in filename or "_00_00_" in filename:
                            found = True
                            # Extract label information
                            label = value["regions"][0]["region_attributes"].get("Alcohol", "Unknown")
                            print(f"Image: {filename} | Label: {label}")

    if not found:
        print("No images with timestamp 00:00 were found.")
    else:
        print("\nCheck the above labels to confirm correctness.")


# Run the check
check_zero_time_labels()
