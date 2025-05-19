import os
import json
import pandas as pd
import re

dataset_path = r"E:\alco\Subset_Iris_under_alcohol_ICPRv2 (1)"

def clean_filename(filename):
    """Fix incorrect filenames by keeping only the valid .bmp name."""
    match = re.match(r"(.+\.bmp)", filename)
    return match.group(1) if match else filename

def find_json_files(directory):
    """Recursively find all JSON files and extract metadata."""
    json_files = []
    metadata_list = []

    for root, _, files in os.walk(directory):
        print(f" Checking folder: {root}")

        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                json_files.append(json_path)
                print(f" Found JSON: {json_path}")

                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        print(f" Processing {file}: {len(data.keys())} images")

                        for image_name in data.keys():
                            clean_name = clean_filename(image_name)

                            # Get label based on folder name
                            folder_name = os.path.basename(os.path.dirname(json_path))
                            parts = folder_name.split('_')

                            # Corrected logic: parts[3] corresponds to the "timeset" value
                            if len(parts) > 3 and parts[3] == '0':
                                label = "Non-Alcohol"
                            else:
                                label = "Alcohol"

                            metadata_list.append([clean_name, label, json_path])

                except json.JSONDecodeError:
                    print(f" Error reading {json_path}\n")

    metadata_df = pd.DataFrame(metadata_list, columns=["Image_Name", "Label", "JSON_Path"])
    metadata_df.to_csv("metadata.csv", index=False)
    print("\n✅ Metadata extraction complete. Saved as metadata.csv")

    return json_files

json_files_found = find_json_files(dataset_path)
print(f"\n Total JSON files found: {len(json_files_found)}")
