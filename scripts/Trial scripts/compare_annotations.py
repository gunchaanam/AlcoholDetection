import json
import os

# Paths to the JSON files
iritech_json_path = "C:/PycharmProjects/Alcohol Detection/dataset/Grupo_0/E_0_0/IriTech/E_0_0_0/E_0_0_0_L_5/via_region_data.json"
lg_json_path = "C:/PycharmProjects/Alcohol Detection/dataset/Grupo_0/E_0_0/LG/E_0_0_0/E_0_0_0_R_10/via_region_data.json"


def load_json(json_path):
    """ Load JSON annotation file """
    if os.path.exists(json_path):
        with open(json_path, "r") as file:
            return json.load(file)
    return None


def extract_polygon_coordinates(json_data):
    """ Extract polygon annotation coordinates from JSON """
    annotations = {}

    if json_data:
        for image_id, details in json_data.items():
            if "regions" in details:
                polygons = []
                for region in details["regions"]:
                    shape = region.get("shape_attributes", {})
                    if shape.get("name") == "polygon":
                        polygons.append({
                            "all_points_x": shape.get("all_points_x", []),
                            "all_points_y": shape.get("all_points_y", [])
                        })
                if polygons:
                    annotations[image_id] = polygons
    return annotations


# Load JSON data
iritech_data = load_json(iritech_json_path)
lg_data = load_json(lg_json_path)

# Extract polygon annotations
iritech_annotations = extract_polygon_coordinates(iritech_data)
lg_annotations = extract_polygon_coordinates(lg_data)

# Compare annotations for common images
common_images = set(iritech_annotations.keys()).intersection(set(lg_annotations.keys()))

print(f"\n🔍 Found {len(common_images)} common images in both datasets.\n")

for image_id in common_images:
    iritech_polygons = iritech_annotations[image_id]
    lg_polygons = lg_annotations[image_id]

    if iritech_polygons != lg_polygons:
        print(f"⚠️ Differences found in annotation for image: {image_id}")
    else:
        print(f"✅ Identical annotations for image: {image_id}")

# Summary
if len(common_images) == 0:
    print("❌ No common images found between IriTech and LG.")

print("\n🔹 Comparison Complete!")
