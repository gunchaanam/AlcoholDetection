import json
import os
import cv2
import matplotlib.pyplot as plt

# Paths to JSON files (update these paths as needed)
lg_json_path = "C:/PycharmProjects/Alcohol Detection/dataset/Grupo_0/E_0_0/LG/E_0_0_0/E_0_0_0_R_10/via_region_data.json"
iritech_json_path = "C:/PycharmProjects/Alcohol Detection/dataset/Grupo_0/E_0_0/IriTech/E_0_0_0/E_0_0_0_L_5/via_region_data.json"


# Load JSON annotations
def load_annotations(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data


lg_data = load_annotations(lg_json_path)
iritech_data = load_annotations(iritech_json_path)

# Find common images
common_images = set(lg_data.keys()).intersection(set(iritech_data.keys()))


# Function to extract polygon points
def extract_polygon(annotation):
    regions = annotation.get('regions', [])
    polygons = []
    for region in regions:
        shape = region['shape_attributes']
        if shape['name'] == 'polygon':
            polygons.append((shape['all_points_x'], shape['all_points_y']))
    return polygons


# Function to visualize annotation differences
def visualize_differences(image_name, lg_annotation, iritech_annotation, img_dir):
    img_path = os.path.join(img_dir, lg_annotation['filename'])
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lg_polygons = extract_polygon(lg_annotation)
    iritech_polygons = extract_polygon(iritech_annotation)

    plt.figure(figsize=(8, 6))
    plt.imshow(img)

    # Draw LG annotations in red
    for poly in lg_polygons:
        plt.plot(poly[0] + [poly[0][0]], poly[1] + [poly[1][0]], 'r-', label='LG' if poly == lg_polygons[0] else "")

    # Draw IriTech annotations in blue
    for poly in iritech_polygons:
        plt.plot(poly[0] + [poly[0][0]], poly[1] + [poly[1][0]], 'b--',
                 label='IriTech' if poly == iritech_polygons[0] else "")

    plt.legend()
    plt.title(f"Annotation Differences for {image_name}")
    plt.show()


# Visualize differences for a few common images
sample_images = list(common_images)[:3]  # Visualizing first 3 common images
for img in sample_images:
    visualize_differences(img, lg_data[img], iritech_data[img],
                          "C:/PycharmProjects/Alcohol Detection/dataset/Grupo_0/E_0_0/LG")
