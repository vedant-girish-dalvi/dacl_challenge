import os
import json
import numpy as np
import cv2
from glob import glob

json_dir = "./annotations/validation"
mask_dir = "./masks_grayscale/validation"  
os.makedirs(mask_dir, exist_ok=True)

CLASS_MAPPING = {
    "Crack": 1,
    "ACrack": 2,
    "Wetspot": 3,
    "Efflorescence": 4,
    "Rust": 5,
    "Rockpocket": 6,
    "Hollowareas": 7,
    "Cavity": 8,
    "Spalling": 9,
    "Graffiti": 10,
    "Weathering": 11,
    "Restformwork": 12,
    "ExposedRebars": 13,
    "Bearing": 14,
    "EJoint": 15,
    "Drainage": 16,
    "PEquipment": 17,
    "JTape": 18,
    "WConccor": 19
}

json_files = glob(os.path.join(json_dir, "*.json"))

for json_file in json_files:
    with open(json_file, "r") as f:
        annotation = json.load(f)

    img_width = annotation["imageWidth"]
    img_height = annotation["imageHeight"]

    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    for shape in annotation["shapes"]:
        label = shape["label"]
        points = np.array(shape["points"], dtype=np.int32)

        if label in CLASS_MAPPING:
            class_id = CLASS_MAPPING[label]
            cv2.fillPoly(mask, [points], color=class_id)

    mask_filename = os.path.splitext(os.path.basename(json_file))[0] + ".png"
    mask_path = os.path.join(mask_dir, mask_filename)
    cv2.imwrite(mask_path, mask)

print("Segmentation masks have been generated successfully!")



# Define RGB color mapping for each class
# CLASS_COLORS = {
#     "Crack": (255, 0, 0),          # Red
#     "ACrack": (0, 255, 0),         # Green
#     "Wetspot": (0, 0, 255),        # Blue
#     "Efflorescence": (255, 255, 0),# Yellow
#     "Rust": (255, 165, 0),         # Orange
#     "Rockpocket": (128, 0, 128),   # Purple
#     "Hollowareas": (0, 255, 255),  # Cyan
#     "Cavity": (255, 192, 203),     # Pink
#     "Spalling": (139, 69, 19),     # Brown
#     "Graffiti": (128, 128, 128),   # Gray
#     "Weathering": (0, 128, 128),   # Teal
#     "Restformwork": (50, 205, 50), # Lime Green
#     "ExposedRebars": (75, 0, 130), # Indigo
#     "Bearing": (255, 20, 147),     # Deep Pink
#     "EJoint": (0, 191, 255),       # Deep Sky Blue
#     "Drainage": (139, 0, 139),     # Dark Magenta
#     "PEquipment": (173, 255, 47),  # Green Yellow
#     "JTape": (220, 20, 60),        # Crimson
#     "WConccor": (0, 100, 0)        # Dark Green
# }

# # Process each JSON annotation file
# json_files = glob(os.path.join(json_dir, "*.json"))

# for json_file in json_files:
#     with open(json_file, "r") as f:
#         annotation = json.load(f)

#     # Get image dimensions
#     img_width = annotation["imageWidth"]
#     img_height = annotation["imageHeight"]
    
#     # Create an empty RGB mask (initialized with zeros)
#     mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)

#     # Draw each polygon with its assigned color
#     for shape in annotation["shapes"]:
#         label = shape["label"]
#         points = np.array(shape["points"], dtype=np.int32)

#         if label in CLASS_COLORS:
#             color = CLASS_COLORS[label]  # Get the RGB color
#             cv2.fillPoly(mask, [points], color=color)  # Draw polygon with color

#     # Save the RGB mask
#     mask_filename = os.path.splitext(os.path.basename(json_file))[0] + ".png"
#     mask_path = os.path.join(mask_dir, mask_filename)
#     cv2.imwrite(mask_path, mask)

# print("RGB segmentation masks have been successfully generated!")
