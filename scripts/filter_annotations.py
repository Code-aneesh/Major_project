import os
import pandas as pd
import cv2
import numpy as np

# Define paths
project_root = "C:/Users/user/OneDrive/Desktop/major project/fetal_ultrasound_project/"
images_folder = os.path.join(project_root, "datasets/mendeley_data/images/")
annotations_file = os.path.join(project_root, "datasets/mendeley_data/ObjectDetection.xlsx")
output_csv = os.path.join(project_root, "datasets/processed_data/filtered_annotations.csv")

def verify_landmark(image_path, x, y, threshold=25):
    """Verify if there's a dark landmark around the specified coordinates"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False
    
    # Get region around the landmark
    h, w = image.shape
    x, y = int(x), int(y)
    
    # Use larger ROI initially
    roi_size = 5
    x1, y1 = max(0, x - roi_size), max(0, y - roi_size)
    x2, y2 = min(w, x + roi_size), min(h, y + roi_size)
    roi = image[y1:y2, x1:x2]
    
    # Calculate local statistics
    local_min = np.min(roi)
    local_mean = np.mean(roi)
    local_std = np.std(roi)
    
    # Criteria based on known good landmarks
    is_dark_enough = local_min < threshold
    has_contrast = (local_mean - local_min) > 15
    is_consistent = local_std < 40
    
    return is_dark_enough and has_contrast and is_consistent

# Load the known good annotations for reference
known_good = pd.read_csv(output_csv)
print(f"Loaded {len(known_good)} known good landmarks for reference")

# Load Excel annotations file
df = pd.read_excel(annotations_file, sheet_name="ObjectDetection")

print("Available columns:", df.columns.tolist())

# Fix filename formatting
df["fname"] = df["fname"].astype(str)
df["fname"] = df["fname"].str.replace(".png", "", regex=False) + ".png"
df["fname"] = df["fname"].str.lower().str.strip()

# Calculate center points
df["x"] = (df["w_min"] + df["w_max"]) / 2
df["y"] = (df["h_min"] + df["h_max"]) / 2

# List all image files
image_files = {f.lower().strip() for f in os.listdir(images_folder)}

# Filter annotations and verify landmarks
valid_annotations = []
for idx, row in df.iterrows():
    if row["fname"] in image_files:
        image_path = os.path.join(images_folder, row["fname"])
        if verify_landmark(image_path, row["x"], row["y"]):
            valid_annotations.append(row)

filtered_annotations = pd.DataFrame(valid_annotations)

# Save the filtered annotations
filtered_annotations.to_csv(output_csv, index=False)

print(f"✅ Filtered annotations saved at: {output_csv}")
print(f"Total annotations kept: {len(filtered_annotations)}")
print("Note: Only annotations with verified landmarks were kept")
