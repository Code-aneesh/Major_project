import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

# Define paths
project_root = "C:/Users/user/OneDrive/Desktop/major project/fetal_ultrasound_project/"
filtered_csv = os.path.join(project_root, "datasets/processed_data/filtered_annotations.csv")
images_folder = os.path.join(project_root, "datasets/mendeley_data/images/")

# Create YOLO dataset structure
dataset_root = os.path.join(project_root, "datasets")
for split in ['train', 'val', 'test']:
    for subdir in ['images', 'labels']:
        os.makedirs(os.path.join(dataset_root, split, subdir), exist_ok=True)

# Load annotations
df = pd.read_csv(filtered_csv)

# Define class mapping
class_mapping = {
    'NT': 0,
    'nasal tip': 1,
    'nasal skin': 2,
    'nasal bone': 3,
    'thalami': 4,
    'midbrain': 5,
    'IT': 6,
    'CM': 7,
    'palate': 8
}

def convert_to_yolo(row, img_width, img_height):
    class_id = class_mapping[row['structure']]
    x_center = (row['w_min'] + row['w_max']) / 2 / img_width
    y_center = (row['h_min'] + row['h_max']) / 2 / img_height
    width = (row['w_max'] - row['w_min']) / img_width
    height = (row['h_max'] - row['h_min']) / img_height
    
    return f"{class_id} {x_center} {y_center} {width} {height}"

# Split dataset
unique_images = df['fname'].unique()
train_images, temp_images = train_test_split(unique_images, test_size=0.2, random_state=42)
val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

# Process each split
splits = {
    'train': train_images,
    'val': val_images,
    'test': test_images
}

for split, images in splits.items():
    print(f"Processing {split} split...")
    for image_name in images:
        # Copy image
        src_img = os.path.join(images_folder, image_name)
        dst_img = os.path.join(dataset_root, split, 'images', image_name)
        shutil.copy2(src_img, dst_img)
        
        # Create YOLO annotation
        image = cv2.imread(src_img)
        if image is None:
            print(f"Warning: Could not read image {image_name}")
            continue
            
        height, width = image.shape[:2]
        
        # Get all annotations for this image
        image_annotations = df[df['fname'] == image_name]
        
        # Create YOLO format annotations
        yolo_annotations = []
        for _, row in image_annotations.iterrows():
            yolo_annotations.append(convert_to_yolo(row, width, height))
        
        # Save annotations
        label_path = os.path.join(dataset_root, split, 'labels', 
                                 os.path.splitext(image_name)[0] + '.txt')
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))

print("Dataset preparation completed!")
