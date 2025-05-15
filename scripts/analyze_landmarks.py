import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

# Define paths
project_root = "C:/Users/user/OneDrive/Desktop/major project/fetal_ultrasound_project/"
filtered_csv = os.path.join(project_root, "datasets/processed_data/filtered_annotations.csv")
images_folder = os.path.join(project_root, "datasets/mendeley_data/images/")

# Load the filtered annotations
df = pd.read_csv(filtered_csv)

# Define structure groups
structure_groups = {
    'NT': ['NT'],
    'Nasal': ['nasal tip', 'nasal skin'],
    'Brain': ['thalami', 'midbrain', 'IT', 'CM']
}

# Print statistics for each group
print("\nStructures Present in Dataset:")
for group_name, structures in structure_groups.items():
    print(f"\n{group_name} Structures:")
    group_data = df[df['structure'].isin(structures)]
    for structure in structures:
        count = len(df[df['structure'] == structure])
        if count > 0:
            struct_data = df[df['structure'] == structure]
            print(f"  {structure}:")
            print(f"    - Number of landmarks: {count}")
            print(f"    - Average position (x, y): ({struct_data['x'].mean():.1f}, {struct_data['y'].mean():.1f})")
            print(f"    - Average size: {(struct_data['w_max'] - struct_data['w_min']).mean():.1f} x {(struct_data['h_max'] - struct_data['h_min']).mean():.1f}")

def show_all_landmarks(image_name):
    image_data = df[df['fname'] == image_name]
    if len(image_data) == 0:
        return
    
    image_path = os.path.join(images_folder, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(image)
    
    colors = {'NT': 'red', 'nasal tip': 'blue', 'nasal skin': 'cyan',
              'thalami': 'green', 'midbrain': 'yellow', 'IT': 'magenta', 'CM': 'white'}
    
    for _, row in image_data.iterrows():
        plt.plot(row['x'], row['y'], '+', color=colors[row['structure']], 
                markersize=10, label=row['structure'])
        rect = plt.Rectangle((row['w_min'], row['h_min']), 
                           row['w_max'] - row['w_min'],
                           row['h_max'] - row['h_min'],
                           fill=False, color=colors[row['structure']])
        plt.gca().add_patch(rect)
    
    plt.title(f"Landmarks in {image_name}")
    plt.legend()
    plt.show()

# Show example images with all landmarks
print("\nShowing example images with landmarks...")
example_images = df['fname'].unique()[:3]  # Show first 3 images
for image_name in example_images:
    show_all_landmarks(image_name)