import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# Define paths
project_root = "C:/Users/user/OneDrive/Desktop/major project/fetal_ultrasound_project/"
images_folder = os.path.join(project_root, "datasets/mendeley_data/images/")
filtered_annotations = os.path.join(project_root, "datasets/processed_data/filtered_annotations.csv")

def show_landmarks(df, num_samples=5):
    # Load random samples
    samples = df.sample(n=min(num_samples, len(df)))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    for idx, (_, row) in enumerate(samples.iterrows()):
        img_path = os.path.join(images_folder, row['fname'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw landmark
        cv2.circle(img, (int(row['x']), int(row['y'])), 5, (255, 0, 0), -1)
        
        axes[idx].imshow(img)
        axes[idx].set_title(f"Image: {row['fname']}")
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# Load and display filtered annotations
df = pd.read_csv(filtered_annotations)
print(f"Total filtered annotations: {len(df)}")
show_landmarks(df)