from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_model(model_path, test_images_path):
    # Create validation directory if it doesn't exist
    validation_dir = Path("runs/validation")
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(data='dataset.yaml', split='test')
    
    # Create metrics DataFrame with proper index
    metrics_dict = {
        'Class': results.names.values(),
        'Precision': results.results_dict['metrics/precision(B)'],
        'Recall': results.results_dict['metrics/recall(B)'],
        'mAP50': results.results_dict['metrics/mAP50(B)'],
        'mAP50-95': results.results_dict['metrics/mAP50-95(B)']
    }
    
    metrics = pd.DataFrame(metrics_dict)
    
    # Plot metrics
    plt.figure(figsize=(12, 6))
    metrics.plot(x='Class', y=['Precision', 'Recall', 'mAP50'], kind='bar')
    plt.title('Model Performance by Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('runs/validation/metrics.png')
    
    return results

def visualize_predictions(model_path, test_path, num_images=5):
    model = YOLO(model_path)
    
    # Get list of test images
    test_images = list(Path(test_path).glob('*.jpg')) + list(Path(test_path).glob('*.png'))
    test_images = test_images[:num_images]  # Take first n images
    
    # Create figure for multiple images
    fig, axes = plt.subplots(len(test_images), 1, figsize=(15, 8*len(test_images)))
    if len(test_images) == 1:
        axes = [axes]
    
    # Process each image
    for idx, img_path in enumerate(test_images):
        # Run prediction
        results = model(str(img_path))
        
        # Plot results
        im_array = results[0].plot()
        axes[idx].imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
        axes[idx].axis('off')
        axes[idx].set_title(f'Detection Results - {img_path.name}')
    
    plt.tight_layout()
    plt.savefig('runs/validation/test_predictions.png')
    plt.show()

if __name__ == "__main__":
    model_path = "runs/train_improved/fetal_ultrasound_v2/weights/best.pt"
    test_path = "datasets/test/images"
    
    print("Evaluating model performance...")
    results = evaluate_model(model_path, test_path)
    
    print("\nModel Performance Summary:")
    print(f"Overall mAP50: {results.results_dict['metrics/mAP50(B)']:.3f}")
    print(f"Overall mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.3f}")
    
    print("\nVisualizing predictions on test images...")
    visualize_predictions(model_path, test_path)