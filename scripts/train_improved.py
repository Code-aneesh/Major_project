from ultralytics import YOLO
import torch
import albumentations as A
from pathlib import Path
import cv2
import numpy as np
import shutil

# Define paths
project_root = Path("C:/Users/user/OneDrive/Desktop/major project/fetal_ultrasound_project")
dataset_path = project_root / "datasets"

# Enhanced data augmentation pipeline
def create_augmentations():
    return A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(p=0.8),
            A.RandomGamma(p=0.8),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.MedianBlur(blur_limit=3, p=0.3),
        ], p=0.4),
        A.OneOf([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
        ], p=0.5),
    ])

# Prepare model configuration
def setup_training():
    # Load a pretrained model
    model = YOLO('yolov8n.pt')
    
    # Custom training settings
    training_args = {
        'data': str(project_root / 'dataset.yaml'),
        'epochs': 100,
        'imgsz': 640,
        'batch': 8,
        'patience': 20,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'close_mosaic': 10,
        'augment': True,
        'mixup': 0.3,
        'mosaic': 0.8,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'project': str(project_root / 'runs/train_improved'),
        'name': 'fetal_ultrasound_v2'
    }
    
    return model, training_args

def train_model():
    print("Setting up training...")
    model, training_args = setup_training()
    
    print("Starting training with improved parameters...")
    try:
        results = model.train(**training_args)
        print("Training completed successfully!")
        
        # Save best model
        best_model_path = Path(training_args['project']) / training_args['name'] / 'weights/best.pt'
        if best_model_path.exists():
            shutil.copy(str(best_model_path), str(project_root / 'models/best_fetal.pt'))
            print(f"Best model saved to: {project_root}/models/best_fetal.pt")
    
    except Exception as e:
        print(f"Error during training: {e}")
        return None
    
    return results

if __name__ == "__main__":
    # Create necessary directories
    (project_root / 'models').mkdir(exist_ok=True)
    
    print("Starting improved training pipeline...")
    results = train_model()