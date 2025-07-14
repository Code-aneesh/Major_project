import os
from pathlib import Path
from scripts.predict import predict_on_image
from scripts.evaluate_predictions import ModelEvaluator

def main():
    # Get project root directory
    project_root = Path(__file__).resolve().parent
    
    # Model path
    model_path = str(project_root / "runs" / "train_improved" / "fetal_ultrasound_v2" / "weights" / "best.pt")
    
    # Test images directory
    test_images_dir = project_root / "datasets" / "mendeley_data" / "images"
    
    if not test_images_dir.exists():
        print(f"❌ Test images directory not found: {test_images_dir}")
        return
    
    # Run predictions on all test images
    print("\n🔍 Running predictions on test images...")
    for image_file in test_images_dir.glob("*.jpg"):
        print(f"\nProcessing: {image_file.name}")
        predict_on_image(model_path, str(image_file))
    
    # Run evaluation
    print("\n📊 Running model evaluation...")
    predictions_dir = project_root / "scripts" / "results"
    ground_truth_path = project_root / "datasets" / "mendeley_data" / "Dataset for Fetus Framework"/ "ObjectDetection.xlsx"
    output_dir = project_root / "scripts" / "results" / "evaluation"
    
    evaluator = ModelEvaluator(predictions_dir, ground_truth_path)
    evaluator.load_predictions()
    if evaluator.load_ground_truth():
        metrics = evaluator.compute_metrics()
        evaluator.plot_metrics(metrics, output_dir)
        evaluator.save_metrics_report(metrics, output_dir)
        print("\n✅ Evaluation complete! Results saved in:", output_dir)
    else:
        print("\n❌ Failed to load ground truth data")

if __name__ == "__main__":
    main() 