from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def predict_on_image(model_path, image_path, conf_threshold=0.25):
    # Load the trained model
    model = YOLO(model_path)
    
    # Read and preprocess image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
        
    # Resize image to match training size
    image = cv2.resize(image, (640, 640))
    
    # Enhance contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Make prediction
    results = model(enhanced, conf=conf_threshold)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    im_array = results[0].plot()
    plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Fetal Ultrasound Detection Results')
    
    # Save and show results
    output_dir = Path("results/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pred_{Path(image_path).name}"
    plt.savefig(str(output_path))
    plt.show()
    
    # Print detected structures
    print("\nDetected Structures:")
    for r in results:
        boxes = r.boxes
        if len(boxes) == 0:
            print("No structures detected. Please ensure this is a fetal ultrasound image showing:")
            print("- NT (Nuchal Translucency)")
            print("- Nasal features")
            print("- Brain structures")
        for box in boxes:
            class_id = int(box.cls)
            conf = float(box.conf)
            class_name = model.names[class_id]
            print(f"{class_name}: {conf:.2%} confidence")

if __name__ == "__main__":
    # Model path
    model_path = "runs/train_improved/fetal_ultrasound_v2/weights/best.pt"
    
    # Test on new images
    test_image = input("Enter the path to your ultrasound image: ")
    predict_on_image(model_path, test_image)