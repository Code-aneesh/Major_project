import os
import json
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import cv2
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def calculate_risk_score(predictions: dict) -> float:
    risk_score = 0.0
    nt_confidence = predictions.get("NT", 1.0)
    nasal_tip_confidence = predictions.get("nasal_tip", 1.0)
    nasal_skin_confidence = predictions.get("nasal_skin", 1.0)

    if nt_confidence < 0.5:
        risk_score += (0.5 - nt_confidence)
    if nasal_tip_confidence < 0.5:
        risk_score += (0.5 - nasal_tip_confidence)
    if nasal_skin_confidence < 0.5:
        risk_score += (0.5 - nasal_skin_confidence)

    return round(risk_score, 2)

def extract_features_from_prediction(json_path: Path) -> dict:
    with open(json_path, 'r') as f:
        data = json.load(f)

    features = {
        'fname': json_path.stem.replace('pred_', ''),
        'thalami': data.get('thalami', 0.0),
        'midbrain': data.get('midbrain', 0.0),
        'IT': data.get('IT', 0.0),
        'CM': data.get('CM', 0.0),
        'NT': data.get('NT', 0.0),
        'nasal_tip': data.get('nasal_tip', 0.0),
        'nasal_skin': data.get('nasal_skin', 0.0),
    }
    features['risk_score'] = calculate_risk_score(features)
    features['heuristic_label'] = 1 if features['risk_score'] > 1.0 else 0
    return features

def generate_features(model_path: str, image_folder: str, output_csv: str):
    model = YOLO(model_path)
    results_dir = Path("scripts/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_rows = []

    for image_file in Path(image_folder).rglob("*.png"):
        print(f"Processing {image_file.name}...")
        image = cv2.imread(str(image_file))
        image = cv2.resize(image, (640, 640))

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        results = model(enhanced, conf=0.25)
        predictions = {}
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls)
                conf = float(box.conf)
                class_name = model.names[class_id]
                predictions[class_name] = conf

        # Save prediction to JSON
        pred_file = results_dir / f"pred_{image_file.stem}.json"
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)

        features = extract_features_from_prediction(pred_file)
        feature_rows.append(features)

    # Save all features to CSV
    df = pd.DataFrame(feature_rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved Random Forest Dataset to: {output_csv}")

if __name__ == "__main__":
    model_path = "runs/train_improved/fetal_ultrasound_v2/weights/best.pt"
    image_dir = "datasets/mendeley_data/Dataset for Fetus Framework/Dataset for Fetus Framework/Set1-Training&Validation Sets CNN/Standard"
    output_file = "scripts/rf_data/rf_features.csv"

    Path("scripts/rf_data").mkdir(parents=True, exist_ok=True)
    generate_features(model_path, image_dir, output_file)
