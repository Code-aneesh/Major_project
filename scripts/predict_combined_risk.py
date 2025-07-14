import json
from pathlib import Path
import joblib
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.predict import predict_on_image, calculate_risk_score
from scripts.generate_report import generate_visual_report

def predict_with_rf(prediction_json_path):
    model_path = Path("scripts/rf_data/rf_model.joblib")  # ✅ corrected model path
    if not model_path.exists():
        print("❌ RF model not found.")
        return None, None, {}

    rf = joblib.load(model_path)

    with open(prediction_json_path, 'r') as f:
        preds = json.load(f)

    structure_names = ['thalami', 'midbrain', 'IT', 'CM', 'NT', 'nasal_tip', 'nasal_skin']
    features = {name: preds.get(name, 0.0) for name in structure_names}

    # ➕ Add risk_score used during training
    risk_score = 0.0
    if preds.get("NT", 1.0) < 0.5:
        risk_score += (0.5 - preds.get("NT", 1.0))
    if preds.get("nasal_tip", 1.0) < 0.5:
        risk_score += (0.5 - preds.get("nasal_tip", 1.0))
    if preds.get("nasal_skin", 1.0) < 0.5:
        risk_score += (0.5 - preds.get("nasal_skin", 1.0))
    features["risk_score"] = risk_score

    df = pd.DataFrame([features])
    label = rf.predict(df)[0]
    prob = rf.predict_proba(df)[0][label]

    return label, prob, features

def generate_rf_visuals(image_stem, features):
    # 🔎 Compare RF vs Heuristic
    heuristic = features["risk_score"]
    rf_conf = features.get("rf_confidence", 1.0)
    label = features.get("rf_label", 0)

    # ➤ Bar Comparison Chart
    plt.figure(figsize=(6, 4))
    plt.bar(["Heuristic Score", "RF Confidence"], [heuristic, rf_conf], color=["orange", "green"])
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.title(f"Risk Evaluation - {'HIGH' if label else 'LOW'} RISK")
    plt.tight_layout()
    save_path = Path(f"scripts/results/evaluation/rf_vs_heuristic_{image_stem}.png")
    plt.savefig(save_path)
    plt.close()

    # ➤ Feature Importance (uses trained model)
    rf = joblib.load("scripts/rf_data/rf_model.joblib")
    importances = rf.feature_importances_
    feature_names = list(features.keys())
    feature_names.remove("rf_confidence")
    feature_names.remove("rf_label")

    plt.figure(figsize=(8, 5))
    sorted_idx = sorted(range(len(feature_names)), key=lambda i: importances[i])
    plt.barh([feature_names[i] for i in sorted_idx], [importances[i] for i in sorted_idx], color="steelblue")
    plt.xlabel("Importance")
    plt.title("RF Model Feature Importance")
    plt.tight_layout()
    plt.savefig("scripts/results/evaluation/rf_feature_importance.png")
    plt.close()

def main(image_path):
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "runs/train_improved/fetal_ultrasound_v2/weights/best.pt"

    # 🧠 Step 1: YOLO prediction + heuristic
    predict_on_image(str(model_path), str(image_path))

    image_stem = Path(image_path).stem
    prediction_path = Path(f"scripts/results/pred_{image_stem}.json")

    # 🤖 Step 2: RF classification
    if prediction_path.exists():
        label, prob, features = predict_with_rf(prediction_path)
        if label is not None:
            print(f"\n🤖 ML-Based RF Prediction: {'HIGH RISK' if label else 'LOW RISK'} (Confidence: {prob:.2f})")
            features["rf_confidence"] = prob
            features["rf_label"] = label

            # 📊 Step 3: Visualize results
            generate_rf_visuals(image_stem, features)

    # 📄 Step 4: Generate final PDF
    generate_visual_report(image_stem)

if __name__ == "__main__":
    test_image = input("Enter the path to your ultrasound image: ")
    main(test_image)
