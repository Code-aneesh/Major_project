from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import sys

def save_predictions(filename, predictions):
    output_dir = Path("scripts/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_filename = f"pred_{Path(filename).stem}.json"
    output_path = output_dir / pred_filename

    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"Predictions saved to: {output_path}")
    return Path(filename).stem

def load_ground_truth(image_name):
    csv_path = Path("datasets/processed_data/filtered_annotations.csv")
    if not csv_path.exists():
        print("⚠️ Ground truth CSV not found.")
        return []

    df = pd.read_csv(csv_path)
    image_base = Path(image_name).name
    ground_truth = df[df["fname"] == image_base]["structure"].tolist()
    return ground_truth

def evaluate(predictions_dict, ground_truth_list, image_stem="sample"):
    all_labels = sorted(set(ground_truth_list + list(predictions_dict.keys())))
    y_true = []
    y_pred = []

    for label in all_labels:
        y_true.append(int(label in ground_truth_list))
        y_pred.append(int(label in predictions_dict))

    accuracy = sum([t == p for t, p in zip(y_true, y_pred)]) / len(y_true)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    npv = tn / (tn + fn) if (tn + fn) else 0

    print("\n📊 Evaluation Results:")
    print(f"Accuracy:    {accuracy:.2f}")
    print(f"Precision:   {precision:.2f}")
    print(f"Recall:      {recall:.2f}")
    print(f"F1 Score:    {f1:.2f}")
    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"NPV:         {npv:.2f}")

    eval_dir = Path("scripts/results/evaluation/")
    eval_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Sensitivity", "Specificity", "NPV"],
        "Value": [accuracy, precision, recall, f1, sensitivity, specificity, npv]
    }).to_csv(eval_dir / f"{image_stem}_metrics.csv", index=False)

    # Bar chart
    plt.figure(figsize=(8, 5))
    labels = ["Accuracy", "Precision", "Recall", "F1", "Sensitivity", "Specificity", "NPV"]
    values = [accuracy, precision, recall, f1, sensitivity, specificity, npv]
    plt.bar(labels, values, color="teal")
    plt.ylim(0, 1)
    plt.title("Overall Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(eval_dir / f"{image_stem}_metrics.png")
    plt.close()

    # Per-class metrics
    per_class = []
    for i, label in enumerate(all_labels):
        tp = int(y_true[i] == 1 and y_pred[i] == 1)
        fp = int(y_true[i] == 0 and y_pred[i] == 1)
        fn = int(y_true[i] == 1 and y_pred[i] == 0)

        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1c = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        per_class.append({
            "Structure": label,
            "Precision": round(prec, 2),
            "Recall": round(rec, 2),
            "F1-Score": round(f1c, 2)
        })

    df_per_class = pd.DataFrame(per_class)
    df_per_class.to_csv(eval_dir / f"per_class_metrics_{image_stem}.csv", index=False)

    plt.figure(figsize=(10, 6))
    for metric in ["Precision", "Recall", "F1-Score"]:
        plt.plot(df_per_class["Structure"], df_per_class[metric], marker="o", label=metric)

    plt.title("Per-Class Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_dir / f"per_class_metrics_{image_stem}.png")
    plt.close()

def calculate_risk_score(predictions: dict) -> float:
    risk_score = 0.0
    reasons = []

    nt_confidence = predictions.get("NT", 1.0)
    nasal_tip_confidence = predictions.get("nasal_tip", 1.0)
    nasal_skin_confidence = predictions.get("nasal_skin", 1.0)

    if nt_confidence < 0.5:
        risk_score += (0.5 - nt_confidence)
        reasons.append(f"NT={nt_confidence:.2f}")
    if nasal_tip_confidence < 0.5:
        risk_score += (0.5 - nasal_tip_confidence)
        reasons.append(f"nasal_tip={nasal_tip_confidence:.2f}")
    if nasal_skin_confidence < 0.5:
        risk_score += (0.5 - nasal_skin_confidence)
        reasons.append(f"nasal_skin={nasal_skin_confidence:.2f}")

    print(f"\n📊 Risk Score: {risk_score:.2f}", end=" ")
    if risk_score > 1.0:
        print("→ HIGH RISK")
    else:
        print("→ LOW RISK")

    if reasons:
        print(f"Because: {', '.join(reasons)} (all low)")

    return risk_score

def predict_on_image(model_path, image_path, conf_threshold=0.25):
    model = YOLO(model_path)
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    image = cv2.resize(image, (640, 640))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    results = model(enhanced, conf=conf_threshold)

    output_dir = Path("results/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_stem = Path(image_path).stem
    pred_image_path = output_dir / f"pred_{image_stem}.png"
    im_array = results[0].plot()
    plt.imsave(pred_image_path, cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Fetal Ultrasound Detection Results')
    plt.show()

    print("\nDetected Structures:")
    predictions = {}
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls)
            conf = float(box.conf)
            class_name = model.names[class_id]
            print(f"{class_name}: {conf:.2%} confidence")
            predictions[class_name] = conf

    if predictions:
        save_predictions(image_path, predictions)
        calculate_risk_score(predictions)
        gt = load_ground_truth(image_path)
        if gt:
            evaluate(predictions, gt, image_stem)

            # ✅ Generate PDF report
            sys.path.append(str(Path(__file__).resolve().parent.parent))
            from scripts.generate_report import generate_visual_report
            generate_visual_report(image_stem)
        else:
            print("⚠️ No ground truth found for evaluation.")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    model_path = str(project_root / "runs" / "train_improved" / "fetal_ultrasound_v2" / "weights" / "best.pt")
    test_image = input("Enter the path to your ultrasound image: ")
    predict_on_image(model_path, test_image)
