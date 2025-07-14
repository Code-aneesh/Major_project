from fpdf import FPDF
from pathlib import Path
from datetime import datetime
import pandas as pd

def generate_visual_report(image_stem="sample", risk_score=None, heuristic_label=None, rf_label=None, rf_conf=None):
    eval_dir = Path("scripts/results/evaluation")
    pred_dir = Path("results/predictions")
    train_dir = Path("runs/train_improved/fetal_ultrasound_v2")
    rf_data_path = Path("scripts/rf_data/rf_features.csv")

    # Output images & plots
    metrics_image = eval_dir / f"{image_stem}_metrics.png"
    per_class_image = eval_dir / f"per_class_metrics_{image_stem}.png"
    prediction_image = pred_dir / f"pred_{image_stem}.png"
    bar_chart_path = eval_dir / f"rf_vs_heuristic_{image_stem}.png"
    feature_importance_path = eval_dir / "rf_feature_importance.png"
    training_graph = train_dir / "results.png"
    confusion_path = eval_dir / "confusion_matrix.png"
    roc_curve_path = eval_dir / "roc_curve.png"

    report_path = eval_dir / f"Fetal_Report_{image_stem}.pdf"

    # Load RF values from CSV if not passed in
    if any(val is None for val in [risk_score, heuristic_label, rf_label, rf_conf]):
        try:
            df = pd.read_csv(rf_data_path)
            row = df[df["fname"].astype(str) == str(image_stem)].iloc[0]
            risk_score = row["risk_score"]
            heuristic_label = "HIGH RISK" if row["heuristic_label"] == 1 else "LOW RISK"
            rf_label = "HIGH RISK" if row.get("rf_label", 0) == 1 else "LOW RISK"
            rf_conf = row.get("rf_confidence", "N/A")
        except Exception:
            risk_score = heuristic_label = rf_label = rf_conf = "N/A"

    # PDF setup
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Fetal Ultrasound AI Evaluation Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")

    # Prediction Visualization
    if prediction_image.exists():
        pdf.ln(8)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "Detected Biomarkers", ln=True)
        pdf.image(str(prediction_image), w=180)

    # Overall Metrics
    if metrics_image.exists():
        pdf.ln(8)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "Overall Evaluation Metrics", ln=True)
        pdf.image(str(metrics_image), w=180)

    # Per-Class Metrics
    if per_class_image.exists():
        pdf.ln(8)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "Per-Class Performance", ln=True)
        pdf.image(str(per_class_image), w=180)

    # Confusion Matrix
    if confusion_path.exists():
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "Confusion Matrix (Predicted vs Ground Truth)", ln=True)
        pdf.image(str(confusion_path), w=180)

    # ROC Curve
    if roc_curve_path.exists():
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "ROC Curve (True Positive vs False Positive Rate)", ln=True)
        pdf.image(str(roc_curve_path), w=180)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 8, "The ROC curve illustrates the trade-off between sensitivity and specificity. "
                             "AUC values closer to 1.0 indicate a better-performing classifier.")

    # Heuristic vs ML Comparison
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Heuristic vs ML-Based Risk Assessment", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.ln(2)
    pdf.cell(60, 10, "Risk Score:", 0)
    pdf.cell(60, 10, str(round(risk_score, 2)) if isinstance(risk_score, float) else str(risk_score), ln=True)
    pdf.cell(60, 10, "Heuristic Label:", 0)
    pdf.cell(60, 10, heuristic_label, ln=True)
    pdf.cell(60, 10, "RF Label:", 0)
    pdf.cell(60, 10, rf_label, ln=True)
    pdf.cell(60, 10, "RF Confidence:", 0)
    pdf.cell(60, 10, str(round(rf_conf, 2)) if isinstance(rf_conf, float) else str(rf_conf), ln=True)

    # Comparison Chart
    if bar_chart_path.exists():
        pdf.ln(6)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "Comparison: Heuristic Risk vs ML Confidence", ln=True)
        pdf.image(str(bar_chart_path), w=180)

    # RF Feature Importance
    if feature_importance_path.exists():
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "Random Forest Feature Importance", ln=True)
        pdf.image(str(feature_importance_path), w=180)

    # YOLO Training Performance
    if training_graph.exists():
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "YOLOv8 Training Performance", ln=True)
        pdf.image(str(training_graph), w=180)
        pdf.set_font("Arial", "", 11)
        pdf.ln(5)
        pdf.multi_cell(0, 8,
            "- box_loss: Bounding box regression error (lower is better)\n"
            "- cls_loss: Class misclassification loss (lower is better)\n"
            "- dfl_loss: Distribution focal loss (lower is better)\n"
            "- val_*: Same metrics on validation set\n"
            "- precision(B): Correct positive detections (higher is better)\n"
            "- recall(B): Actual positive coverage (higher is better)\n"
            "- mAP50 / mAP50-95: Mean average precision at multiple thresholds (higher is better)")
        


    # Final Summary
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10,
        "Summary:\n"
        "- This report integrates object detection (YOLOv8) with biomarker risk analysis.\n"
        "- Two approaches for Down Syndrome risk scoring are used:\n"
        "   (a) Heuristic threshold-based rules\n"
        "   (b) ML-based Random Forest prediction\n"
        "- This dual-method analysis provides both interpretability and predictive power.\n"
        "- Charts and model graphs support transparency and future clinical validation.")

    pdf.output(str(report_path))
    print(f"📄 PDF report saved to: {report_path}")
