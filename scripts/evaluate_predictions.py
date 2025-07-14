import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, predictions_dir, ground_truth_path):
        self.predictions_dir = Path(predictions_dir)
        self.ground_truth_path = Path(ground_truth_path)
        self.predictions = {}
        self.ground_truth = {}
        self.classes = set()

    def load_predictions(self):
        for json_file in self.predictions_dir.glob('*.json'):
            with open(json_file, 'r') as f:
                pred_data = json.load(f)
                filename = json_file.stem.replace('pred_', '')
                self.predictions[filename] = pred_data
                self.classes.update(pred_data.keys())

    def load_ground_truth(self):
        try:
            df = pd.read_excel(self.ground_truth_path, sheet_name='ObjectDetection')
            for _, row in df.iterrows():
                filename = Path(row['fname']).stem
                structure = row['structure']
                if filename not in self.ground_truth:
                    self.ground_truth[filename] = set()
                self.ground_truth[filename].add(structure)
                self.classes.add(structure)
        except Exception as e:
            print(f"Error loading ground truth: {e}")
            return False
        return True

    def compute_metrics(self):
        metrics = {}
        all_classes = sorted(list(self.classes))
        cm = np.zeros((len(all_classes), len(all_classes)))
        y_true_all = []
        y_pred_all = []
        y_score_all = []

        for filename in self.predictions:
            if filename not in self.ground_truth:
                print(f"Warning: No ground truth for {filename}")
                continue

            pred = self.predictions[filename]
            gt = self.ground_truth[filename]

            for i, pred_class in enumerate(all_classes):
                for j, gt_class in enumerate(all_classes):
                    if pred_class in pred and gt_class in gt:
                        cm[i, j] += 1

            for cls in all_classes:
                y_true_all.append(1 if cls in gt else 0)
                y_pred_all.append(1 if cls in pred else 0)
                y_score_all.append(pred.get(cls, 0))

        for cls in all_classes:
            metrics[cls] = {
                'precision': precision_score(
                    [1 if cls in self.ground_truth.get(f, set()) else 0 for f in self.predictions],
                    [1 if cls in self.predictions.get(f, {}) else 0 for f in self.predictions],
                    zero_division=0
                ),
                'recall': recall_score(
                    [1 if cls in self.ground_truth.get(f, set()) else 0 for f in self.predictions],
                    [1 if cls in self.predictions.get(f, {}) else 0 for f in self.predictions],
                    zero_division=0
                ),
                'f1': f1_score(
                    [1 if cls in self.ground_truth.get(f, set()) else 0 for f in self.predictions],
                    [1 if cls in self.predictions.get(f, {}) else 0 for f in self.predictions],
                    zero_division=0
                )
            }

        metrics['overall'] = {
            'accuracy': accuracy_score(y_true_all, y_pred_all),
            'precision': precision_score(y_true_all, y_pred_all, zero_division=0),
            'recall': recall_score(y_true_all, y_pred_all, zero_division=0),
            'f1': f1_score(y_true_all, y_pred_all, zero_division=0)
        }

        if any(y_score_all):
            fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
            roc_auc = auc(fpr, tpr)
            metrics['overall']['roc_auc'] = roc_auc
            metrics['overall']['fpr'] = fpr
            metrics['overall']['tpr'] = tpr

        metrics['confusion_matrix'] = cm
        metrics['classes'] = all_classes
        return metrics

    def plot_metrics(self, metrics, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        classes = metrics['classes']
        precisions = [metrics[cls]['precision'] for cls in classes]
        recalls = [metrics[cls]['recall'] for cls in classes]
        f1_scores = [metrics[cls]['f1'] for cls in classes]

        plt.figure(figsize=(15, 6))
        x = np.arange(len(classes))
        width = 0.25
        plt.bar(x - width, precisions, width, label='Precision')
        plt.bar(x, recalls, width, label='Recall')
        plt.bar(x + width, f1_scores, width, label='F1-Score')
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Per-Class Metrics')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'per_class_metrics.png')
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics['confusion_matrix'], 
                   annot=True, fmt='g', 
                   xticklabels=classes,
                   yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png')
        plt.close()

        if 'roc_auc' in metrics['overall']:
            plt.figure(figsize=(8, 6))
            plt.plot(metrics['overall']['fpr'], 
                     metrics['overall']['tpr'], 
                     label=f'ROC curve (AUC = {metrics["overall"]["roc_auc"]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'roc_curve.png')
            plt.close()

    def save_metrics_report(self, metrics, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        class_metrics = []
        for cls in metrics['classes']:
            class_metrics.append({
                'Class': cls,
                'Precision': metrics[cls]['precision'],
                'Recall': metrics[cls]['recall'],
                'F1-Score': metrics[cls]['f1']
            })

        pd.DataFrame(class_metrics).to_csv(output_dir / 'per_class_metrics.csv', index=False)

        overall_metrics = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [
                metrics['overall']['accuracy'],
                metrics['overall']['precision'],
                metrics['overall']['recall'],
                metrics['overall']['f1']
            ]
        }
        if 'roc_auc' in metrics['overall']:
            overall_metrics['Metric'].append('ROC AUC')
            overall_metrics['Value'].append(metrics['overall']['roc_auc'])

        pd.DataFrame(overall_metrics).to_csv(output_dir / 'overall_metrics.csv', index=False)

# ✅ New: Evaluate only a single prediction + GT
def evaluate_single_image(prediction_json_path, ground_truth_excel_path, output_dir):
    output_dir = Path(output_dir)
    with open(prediction_json_path, 'r') as f:
        predictions = json.load(f)
    image_stem = Path(prediction_json_path).stem.replace("pred_", "")

    df_gt = pd.read_excel(ground_truth_excel_path, sheet_name="ObjectDetection")
    gt_structures = df_gt[df_gt["fname"].str.contains(image_stem, na=False)]["structure"].tolist()

    labels = sorted(set(list(predictions.keys()) + gt_structures))
    y_true = [1 if cls in gt_structures else 0 for cls in labels]
    y_pred = [1 if cls in predictions else 0 for cls in labels]
    y_score = [predictions.get(cls, 0) for cls in labels]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_true, y_pred)

    # ROC
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"roc_curve_{image_stem}.png")
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{image_stem}.png")
    plt.close()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "labels": labels
    }

def main():
    project_root = Path(__file__).resolve().parent.parent
    predictions_dir = project_root / 'scripts' / 'results'
    ground_truth_path = project_root / 'datasets' / 'mendeley_data' / 'ObjectDetection.xlsx'
    output_dir = project_root / 'scripts' / 'results' / 'evaluation'

    evaluator = ModelEvaluator(predictions_dir, ground_truth_path)
    evaluator.load_predictions()
    if evaluator.load_ground_truth():
        metrics = evaluator.compute_metrics()
        evaluator.plot_metrics(metrics, output_dir)
        evaluator.save_metrics_report(metrics, output_dir)
        print("✅ Evaluation complete! Results saved in:", output_dir)
    else:
        print("❌ Failed to load ground truth data")

if __name__ == "__main__":
    main()
