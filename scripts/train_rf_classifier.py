import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# Paths
rf_dir = Path("scripts/rf_data")
rf_dir.mkdir(parents=True, exist_ok=True)
csv_path = rf_dir / "rf_features.csv"
model_path = rf_dir / "rf_model.joblib"
report_path = rf_dir / "rf_metrics.txt"
confmat_path = rf_dir / "rf_confusion_matrix.csv"
feature_plot_path = Path("scripts/results/evaluation/rf_feature_importance.png")

# Load data
df = pd.read_csv(csv_path)

# Features and Labels
X = df.drop(columns=["fname", "heuristic_label"])
y = df["heuristic_label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict & Evaluate
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, digits=4)
conf_matrix = confusion_matrix(y_test, y_pred)

# Save model
joblib.dump(clf, model_path)

# Save classification report
with open(report_path, "w") as f:
    f.write("Random Forest Evaluation Metrics\n")
    f.write("=" * 40 + "\n")
    f.write(report)

# Save confusion matrix
pd.DataFrame(conf_matrix,
             index=["Actual 0", "Actual 1"],
             columns=["Predicted 0", "Predicted 1"]
             ).to_csv(confmat_path)

# Plot feature importance
importances = clf.feature_importances_
feature_names = X.columns
sorted_idx = importances.argsort()

plt.figure(figsize=(10, 6))
plt.barh(feature_names[sorted_idx], importances[sorted_idx], color="green")
plt.xlabel("Importance Score")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

# Save feature importance plot
feature_plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(feature_plot_path)
plt.close()

# ✅ Final print
print("✅ Random Forest trained and saved to:", model_path)
print("📊 Evaluation report saved to:", report_path)
print("📉 Confusion matrix saved to:", confmat_path)
print("🌲 Feature importance saved to:", feature_plot_path)
