
# CodeAlpha - Credit Scoring Model
# Step 6: Model Evaluation and Visualization

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)

# ----------------------------------
# 1. Project Paths
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------
# 2. Load Testing Data
# ----------------------------------

X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_test = pd.read_csv(
    DATA_DIR / "y_test.csv"
).squeeze()

print("Testing dataset loaded successfully!")

# ----------------------------------
# 3. Load Trained Models
# ----------------------------------

models = {
    "Logistic Regression": joblib.load(
        MODELS_DIR / "logistic_regression.joblib"
    ),

    "Decision Tree": joblib.load(
        MODELS_DIR / "decision_tree.joblib"
    ),

    "Random Forest": joblib.load(
        MODELS_DIR / "random_forest.joblib"
    )
}

# ----------------------------------
# 4. Evaluate Each Model
# ----------------------------------

evaluation_results = []

plt.figure(figsize=(9, 7))

for name, model in models.items():

    print("\n================================")
    print("Evaluating:", name)
    print("================================")

    # Predictions
    y_pred = model.predict(X_test)

    # Probability of bad credit (class 0)
    y_probability = model.predict_proba(X_test)[:, 0]

    # Convert bad credit into positive class
    y_test_bad = (y_test == 0).astype(int)
    y_pred_bad = (y_pred == 0).astype(int)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test_bad,
        y_pred_bad,
        zero_division=0
    )

    recall = recall_score(
        y_test_bad,
        y_pred_bad,
        zero_division=0
    )

    f1 = f1_score(
        y_test_bad,
        y_pred_bad,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test_bad,
        y_probability
    )

    print("\nAccuracy:", round(accuracy, 4))
    print("Bad Credit Precision:", round(precision, 4))
    print("Bad Credit Recall:", round(recall, 4))
    print("Bad Credit F1 Score:", round(f1, 4))
    print("Bad Credit ROC-AUC:", round(roc_auc, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Bad Credit", "Good Credit"],
            zero_division=0
        )
    )

    # ----------------------------------
    # 5. Confusion Matrix
    # ----------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Bad Credit", "Good Credit"],
        yticklabels=["Bad Credit", "Good Credit"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    filename = name.lower().replace(" ", "_")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / f"{filename}_confusion_matrix.png"
    )

    plt.close()

    # ----------------------------------
    # 6. ROC Curve
    # ----------------------------------

    fpr, tpr, thresholds = roc_curve(
        y_test_bad,
        y_probability
    )

    plt.figure(1)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.3f})"
    )

    # Store results
    evaluation_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Bad Credit Precision": precision,
        "Bad Credit Recall": recall,
        "Bad Credit F1 Score": f1,
        "Bad Credit ROC-AUC": roc_auc
    })

# ----------------------------------
# 7. Save ROC Curve
# ----------------------------------

plt.figure(1)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "roc_curve_comparison.png"
)

plt.close()

# ----------------------------------
# 8. Compare Model Performance
# ----------------------------------

results_df = pd.DataFrame(evaluation_results)

results_df = results_df.sort_values(
    by="Bad Credit F1 Score",
    ascending=False
)

print("\n========== FINAL MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# Save evaluation results
results_df.to_csv(
    RESULTS_DIR / "evaluation_metrics.csv",
    index=False
)

# ----------------------------------
# 9. Performance Comparison Graph
# ----------------------------------

metrics = [
    "Accuracy",
    "Bad Credit Precision",
    "Bad Credit Recall",
    "Bad Credit F1 Score",
    "Bad Credit ROC-AUC"
]

plot_data = results_df.set_index("Model")[metrics]

plot_data.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Machine Learning Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Machine Learning Models")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "model_performance_comparison.png"
)

plt.close()

# ----------------------------------
# 10. Best Model Selection
# ----------------------------------

best_model = results_df.iloc[0]

print("\n========== BEST MODEL ==========")

print("Best Model:", best_model["Model"])

print(
    "Bad Credit Recall:",
    round(best_model["Bad Credit Recall"], 4)
)

print(
    "Bad Credit F1 Score:",
    round(best_model["Bad Credit F1 Score"], 4)
)

print("\nAll evaluation tasks completed successfully!")
