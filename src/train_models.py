
# CodeAlpha - Credit Scoring Model
# Step 5: Machine Learning Model Training

import pandas as pd
import joblib

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ----------------------------------
# 1. Set Project Paths
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------
# 2. Load Processed Dataset
# ----------------------------------

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

print("Processed datasets loaded successfully!")

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# ----------------------------------
# 3. Define ML Models
# ----------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
}

# ----------------------------------
# 4. Train and Evaluate Models
# ----------------------------------

results = []

for name, model in models.items():

    print("\n================================")
    print("Training:", name)
    print("================================")

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Prediction probabilities
    y_probability = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test, y_probability
    )

    # Display results
    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))

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

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Store evaluation results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    # Save trained model
    model_filename = (
        name.lower().replace(" ", "_") + ".joblib"
    )

    joblib.dump(
        model,
        MODELS_DIR / model_filename
    )

    print("\nModel saved:", model_filename)

# ----------------------------------
# 5. Compare All Models
# ----------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")

print(results_df.to_string(index=False))

# Save comparison results
results_df.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")

print("\nAll models trained successfully!")
