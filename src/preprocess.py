
# CodeAlpha - Credit Scoring Model
# Step 4: Data Preprocessing and Feature Engineering

import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ----------------------------------
# 1. Load Dataset
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "data" / "credit_data.csv")

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)

# ----------------------------------
# 2. Separate Features and Target
# ----------------------------------

X = df.drop("class", axis=1)

# Good = 1, Bad = 0
y = df["class"].map({
    "good": 1,
    "bad": 0
})

if y.isnull().any():
    raise ValueError("Unexpected target labels found!")

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# ----------------------------------
# 3. Identify Column Types
# ----------------------------------

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical columns:", len(numerical_features))
print("Categorical columns:", len(categorical_features))

# ----------------------------------
# 4. Preprocessing Pipelines
# ----------------------------------

# Numerical data preprocessing
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical data preprocessing
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

# Combine both pipelines
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# ----------------------------------
# 5. Split Dataset
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ----------------------------------
# 6. Fit and Transform Data
# ----------------------------------

# Learn preprocessing parameters from training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same transformations to test data
X_test_processed = preprocessor.transform(X_test)

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)

# ----------------------------------
# 7. Save Processed Data
# ----------------------------------

data_dir = BASE_DIR / "data"
models_dir = BASE_DIR / "models"

data_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

X_train_processed.to_csv(
    data_dir / "X_train.csv", index=False
)

X_test_processed.to_csv(
    data_dir / "X_test.csv", index=False
)

y_train.to_csv(
    data_dir / "y_train.csv", index=False
)

y_test.to_csv(
    data_dir / "y_test.csv", index=False
)

joblib.dump(
    preprocessor,
    models_dir / "preprocessor.joblib"
)

# ----------------------------------
# 8. Display Final Results
# ----------------------------------

print("\n========== PREPROCESSING RESULTS ==========")

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nData preprocessing completed successfully!")
