
# CodeAlpha - Credit Scoring Model

import pandas as pd
from sklearn.datasets import fetch_openml

print("Downloading Credit Risk Dataset...")

# Download dataset
credit_data = fetch_openml(
    name="credit-g",
    version=1,
    as_frame=True
)

# Convert into DataFrame
df = credit_data.frame

# Display dataset information
print("\nDataset downloaded successfully!")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

# Save dataset
import os

os.makedirs("data", exist_ok=True)
df.to_csv("data/credit_data.csv", index=False)

print("\nDataset saved successfully in data folder!")
