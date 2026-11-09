
# CodeAlpha - Credit Scoring Model
# Step 3: Exploratory Data Analysis

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/credit_data.csv")

print("========== DATASET INFORMATION ==========")

# 1. Dataset shape
print("\nDataset Shape:")
print(df.shape)

# 2. Dataset information
print("\nDataset Information:")
df.info()

# 3. Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# 5. Statistical summary
print("\nStatistical Summary:")
print(df.describe(include="all"))

# 6. Target variable distribution
print("\nCredit Risk Distribution:")
print(df["class"].value_counts())

# 7. Visualize credit risk distribution
sns.countplot(data=df, x="class")
plt.title("Good vs Bad Credit Risk")
plt.xlabel("Credit Class")
plt.ylabel("Number of Customers")
os.makedirs("results", exist_ok=True)
plt.savefig("results/credit_distribution.png")
plt.show()

print("\nEDA completed successfully!")
