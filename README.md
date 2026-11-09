
# CodeAlpha - Credit Scoring Model

## 1. Project Overview

This project aims to predict an individual's creditworthiness using Machine Learning classification algorithms.

It classifies customers into two categories:
- Good Credit Risk
- Bad Credit Risk

This project was developed as part of the CodeAlpha Machine Learning Internship.

## 2. Project Objective

To build and evaluate Machine Learning models that predict credit risk based on historical financial information.

## 3. Dataset Information

- Dataset: German Credit Risk Dataset
- Source: OpenML
- Total Records: 1,000
- Original Features: 20
- Target Variable: class
- Good Credit: 700 records
- Bad Credit: 300 records

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## 5. Machine Learning Algorithms

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## 6. Project Workflow

1. Dataset Collection
2. Exploratory Data Analysis (EDA)
3. Data Cleaning and Preprocessing
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Performance Comparison

## 7. Model Evaluation Results

| Model | Accuracy | Bad Credit Precision | Bad Credit Recall | Bad Credit F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 70.50% | 50.94% | 45.00% | 47.79% | 75.94% |
| Random Forest | 73.50% | 60.00% | 35.00% | 44.21% | 78.54% |
| Decision Tree | 64.50% | 41.27% | 43.33% | 42.28% | 58.45% |

## 8. Conclusion

Logistic Regression achieved the highest F1-score for bad-credit identification.

Random Forest achieved the highest accuracy and ROC-AUC.

Further optimization is required to improve bad-credit detection.

## 9. Project Structure

```text
CodeAlpha_CreditScoringModel/
│
├── data/
├── models/
├── notebooks/
├── results/
├── src/
│   ├── eda.py
│   ├── preprocess.py
│   ├── train_models.py
│   └── evaluate_models.py
│
├── main.py
├── requirements.txt
└── README.md
```

## 10. How to Run the Project

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the scripts in the following order:

```bash
python main.py
python src/eda.py
python src/preprocess.py
python src/train_models.py
python src/evaluate_models.py
```

## 11. Internship Details

Organization: CodeAlpha

Domain: Machine Learning

Project: Credit Scoring Model

## 12. Author

Tripti Varshney
