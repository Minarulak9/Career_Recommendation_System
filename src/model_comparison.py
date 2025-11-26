# src/model_comparison.py
"""
MODEL COMPARISON REPORT
-------------------------------------
Compares 5 ML models on the SAME dataset:
1. Logistic Regression
2. SVM (RBF)
3. Decision Tree
4. Random Forest
5. XGBoost

Outputs:
- Accuracy
- Macro F1
- Weighted F1
- Classification Reports
- Confusion Matrices
- Comparison CSV
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from pipeline import build_preprocessor, drop_features
from skill_features import extract_skill_flags
from pipeline import ensure_full_schema

import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "final_training_dataset.csv"
RESULTS_DIR = ROOT / "reports"
RESULTS_DIR.mkdir(exist_ok=True)

TARGET_COL = "Target Job Role"


# ----------------------------------------------------------
# Load & preprocess dataset
# ----------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=drop_features, errors="ignore")
    df = extract_skill_flags(df)
    df = ensure_full_schema(df)
    df = df.dropna(subset=[TARGET_COL])
    return df


df = load_data()

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

preprocessor = build_preprocessor()

# ----------------------------------------------------------
# Define models
# ----------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, n_jobs=-1),
    "SVM (RBF)": SVC(kernel="rbf", C=3, gamma="scale"),
    "Decision Tree": DecisionTreeClassifier(max_depth=10),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        n_jobs=-1
    ),
    "XGBoost": xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1
    )
}

comparison_results = []

# ----------------------------------------------------------
# Train & evaluate each model
# ----------------------------------------------------------

for name, model in models.items():
    print(f"\n==============================")
    print(f"Training: {name}")
    print("==============================")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    comparison_results.append([name, acc, f1_macro, f1_weighted])

    # Save classification report
    with open(RESULTS_DIR / f"{name.replace(' ', '_')}_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred, target_names=le.classes_))

    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} Macro F1: {f1_macro:.4f}")

# ----------------------------------------------------------
# Save comparison table
# ----------------------------------------------------------
df_comp = pd.DataFrame(comparison_results, columns=["Model", "Accuracy", "Macro F1", "Weighted F1"])
df_comp.to_csv(RESULTS_DIR / "model_comparison_results.csv", index=False)

print("\n======================================")
print("MODEL COMPARISON COMPLETE")
print("Results saved to reports/model_comparison_results.csv")
print("======================================")
