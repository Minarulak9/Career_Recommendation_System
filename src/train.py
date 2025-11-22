# train.py
"""
Model training script with:
- Stratified split
- Class balancing
- Better hyperparameters for accuracy
- Comprehensive evaluation
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
import xgboost as xgb
import warnings

from pipeline import build_preprocessor, drop_features

warnings.filterwarnings("ignore")

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data.csv"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

TARGET_COL = "Target Job Role"


def load_data():
    """Load and clean data."""
    df = pd.read_csv(DATA_PATH)
    
    # Drop unnecessary columns
    df = df.drop(columns=drop_features, errors="ignore")
    
    # Remove rows with missing target
    df = df.dropna(subset=[TARGET_COL])
    
    print(f"Loaded {len(df)} samples")
    print(f"Target distribution:\n{df[TARGET_COL].value_counts()}")
    
    return df


def main():
    print("=" * 50)
    print("TRAINING CAREER PREDICTION MODEL")
    print("=" * 50)
    
    # Load data
    print("\n[1/5] Loading data...")
    df = load_data()
    
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    print(f"\nClasses ({n_classes}): {list(le.classes_)}")
    
    # Split data
    print("\n[2/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded  # Maintain class balance
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Build preprocessor
    print("\n[3/5] Building preprocessor...")
    preprocessor = build_preprocessor()
    
    # XGBoost classifier with optimized params
    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False
    )
    
    # Full pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])
    
    # Train
    print("\n[4/5] Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n[5/5] Evaluating...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Accuracy:       {acc:.4f}")
    print(f"F1 (Macro):     {f1_macro:.4f}")
    print(f"F1 (Weighted):  {f1_weighted:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save model and encoder
    joblib.dump(model, MODEL_DIR / "final_model.joblib")
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")
    
    print(f"\n✅ Model saved: {MODEL_DIR / 'final_model.joblib'}")
    print(f"✅ Encoder saved: {MODEL_DIR / 'label_encoder.joblib'}")
    
    # Save report
    report_path = REPORTS_DIR / "training_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (Macro): {f1_macro:.4f}\n")
        f.write(f"F1 (Weighted): {f1_weighted:.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print(f"✅ Report saved: {report_path}")


if __name__ == "__main__":
    main()