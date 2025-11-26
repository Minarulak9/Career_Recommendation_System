"""
FINAL TRAINING SCRIPT
---------------------
Integrates:
- skill feature extraction
- corrected schema order
- updated hashing pipeline
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
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

from pipeline import (
    build_preprocessor,
    drop_features,
    ensure_full_schema,
    numeric_features as BASE_NUMERIC
)
from skill_features import extract_skill_flags

import warnings
warnings.filterwarnings("ignore")

# Paths
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / (
    "cleaned_dataset_v2.csv" if (ROOT / "data" / "cleaned_dataset_v2.csv").exists()
    else "data.csv"
)

MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

TARGET_COL = "Target Job Role"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=drop_features, errors="ignore")
    df = df.dropna(subset=[TARGET_COL])
    print(f"\nLoaded {len(df)} samples from {DATA_PATH.name}")
    return df


def main():
    print("=" * 70)
    print("TRAINING CAREER PREDICTION MODEL")
    print("=" * 70)

    print("[1/6] Loading dataset...")
    df = load_data()

    print("[2/6] Extracting skill features...")
    df = extract_skill_flags(df)

    print("[3/6] Preparing X and y...")

    # Extract target
    y = df[TARGET_COL]

    # Remove target column before schema
    X = df.drop(columns=[TARGET_COL])

    # ADD skill columns to numeric_features dynamically
    new_skill_cols = [
        col for col in X.columns
        if col.startswith("skill_") or col.endswith("_skill_count") or col == "total_skill_hits"
    ]
    BASE_NUMERIC.extend(new_skill_cols)

    # Enforce schema now that numeric list is updated
    X = ensure_full_schema(X)

    print(f"Total features after engineering: {X.shape[1]}")

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Classes: {list(le.classes_)}")

    print("\n[4/6] Splitting training/testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n[5/6] Building preprocessor...")
    preprocessor = build_preprocessor()

    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=350,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])

    print("\n[6/6] Computing class weights...")
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_map = {cls: weight for cls, weight in zip(classes, class_weights)}
    sample_weights = np.array([weight_map[c] for c in y_train])

    print("Training model...")
    model.fit(X_train, y_train, clf__sample_weight=sample_weights)

    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print("\n===== RESULTS =====")
    print(f"Accuracy:       {acc:.4f}")
    print(f"F1 Macro:       {f1_macro:.4f}")
    print(f"F1 Weighted:    {f1_weighted:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model + encoder
    joblib.dump(model, MODEL_DIR / "final_model.joblib")
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")

    cm = confusion_matrix(y_test, y_pred)
    with open(REPORTS_DIR / "confusion_matrix.txt", "w") as f:
        f.write(np.array2string(cm))

    with open(REPORTS_DIR / "training_report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Macro: {f1_macro:.4f}\n")
        f.write(f"F1 Weighted: {f1_weighted:.4f}\n\n")
        f.write(classification_report(y_test, y_pred, target_names=le.classes_))

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
