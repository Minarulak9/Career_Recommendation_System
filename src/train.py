# src/train.py
"""
FINAL TRAINING SCRIPT (STABLE + SAFE)
------------------------------------------
This version:
- Does NOT drop valid categorical columns
- Avoids passing strings to XGBoost
- Uses skill_features + updated pipeline
- Works with tune.py (same preprocessing structure)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

from pipeline import (
    build_preprocessor,
    drop_features,
    ensure_full_schema,
    DEFAULT_NUMERIC,
    DEFAULT_CATEGORICAL,
    DEFAULT_TEXT
)
from skill_features import extract_skill_flags

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "final_training_dataset.csv"


MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

TARGET_COL = "Target Job Role"


# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=drop_features, errors="ignore")
    df = df.dropna(subset=[TARGET_COL])
    print(f"\nLoaded {len(df)} samples from {DATA_PATH.name}")
    return df


# ============================================================
# TRAINING FUNCTION
# ============================================================
def main():
    print("=" * 70)
    print("TRAINING CAREER PREDICTION MODEL (FINAL SAFE VERSION)")
    print("=" * 70)

    print("[1/6] Loading dataset...")
    df = load_data()

    print("[2/6] Extracting skill features...")
    df = extract_skill_flags(df)

    print("[3/6] Preparing X and y...")

    # Backup original label
    df["__original_label"] = df[TARGET_COL]

    # Separate target
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Remove helper column from training input
    X = X.drop(columns=["__original_label"], errors="ignore")

    # Identify engineered skill columns
    engineered_cols = [
        c for c in X.columns
        if c.startswith("skill_") or c == "total_skill_hits"
    ]

    # Build numeric_features (do NOT mutate DEFAULT_NUMERIC)
    numeric_features = DEFAULT_NUMERIC.copy()
    numeric_features.extend(engineered_cols)

    # Guarantee base schema
    X = ensure_full_schema(X)

    # ====================================================
    # SAFE ALLOWED FEATURE LIST
    # NOTHING gets dropped incorrectly here
    # ====================================================
    ALLOWED = set(numeric_features + DEFAULT_CATEGORICAL + DEFAULT_TEXT + engineered_cols)

    # Drop only unknown garbage columns
    drop_cols = [col for col in X.columns if col not in ALLOWED]
    if drop_cols:
        print("Dropping unknown columns:", drop_cols)
        X = X.drop(columns=drop_cols)

    # Final check — categorical columns remain strings (that's fine)
    # Only numeric columns must be numeric
    # Categorical columns will be encoded by OneHotEncoder
    print(f"\nFinal feature count: {len(X.columns)}")

    # ====================================================
    # Label encode
    # ====================================================
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("Classes:", list(le.classes_))

    # ====================================================
    # Train/test split
    # ====================================================
    print("\n[4/6] Splitting training/testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ====================================================
    # Build preprocessor
    # ====================================================
    print("\n[5/6] Building preprocessor...")
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=DEFAULT_CATEGORICAL,
        text_features=DEFAULT_TEXT
    )

    # ====================================================
    # Class weights
    # ====================================================
    print("\n[6/6] Computing class weights...")
    classes = np.unique(y_train)
    base_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    base_map = {cls: w for cls, w in zip(classes, base_weights)}

    boost = {
        "Backend Developer": 2.0,
        "UX Designer": 2.5,
        "Project Manager": 1.2,
        "Software Engineer": 1.2
    }

    final_map = {
        int(cls): base_map[cls] * boost.get(le.inverse_transform([cls])[0], 1.0)
        for cls in classes
    }

    sample_weights = np.array([final_map[int(c)] for c in y_train])

    # ====================================================
    # Build model
    # ====================================================
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

    print("\nTraining model...")
    model.fit(X_train, y_train, clf__sample_weight=sample_weights)

    # ====================================================
    # Evaluation
    # ====================================================
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

    # Save artifacts
    joblib.dump(model, MODEL_DIR / "final_model.joblib")
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")

    cm = confusion_matrix(y_test, y_pred)
    with open(REPORTS_DIR / "confusion_matrix.txt", "w") as f:
        f.write(np.array2string(cm))

    print("\nTraining complete.\n")


if __name__ == "__main__":
    main()
