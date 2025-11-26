# src/tune.py
"""
FAST HYPERPARAMETER TUNING
-------------------------------------
Optimized for:
- Intel HD Graphics 620 (no GPU)
- i5-7300U (4 threads)
- 16GB RAM

Uses CPU hist algorithm (fastest on non-NVIDIA machines)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import xgboost as xgb
import warnings

from pipeline import build_preprocessor, drop_features, ensure_full_schema
from skill_features import extract_skill_flags

warnings.filterwarnings("ignore")

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "final_training_dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET_COL = "Target Job Role"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=drop_features, errors="ignore")
    df = extract_skill_flags(df)
    df = ensure_full_schema(df)
    df = df.dropna(subset=[TARGET_COL])
    return df


def main():
    print("=" * 50)
    print("FAST HYPERPARAMETER TUNING (CPU ONLY)")
    print("=" * 50)

    # Load data
    print("\n[1/4] Loading & preparing data...")
    df = load_data()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42
    )

    print("\n[2/4] Building pipeline...")
    preprocessor = build_preprocessor()

    # Base model (FAST CPU SETTINGS)
    base_clf = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",      # FASTEST on CPU
        max_bin=256,
        n_jobs=-1,               # Use all CPU threads
        random_state=42
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", base_clf)
    ])

    # FAST Search space
    param_dist = {
        "clf__n_estimators": [100, 200, 300],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__max_depth": [4, 5, 6],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
        "clf__gamma": [0, 0.1, 0.2],
        "clf__min_child_weight": [1, 2, 3],
        "clf__reg_alpha": [0, 0.1, 0.5],
        "clf__reg_lambda": [1.0, 1.5, 2.0]
    }

    # FAST CV (3 folds instead of 5)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scorer = "f1_macro"

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=20,  # Reduced for speed
        scoring=scorer,
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=42,
        return_train_score=False
    )

    print("\n[3/4] Running FAST search...")
    search.fit(X_train, y_train)

    print("\n" + "=" * 50)
    print("TUNING RESULTS")
    print("=" * 50)
    print(f"\nBest CV Score (F1 Macro): {search.best_score_:.4f}")
    print("\nBest Parameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    # Evaluate
    y_pred = search.best_estimator_.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\nTest Set F1 (Macro): {test_f1:.4f}")

    print("\n[4/4] Saving best tuned model...")
    joblib.dump(search.best_estimator_, MODEL_DIR / "tuned_model.joblib")
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")

    print("\n✅ Tuned model saved successfully!")


if __name__ == "__main__":
    main()
