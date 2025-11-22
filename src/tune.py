# tune.py
"""
Hyperparameter tuning with RandomizedSearchCV.
- Broader search space
- Better scoring
- Saves best model
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
import xgboost as xgb
import warnings

from pipeline import build_preprocessor, drop_features

warnings.filterwarnings("ignore")

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET_COL = "Target Job Role"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=drop_features, errors="ignore")
    df = df.dropna(subset=[TARGET_COL])
    return df


def main():
    print("=" * 50)
    print("HYPERPARAMETER TUNING")
    print("=" * 50)
    
    # Load data
    print("\n[1/4] Loading data...")
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
    
    # Build pipeline
    print("\n[2/4] Building pipeline...")
    preprocessor = build_preprocessor()
    
    base_clf = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False
    )
    
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", base_clf)
    ])
    
    # Search space (comprehensive)
    param_dist = {
        "clf__n_estimators": [100, 200, 300, 400, 500],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
        "clf__max_depth": [3, 4, 5, 6, 7, 8],
        "clf__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "clf__gamma": [0, 0.1, 0.2, 0.3, 0.5],
        "clf__min_child_weight": [1, 2, 3, 5],
        "clf__reg_alpha": [0, 0.1, 0.5, 1.0],
        "clf__reg_lambda": [0.5, 1.0, 1.5, 2.0]
    }
    
    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Search
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=30,  # Number of combinations to try
        scoring="f1_macro",
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=42,
        return_train_score=True
    )
    
    print("\n[3/4] Running search (this may take 10-20 minutes)...")
    search.fit(X_train, y_train)
    
    # Results
    print("\n" + "=" * 50)
    print("TUNING RESULTS")
    print("=" * 50)
    print(f"\nBest CV Score (F1 Macro): {search.best_score_:.4f}")
    print(f"\nBest Parameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    
    # Evaluate on test set
    y_pred = search.best_estimator_.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\nTest Set F1 (Macro): {test_f1:.4f}")
    
    # Save
    print("\n[4/4] Saving model...")
    joblib.dump(search.best_estimator_, MODEL_DIR / "tuned_model.joblib")
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")
    
    print(f"\n✅ Tuned model saved: {MODEL_DIR / 'tuned_model.joblib'}")


if __name__ == "__main__":
    main()