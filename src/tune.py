# tune.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import xgboost as xgb

from pipeline import build_preprocessor, drop_features

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET_COL = "Target Job Role"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=drop_features, errors="ignore")
    return df


def main():
    print("Loading:", DATA_PATH)
    df = load_data()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor()

    base_clf = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", base_clf)
    ])

    # Hyperparameter search space
    param_dist = {
        "clf__n_estimators": [100, 200, 300, 400, 500],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
        "clf__max_depth": [3, 4, 5, 6, 7],
        "clf__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "clf__gamma": [0, 0.1, 0.2, 0.3],
        "clf__min_child_weight": [1, 2, 3, 4]
    }

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1_macro",
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    print("Running hyperparameter search (this may take a few minutes)...")
    search.fit(X_train, y_train)

    print("Best parameters:")
    print(search.best_params_)

    print("Best CV score (macro F1):", search.best_score_)

    # Save tuned model
    joblib.dump(search.best_estimator_, MODEL_DIR / "tuned_model.joblib")
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")

    print("Saved tuned model to:", MODEL_DIR / "tuned_model.joblib")


if __name__ == "__main__":
    main()
