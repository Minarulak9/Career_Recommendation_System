import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb

from pipeline import build_preprocessor, drop_features

# AUTO-DETECT PROJECT ROOT
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data.csv"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


TARGET_COL = "Target Job Role"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=drop_features, errors="ignore")
    return df


def main():
    print("Loading:", DATA_PATH)
    df = load_data()
    print("Shape:", df.shape)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # label encode target
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # build preprocessor
    preprocessor = build_preprocessor()

    # final model
    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=300,   # SAFE FOR LAPTOP
        max_depth=5,
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # pipeline (preprocess + model)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("Accuracy:", acc)
    print("Macro F1:", f1)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # save model + label encoder
    joblib.dump(model, MODEL_DIR / "final_model.joblib")
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")

    print("Model saved at:", MODEL_DIR / "final_model.joblib")


if __name__ == "__main__":
    main()
