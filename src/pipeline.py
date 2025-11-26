"""
FINAL STABLE PIPELINE
-----------------------------------
- HashingVectorizer (no TF-IDF issues)
- TextCleaner for safe text processing
- Version-safe OneHotEncoder + SimpleImputer
- Schema validator that preserves engineered columns
"""

import inspect
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder as _OHE

# -----------------------
# STATIC SCHEMA FEATURES
# -----------------------

drop_features = [
    "User ID", "Timestamp", "Skill Embedding",
    "Course Keywords", "Project Keywords", "Work Keywords"
]

numeric_features = [
    "Age", "Class 10 Percentage", "Class 12 Percentage",
    "Graduate CGPA", "PG CGPA", "Academic Consistency",
    "Tech Skill Proficiency", "Soft Skill Proficiency",
    "Courses Completed", "Avg Course Difficulty", "Total Hours Learning",
    "Project Count", "Avg Project Complexity", "Experience Months",
    "Interest STEM", "Interest Business", "Interest Arts",
    "Interest Design", "Interest Medical", "Interest Social Science",
    "Conscientiousness", "Extraversion", "Openness",
    "Agreeableness", "Emotional Stability"
]

categorical_features = [
    "Gender", "Location", "Class 12 Stream",
    "Graduate Major", "PG Major", "Highest Education",
    "Technical Skills", "Soft Skills", "Experience Types",
    "Job Level", "Career Preference", "Work Preference",
    "Preferred Industries", "Current Status"
]

text_features = [
    "Languages Spoken", "Preferred Industries"
]


# -------------------------------
# Version-safe OneHotEncoder
# -------------------------------
def _make_onehot_encoder():
    sig = inspect.signature(_OHE)
    params = sig.parameters

    if "sparse_output" in params:
        return _OHE(handle_unknown="ignore", sparse_output=False)
    if "sparse" in params:
        return _OHE(handle_unknown="ignore", sparse=False)
    return _OHE(handle_unknown="ignore")


# --------------------------
# Version-safe SimpleImputer
# --------------------------
def _make_safe_imputer():
    sig = inspect.signature(SimpleImputer)
    params = sig.parameters

    if "keep_empty_features" in params:
        return SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)

    return SimpleImputer(strategy="constant", fill_value="Missing")


# -------------------------
# Text Cleaning
# -------------------------
class TextCleaner(BaseEstimator, TransformerMixin):
    """Clean text safely to avoid empty TF problems."""
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = (
            X[self.key].astype(str).fillna("")
            .str.replace(r"[^A-Za-z0-9 ]+", " ", regex=True)
            .str.replace("&", " and ", regex=False)
            .str.lower().str.strip()
        )

        if s.str.strip().eq("").all():
            return ["unknowntext"] * len(s)

        s = s.replace("", "unknowntext")
        return s.tolist()


# -------------------------
# Schema Validation
# -------------------------
def get_expected_columns():
    return numeric_features + categorical_features + text_features


def ensure_full_schema(df: pd.DataFrame):
    """Ensure required base columns exist but KEEP extra engineered columns."""
    df = df.copy()
    expected = get_expected_columns()

    for col in expected:
        if col not in df.columns:
            df[col] = np.nan

    return df  # DO NOT filter columns → preserves engineered skill flags


# -------------------------
# Build Preprocessor
# -------------------------
def build_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", _make_safe_imputer()),
        ("onehot", _make_onehot_encoder())
    ])

    text_pipelines = []
    for col in text_features:
        text_pipelines.append(
            (
                f"hash_{col}",
                Pipeline([
                    ("selector", TextCleaner(col)),
                    ("hashing", HashingVectorizer(
                        n_features=512,
                        alternate_sign=False,
                        norm="l2"
                    ))
                ]),
                [col]
            )
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
            *text_pipelines
        ],
        remainder="passthrough",
        n_jobs=-1
    )

    return preprocessor
