# pipeline.py
"""
Production-ready preprocessing pipeline.
- Handles numeric, categorical, and text features
- Optimized for large datasets
- Clean feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


class TextSelector(BaseEstimator, TransformerMixin):
    """Select and clean a single text column."""
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key].fillna("").astype(str).values


# Columns to drop (not useful for prediction)
drop_features = [
    'User ID', 'Timestamp', 'Skill Embedding',
    'Course Keywords', 'Project Keywords', 'Work Keywords'
]

# Numeric features
numeric_features = [
    'Age', 'Class 10 Percentage', 'Class 12 Percentage',
    'Graduate CGPA', 'PG CGPA', 'Academic Consistency',
    'Tech Skill Proficiency', 'Soft Skill Proficiency',
    'Courses Completed', 'Avg Course Difficulty', 'Total Hours Learning',
    'Project Count', 'Avg Project Complexity', 'Experience Months',
    'Interest STEM', 'Interest Business', 'Interest Arts',
    'Interest Design', 'Interest Medical', 'Interest Social Science',
    'Conscientiousness', 'Extraversion', 'Openness',
    'Agreeableness', 'Emotional Stability'
]

# Categorical features
categorical_features = [
    'Gender', 'Location', 'Class 12 Stream',
    'Graduate Major', 'PG Major', 'Highest Education',
    'Technical Skills', 'Soft Skills', 'Experience Types',
    'Job Level', 'Career Preference', 'Work Preference',
    'Preferred Roles', 'Current Status'
]

# Text features (for TF-IDF)
text_features = [
    'Languages Spoken', 'Preferred Industries'
]


def build_preprocessor():
    """Build the complete preprocessing pipeline."""
    
    # Numeric: median impute + scale
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical: impute missing + one-hot encode
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            min_frequency=5  # Ignore rare categories (helps with large datasets)
        ))
    ])

    # Text: TF-IDF vectorization
    text_pipelines = []
    for col in text_features:
        text_pipelines.append(
            (
                f"tfidf_{col}",
                Pipeline([
                    ("selector", TextSelector(col)),
                    ("tfidf", TfidfVectorizer(
                        max_features=500,      # Reduced for efficiency
                        ngram_range=(1, 2),    # Unigrams + bigrams
                        min_df=2,              # Ignore very rare terms
                        max_df=0.95            # Ignore too common terms
                    ))
                ]),
                [col]
            )
        )

    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
            *text_pipelines
        ],
        remainder="drop",
        n_jobs=-1  # Parallel processing
    )

    return preprocessor


def get_feature_names_from_preprocessor(preprocessor):
    """Extract feature names after fitting."""
    names = []
    
    for name, trans, cols in preprocessor.transformers_:
        if trans == "drop" or trans is None:
            continue

        if hasattr(trans, "named_steps"):
            steps = trans.named_steps
            
            if "tfidf" in steps:
                tf = steps["tfidf"]
                feats = tf.get_feature_names_out()
                base = cols[0] if isinstance(cols, (list, tuple)) else cols
                names.extend([f"{base}:{f}" for f in feats])
                
            elif "onehot" in steps:
                ohe = steps["onehot"]
                feats = ohe.get_feature_names_out(cols)
                names.extend(list(feats))
                
            else:
                names.extend(list(cols) if isinstance(cols, list) else [cols])
        else:
            try:
                feats = trans.get_feature_names_out()
                names.extend(list(feats))
            except:
                names.extend(list(cols) if isinstance(cols, list) else [cols])

    return names