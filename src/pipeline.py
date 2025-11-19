import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


class TextSelector(BaseEstimator, TransformerMixin):
    """Select a single text column."""
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key].fillna("").astype(str).values


drop_features = [
    'User ID', 'Timestamp', 'Skill Embedding',
    'Course Keywords', 'Project Keywords', 'Work Keywords'
]

numeric_features = [
    'Age','Class 10 Percentage','Class 12 Percentage','Graduate CGPA','PG CGPA',
    'Academic Consistency','Tech Skill Proficiency','Soft Skill Proficiency',
    'Courses Completed','Avg Course Difficulty','Total Hours Learning',
    'Project Count','Avg Project Complexity','Experience Months',
    'Interest STEM','Interest Business','Interest Arts','Interest Design',
    'Interest Medical','Interest Social Science','Conscientiousness',
    'Extraversion','Openness','Agreeableness','Emotional Stability'
]

categorical_features = [
    'Gender','Location','Class 12 Stream','Graduate Major','PG Major',
    'Highest Education','Technical Skills','Soft Skills','Experience Types',
    'Job Level','Career Preference','Work Preference','Preferred Roles',
    'Current Status'
]

text_features = [
    'Languages Spoken', 'Preferred Industries'
]


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    text_pipelines = []
    for col in text_features:
        text_pipelines.append(
            (
                f"tfidf_{col}",
                Pipeline([
                    ("selector", TextSelector(col)),
                    ("tfidf", TfidfVectorizer(max_features=1000))   # SAFE FOR YOUR LAPTOP
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
        remainder="drop"
    )

    return preprocessor


"""
   What this pipeline does
✔ Drops useless columns

(you will drop them in train.py)

✔ Numeric → median impute + StandardScaler
✔ Categorical → impute “Missing” + OneHot
✔ Text → TF-IDF (2000 features max per column)
✔ Output is ready for XGBoost / RandomForest / any model
"""