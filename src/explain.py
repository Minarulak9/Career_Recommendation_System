"""
Final robust explain.py for your project.
Usage:
    python src/explain.py --index 0
    python src/explain.py --user_id <UserID>
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
import warnings
from scipy import sparse

warnings.filterwarnings("ignore")

# -----------------------------------------------------------
# PROJECT PATHS (matches train.py)
# -----------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data.csv"
MODEL_DIR = ROOT / "models"

MODEL_PATHS = [
    MODEL_DIR / "tuned_model.joblib",
    MODEL_DIR / "final_model.joblib",
    MODEL_DIR / "xgb_model.joblib",
    MODEL_DIR / "model.joblib",
]

LABEL_ENCODER_PATHS = [
    MODEL_DIR / "label_encoder.joblib",
    MODEL_DIR / "label_encoder.pkl",
]

TARGET_COL = "Target Job Role"

# -----------------------------------------------------------
# SKILLS DEFINITIONS
# -----------------------------------------------------------
ROLE_SKILLS = {
    "Software Engineer": ["data structures", "algorithms", "programming", "python", "java", "c++", "debugging", "apis", "git"],
    "Frontend Developer": ["html", "css", "javascript", "react", "ui design"],
    "Backend Developer": ["databases", "sql", "node.js", "apis"],
    "AI Engineer": ["python", "machine learning", "statistics", "tensorflow"],
    "Data Analyst": ["excel", "sql", "power bi", "statistics"],
    "Data Scientist": ["python", "machine learning", "statistics"],
    "Product Manager": ["communication", "roadmap", "user research"]
}

SOFT_SKILLS_KEYWORDS = [
    "communication", "teamwork", "leadership", "problem solving",
    "time management", "adaptability", "creativity"
]

SKILLS_FIELDS = [
    "Technical Skills", "Soft Skills", "Course Keywords", "Project Keywords",
    "Preferred Roles", "Work Keywords", "Skill Embedding",
    "Graduate Major", "PG Major", "Highest Education"
]

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------
def load_model():
    for p in MODEL_PATHS:
        if p.exists():
            print(f"Loaded model from: {p}")
            return joblib.load(p)
    raise FileNotFoundError("No model found in models/ directory.")

def load_label_encoder():
    for p in LABEL_ENCODER_PATHS:
        if p.exists():
            print(f"Loaded label encoder from: {p}")
            return joblib.load(p)
    return None

def load_data(path=DATA_PATH):
    return pd.read_csv(path)

def normalize_split(text):
    if pd.isna(text):
        return set()
    s = str(text)
    for sep in [",", ";", "|", "/", " and ", "\n"]:
        s = s.replace(sep, ",")
    return {tok.strip().lower() for tok in s.split(",") if tok.strip()}

def extract_user_skills(row):
    out = set()
    for f in SKILLS_FIELDS:
        if f in row:
            out |= normalize_split(row.get(f, ""))

    for f in ["Graduate Major", "PG Major", "Highest Education"]:
        if f in row and not pd.isna(row[f]):
            out.add(str(row[f]).strip().lower())

    soft = {x for x in out if x in SOFT_SKILLS_KEYWORDS}
    return {"all": out, "soft": soft}

def compute_gap(user_set, required):
    req = {r.lower() for r in required}
    have = {r for r in req if any((r in u) or (u in r) for u in user_set)}
    missing = sorted(list(req - have))
    return {"have": sorted(list(have)), "missing": missing}

# -----------------------------------------------------------
# Extract full feature names from preprocessor
# -----------------------------------------------------------
def get_feature_names(preprocessor):
    names = []

    if not hasattr(preprocessor, "transformers_"):
        raise RuntimeError("Preprocessor must be fitted.")

    for name, trans, cols in preprocessor.transformers_:
        if trans == "drop" or trans is None:
            continue

        obj = trans
        if hasattr(obj, "named_steps"):  # pipeline inside ColumnTransformer
            steps = obj.named_steps

            # TF-IDF
            if "tfidf" in steps:
                tf = steps["tfidf"]
                try:
                    feats = tf.get_feature_names_out()
                except:
                    feats = tf.get_feature_names()
                base = cols[0] if isinstance(cols, (list, tuple)) else cols
                names.extend([f"{base}:{f}" for f in feats])
                continue

            # OneHot
            if "onehot" in steps:
                ohe = steps["onehot"]
                try:
                    feats = ohe.get_feature_names_out(cols)
                except:
                    feats = ohe.get_feature_names(cols)
                names.extend(list(feats))
                continue

            # numeric
            if isinstance(cols, (list, tuple)):
                names.extend(list(cols))
            else:
                names.append(cols)

        else:
            try:
                feats = obj.get_feature_names_out()
                names.extend(list(feats))
            except Exception:
                if isinstance(cols, (list, tuple)):
                    names.extend(list(cols))
                else:
                    names.append(cols)

    return names

# -----------------------------------------------------------
# SHAP on transformed space
# -----------------------------------------------------------
def build_shap_explainer(clf, preprocessor, Xraw, bg_size=50):
    bg_raw = Xraw.sample(n=min(bg_size, len(Xraw)), random_state=42)
    bg_trans = preprocessor.transform(bg_raw)

    if sparse.issparse(bg_trans):
        bg_trans = bg_trans.toarray()
    else:
        bg_trans = np.asarray(bg_trans)

    feature_names = get_feature_names(preprocessor)
    # align length
    if len(feature_names) != bg_trans.shape[1]:
        if len(feature_names) > bg_trans.shape[1]:
            feature_names = feature_names[:bg_trans.shape[1]]
        else:
            feature_names += [f"f_{i}" for i in range(len(feature_names), bg_trans.shape[1])]

    def predict_proba_from_transformed(x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return clf.predict_proba(x)

    explainer = shap.KernelExplainer(predict_proba_from_transformed, bg_trans, link="logit")
    return explainer, feature_names

# -----------------------------------------------------------
# Safe scalar converter (fixes all previous SHAP errors)
# -----------------------------------------------------------
def to_scalar(v):
    try:
        arr = np.array(v).astype(float).ravel()
        return float(arr[0])
    except:
        try:
            return float(v)
        except:
            return 0.0

# -----------------------------------------------------------
# MAIN EXPLAIN FUNCTION
# -----------------------------------------------------------
def run_explain(index=None, user_id=None):
    df = load_data()
    model = load_model()
    le = load_label_encoder()

    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["clf"]

    if not hasattr(pre, "transformers_"):
        Xall = df.drop(columns=[TARGET_COL], errors="ignore")
        pre.fit(Xall)

    Xraw = df.drop(columns=[TARGET_COL], errors="ignore")

    explainer, feat_names = build_shap_explainer(clf, pre, Xraw)

    # select row
    if index is not None:
        row = Xraw.iloc[[index]].reset_index(drop=True)
    else:
        sel = df[df["User ID"] == user_id]
        if sel.empty:
            raise ValueError("User ID not found")
        row = sel.drop(columns=[TARGET_COL]).iloc[[0]].reset_index(drop=True)

    # transform
    x_t = pre.transform(row)
    if sparse.issparse(x_t):
        x_t = x_t.toarray()
    x_t = np.asarray(x_t)

    # prediction
    proba = clf.predict_proba(x_t)[0]
    pred_idx = int(np.argmax(proba))
    pred_prob = float(proba[pred_idx])

    if le:
        pred_role = le.inverse_transform([pred_idx])[0]
    else:
        pred_role = clf.classes_[pred_idx]

    # SHAP
    try:
        shap_vals = explainer.shap_values(x_t, nsamples=200)
        if isinstance(shap_vals, list):
            sv = np.array(shap_vals[pred_idx])[0]
        else:
            sv = np.array(shap_vals)[0]
        feat_shap = list(zip(feat_names, sv.tolist()))
        feat_sorted = sorted(feat_shap, key=lambda x: abs(to_scalar(x[1])), reverse=True)
        why = [
            f"{fname} {'increased' if to_scalar(val)>0 else 'decreased'} probability by {abs(to_scalar(val)):.4f}"
            for fname, val in feat_sorted[:10]
        ]
    except Exception as e:
        feat_sorted = None
        why = [f"SHAP explanation failed: {e}"]

    # skill extraction
    original_row = df.iloc[index] if index is not None else sel.iloc[0]
    user = extract_user_skills(original_row)
    user_all = user["all"]
    user_soft = sorted(list(user["soft"]))

    gap = compute_gap(user_all, ROLE_SKILLS.get(pred_role, []))

    # alternatives
    alts = []
    for role, skills in ROLE_SKILLS.items():
        if role == pred_role:
            continue
        overlap = len(set(skills) & user_all)
        rgap = compute_gap(user_all, skills)
        alts.append({
            "role": role,
            "overlap": overlap,
            "missing": rgap["missing"],
            "have": rgap["have"]
        })
    alts = sorted(alts, key=lambda x: x["overlap"], reverse=True)[:3]

    out = {
        "predicted_role": pred_role,
        "predicted_probability": pred_prob,
        "why": why,
        "top_shap_features": feat_sorted[:10] if feat_sorted else None,
        "user_skills_sample": sorted(list(user_all)),
        "user_soft_skills": user_soft,
        "missing_skills_for_predicted_role": gap["missing"],
        "alternatives": alts
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out

# -----------------------------------------------------------
# CLI ENTRY
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int)
    g.add_argument("--user_id", type=str)
    args = parser.parse_args()
    run_explain(index=args.index, user_id=args.user_id)
