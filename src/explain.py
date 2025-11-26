"""
EXPLAIN.PY — ENTERPRISE VERSION (FORMAL OUTPUT + LEARNING RESOURCES)

Fully compatible with upgraded SkillsEngine:
- Detailed JSON output
- Formal academic paragraph
- Learning roadmap (courses + projects + duration)
- SHAP explanations (TreeExplainer + fallback)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
from scipy import sparse
import warnings

warnings.filterwarnings("ignore")

# Import upgraded skill engine
from skills_engine import SkillsEngine

# Import preprocessing pipeline functions
from skill_features import extract_skill_flags
from pipeline import ensure_full_schema

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "final_training_dataset.csv"
MODEL_DIR = ROOT / "models"

MODEL_CANDIDATES = [
    MODEL_DIR / "tuned_model.joblib",
    MODEL_DIR / "final_model.joblib",
]

LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"

TARGET_COL = "Target Job Role"

# ============================================================
# LOADERS
# ============================================================
def load_model():
    for p in MODEL_CANDIDATES:
        if p.exists():
            print(f"Loaded model from: {p}")
            return joblib.load(p)
    raise FileNotFoundError("Model not found.")

def load_label_encoder():
    if LABEL_ENCODER_PATH.exists():
        print(f"Loaded label encoder from: {LABEL_ENCODER_PATH}")
        return joblib.load(LABEL_ENCODER_PATH)
    raise FileNotFoundError("LabelEncoder missing.")

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = extract_skill_flags(df)
    df = ensure_full_schema(df)
    return df

# ============================================================
# FEATURE NAME EXTRACTOR
# ============================================================
def get_feature_names(pre):
    names = []
    for name, trans, cols in pre.transformers_:
        if trans in ("drop", None):
            continue

        if hasattr(trans, "named_steps"):

            # OneHotEncoder
            if "onehot" in trans.named_steps:
                ohe = trans.named_steps["onehot"]
                names.extend(ohe.get_feature_names_out(cols).tolist())

            # HashingVectorizer
            elif "hashing" in trans.named_steps:
                n = trans.named_steps["hashing"].n_features
                base = cols[0]
                names.extend([f"{base}_hash_{i}" for i in range(n)])

            else:
                names.extend(cols)

        else:
            names.extend(cols)

    return names

# ============================================================
# SHAP EXPLAINER
# ============================================================
def build_shap_explainer(clf, pre, Xraw):
    print("\nBuilding SHAP explainer...")

    try:
        explainer = shap.TreeExplainer(clf)
        return explainer, "tree"
    except:
        print("TreeExplainer failed → Using KernelExplainer (slow).")

        bg = Xraw.sample(n=min(40, len(Xraw)), random_state=42)
        bg_t = pre.transform(bg)

        if sparse.issparse(bg_t):
            bg_t = bg_t.toarray()

        def predict_fn(x):
            x = np.array(x)
            return clf.predict_proba(x)

        explainer = shap.KernelExplainer(predict_fn, bg_t)
        return explainer, "kernel"

# ============================================================
# MAIN EXPLANATION FUNCTION
# ============================================================
def run_explain(index=None, user_id=None):

    # Load everything
    df = load_data()
    model = load_model()
    le = load_label_encoder()
    engine = SkillsEngine()

    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["clf"]
    Xraw = df.drop(columns=[TARGET_COL])

    # Select sample
    if index is not None:
        row_raw = df.iloc[index]
        row = Xraw.iloc[[index]]
    else:
        s = df[df["User ID"] == user_id]
        if s.empty:
            raise ValueError(f"User ID '{user_id}' not found.")
        row_raw = s.iloc[0]
        row = s.drop(columns=[TARGET_COL]).iloc[[0]]

    # Transform row
    xx = pre.transform(row)
    if sparse.issparse(xx):
        xx = xx.toarray()

    # Prediction
    proba = clf.predict_proba(xx)[0]
    pred_idx = int(np.argmax(proba))
    pred_role = le.inverse_transform([pred_idx])[0]
    pred_prob = float(proba[pred_idx])

    # SHAP explanations
    explainer, mode = build_shap_explainer(clf, pre, Xraw)
    feature_names = get_feature_names(pre)

    try:
        vals = explainer.shap_values(xx)
        if isinstance(vals, list):
            sv = vals[pred_idx][0]
        else:
            sv = vals[0]

        ranked = sorted(
            zip(feature_names, sv.tolist()),
            key=lambda z: abs(z[1]),
            reverse=True
        )

        top_reasons = [
            f"{name} (impact {round(val,3)})"
            for name, val in ranked[:5]
        ]

    except Exception:
        top_reasons = [f"The feature profile suggests alignment with {pred_role}."]

    # Skill extraction
    detected = engine.extract_from_row(row_raw)
    seniority = engine.seniority_estimate(detected)

    # Gap analysis
    gaps = engine.compute_gap(detected, pred_role)
    total_missing = (
        len(gaps["critical"]["missing"]) +
        len(gaps["important"]["missing"])
    )

    # Learning resources
    missing_skills = (
        gaps["critical"]["missing"] +
        gaps["important"]["missing"]
    )
    learning_roadmap = engine.learning_path(missing_skills)

    # Role match score
    match_score = engine.compute_role_match(detected, pred_role)

    # Project recommendation
    flagship_project = engine.recommend_project(pred_role)

    # Alternatives
    alternatives = engine.alternatives(detected, exclude=pred_role)

    # Effort estimation
    effort_required = engine.estimate_effort(total_missing)

    # ----------------------------------------------------------
    # FORMAL PARAGRAPH
    # ----------------------------------------------------------
    paragraph = (
        f"Based on a formal evaluation of your technical profile, skill indicators, "
        f"and experience attributes, the predicted role is '{pred_role}' with a "
        f"confidence level of {pred_prob * 100:.1f}%. The assessment identifies "
        f"notable strengths in several foundational areas; however, development is "
        f"recommended in crucial skills such as "
        f"{', '.join(gaps['critical']['missing'][:2]) if gaps['critical']['missing'] else 'core domain fundamentals'}. "
        f"Your current competency level is classified as '{seniority}', and the "
        f"proposed learning roadmap provides a structured path to strengthen readiness "
        f"for this career direction."
    )

    # ----------------------------------------------------------
    # JSON OUTPUT
    # ----------------------------------------------------------
    output = {
        "summary": {
            "predicted_role": pred_role,
            "confidence": f"{pred_prob * 100:.1f}%",
            "match_score": f"{match_score}%",
            "seniority": seniority,
            "formal_explanation": paragraph,
        },

        "prediction_reasons": top_reasons,

        "skills_detected": sorted(list(detected)),

        "skill_gaps": gaps,

        "learning_path": {
            "skills_based_courses_projects": learning_roadmap,
            "flagship_project": flagship_project,
            "effort_required": effort_required,
        },

        "alternative_roles": [
            {
                "role": r,
                "match_score": f"{score}%"
            }
            for r, score in alternatives
        ],
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return output

# ============================================================
# CLI ENTRY
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int)
    g.add_argument("--user_id", type=str)
    args = parser.parse_args()

    run_explain(index=args.index, user_id=args.user_id)
