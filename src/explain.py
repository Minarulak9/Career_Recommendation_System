# explain.py (ENHANCED VERSION)
"""
Enhanced explanation engine with:
1. ✅ Better "Why" explanations (human-readable)
2. ✅ Removed redundant SHAP arrays
3. ✅ Skill gaps with priorities (critical/important/nice-to-have)
4. ✅ Learning paths with duration estimates
5. ✅ Better alternatives format with match scores
6. ✅ Summary section with actionable insights
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

# Import enhanced skills engine
from skills_engine import (
    SkillsEngine,
    extract_skills_from_row,
    ROLE_SKILLS
)

# -----------------------------------------------------------
# PROJECT PATHS
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

def get_feature_names(preprocessor):
    """Extract feature names from fitted preprocessor"""
    names = []
    if not hasattr(preprocessor, "transformers_"):
        raise RuntimeError("Preprocessor must be fitted.")

    for name, trans, cols in preprocessor.transformers_:
        if trans == "drop" or trans is None:
            continue

        obj = trans
        if hasattr(obj, "named_steps"):
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

            # Numeric
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

def build_shap_explainer(clf, preprocessor, Xraw, bg_size=50):
    """Build SHAP explainer on transformed feature space"""
    bg_raw = Xraw.sample(n=min(bg_size, len(Xraw)), random_state=42)
    bg_trans = preprocessor.transform(bg_raw)

    if sparse.issparse(bg_trans):
        bg_trans = bg_trans.toarray()
    else:
        bg_trans = np.asarray(bg_trans)

    feature_names = get_feature_names(preprocessor)
    
    # Align feature names length
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

def to_scalar(v):
    """Safe conversion to scalar (fixes SHAP array issues)"""
    try:
        arr = np.array(v).astype(float).ravel()
        return float(arr[0])
    except:
        try:
            return float(v)
        except:
            return 0.0

# -----------------------------------------------------------
# IMPROVEMENT 1: Human-Readable "Why" Explanations
# -----------------------------------------------------------
def humanize_shap_feature(feature_name, shap_value):
    """Convert raw SHAP feature to human-readable explanation"""
    val = to_scalar(shap_value)
    direction = "increased" if val > 0 else "decreased"
    
    # Extract meaningful parts
    if "Preferred Roles_" in feature_name:
        role = feature_name.replace("Preferred Roles_", "")
        return f"Listed '{role}' as preferred role ({direction} prediction)"
    
    elif "Technical Skills" in feature_name or "Soft Skills" in feature_name:
        skill = feature_name.split("_")[-1] if "_" in feature_name else feature_name
        return f"Skill: {skill} ({direction} match)"
    
    elif "Job Level_" in feature_name:
        level = feature_name.replace("Job Level_", "")
        return f"Current level: {level} ({direction} prediction)"
    
    elif "Experience Months" in feature_name:
        return f"Work experience duration ({direction} prediction)"
    
    elif "Languages Spoken:" in feature_name:
        lang = feature_name.split(":")[-1]
        return f"Language: {lang} ({direction} relevance)"
    
    elif "Preferred Industries:" in feature_name:
        ind = feature_name.split(":")[-1]
        return f"Industry interest: {ind} ({direction} alignment)"
    
    elif "CGPA" in feature_name or "Percentage" in feature_name:
        return f"Academic performance ({direction} prediction)"
    
    elif "Project Count" in feature_name or "Project" in feature_name:
        return f"Project experience ({direction} match)"
    
    else:
        return f"{feature_name} ({direction} prediction)"

# -----------------------------------------------------------
# IMPROVEMENT 3: Structured Skill Gaps
# -----------------------------------------------------------
def structure_skill_gaps(gap_analysis):
    """Convert gap analysis to structured format with priorities"""
    return {
        "critical": {
            "have": gap_analysis["critical"]["have"],
            "missing": gap_analysis["critical"]["missing"]
        },
        "important": {
            "have": gap_analysis["important"]["have"],
            "missing": gap_analysis["important"]["missing"]
        },
        "nice_to_have": {
            "have": gap_analysis["nice_to_have"]["have"],
            "missing": gap_analysis["nice_to_have"]["missing"]
        }
    }

# -----------------------------------------------------------
# IMPROVEMENT 4: Learning Recommendations
# -----------------------------------------------------------
def generate_learning_recommendations(missing_skills, skills_engine, top_n=5):
    """Generate top N learning recommendations with resources"""
    # Prioritize critical skills
    critical = missing_skills.get("critical", {}).get("missing", [])
    important = missing_skills.get("important", {}).get("missing", [])
    
    # Take top critical skills first
    priority_skills = critical[:3] + important[:2]
    priority_skills = priority_skills[:top_n]
    
    learning_paths = skills_engine.make_learning_paths(priority_skills)
    
    recommendations = {}
    for skill, path_info in learning_paths.items():
        recommendations[skill] = {
            "resources": path_info["resources"],
            "duration": path_info["duration"],
            "difficulty": path_info["difficulty"],
            "priority": "Critical" if skill in critical else "Important"
        }
    
    return recommendations

# -----------------------------------------------------------
# IMPROVEMENT 5: Better Alternatives Format
# -----------------------------------------------------------
def format_alternatives(alternatives_raw, user_skills, skills_engine):
    """Format alternative roles with match scores and effort estimates"""
    formatted = []
    
    for role_name, overlap, total in alternatives_raw:
        gap = skills_engine.compute_gap_for_role(user_skills, role_name)
        
        all_missing = (
            gap["critical"]["missing"] + 
            gap["important"]["missing"]
        )
        
        all_have = (
            gap["critical"]["have"] + 
            gap["important"]["have"]
        )
        
        match_score = int((overlap / total) * 100) if total > 0 else 0
        effort = skills_engine.estimate_effort(len(all_missing))
        
        formatted.append({
            "role": role_name,
            "match_score": f"{match_score}%",
            "you_have": all_have[:5],  # Top 5 matching skills
            "you_need": all_missing[:5],  # Top 5 missing skills
            "effort": effort
        })
    
    return formatted

# -----------------------------------------------------------
# IMPROVEMENT 6: Summary Section
# -----------------------------------------------------------
def generate_summary(pred_role, pred_prob, user_skills, gap_analysis):
    """Generate executive summary with actionable insights"""
    confidence_level = "High" if pred_prob > 0.8 else "Medium" if pred_prob > 0.6 else "Low"
    
    # Extract strengths (skills user has)
    strengths = (
        gap_analysis["critical"]["have"][:3] + 
        gap_analysis["important"]["have"][:2]
    )
    
    # Top priority (critical missing skills)
    critical_missing = gap_analysis["critical"]["missing"]
    top_priority = critical_missing[:2] if critical_missing else gap_analysis["important"]["missing"][:2]
    
    return {
        "predicted_role": pred_role,
        "confidence": f"{confidence_level} ({pred_prob*100:.1f}%)",
        "your_strengths": strengths if strengths else ["Adaptable learner"],
        "top_priority": top_priority if top_priority else ["Continue building on current skills"],
        "overall_readiness": calculate_readiness(gap_analysis)
    }

def calculate_readiness(gap_analysis):
    """Calculate overall readiness score"""
    critical_have = len(gap_analysis["critical"]["have"])
    critical_total = critical_have + len(gap_analysis["critical"]["missing"])
    
    if critical_total == 0:
        return "Ready to apply"
    
    readiness = (critical_have / critical_total) * 100
    
    if readiness >= 80:
        return "Strong match - Start applying!"
    elif readiness >= 60:
        return "Good match - Focus on 2-3 key skills"
    elif readiness >= 40:
        return "Developing - Need 3-6 months of focused learning"
    else:
        return "Early stage - Build foundational skills first"

# -----------------------------------------------------------
# MAIN EXPLAIN FUNCTION (ENHANCED)
# -----------------------------------------------------------
def run_explain(index=None, user_id=None):
    """Enhanced explanation with all 6 improvements"""
    
    # Load data and model
    df = load_data()
    model = load_model()
    le = load_label_encoder()
    skills_engine = SkillsEngine()

    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["clf"]

    if not hasattr(pre, "transformers_"):
        Xall = df.drop(columns=[TARGET_COL], errors="ignore")
        pre.fit(Xall)

    Xraw = df.drop(columns=[TARGET_COL], errors="ignore")

    # Select row
    if index is not None:
        row = Xraw.iloc[[index]].reset_index(drop=True)
        original_row = df.iloc[index]
    else:
        sel = df[df["User ID"] == user_id]
        if sel.empty:
            raise ValueError("User ID not found")
        row = sel.drop(columns=[TARGET_COL]).iloc[[0]].reset_index(drop=True)
        original_row = sel.iloc[0]

    # Transform and predict
    x_t = pre.transform(row)
    if sparse.issparse(x_t):
        x_t = x_t.toarray()
    x_t = np.asarray(x_t)

    proba = clf.predict_proba(x_t)[0]
    pred_idx = int(np.argmax(proba))
    pred_prob = float(proba[pred_idx])

    if le:
        pred_role = le.inverse_transform([pred_idx])[0]
    else:
        pred_role = clf.classes_[pred_idx]

    # SHAP explanation
    explainer, feat_names = build_shap_explainer(clf, pre, Xraw)
    
    try:
        shap_vals = explainer.shap_values(x_t, nsamples=200)
        if isinstance(shap_vals, list):
            sv = np.array(shap_vals[pred_idx])[0]
        else:
            sv = np.array(shap_vals)[0]
        
        feat_shap = list(zip(feat_names, sv.tolist()))
        feat_sorted = sorted(feat_shap, key=lambda x: abs(to_scalar(x[1])), reverse=True)
        
        # IMPROVEMENT 1: Human-readable explanations
        why = [
            humanize_shap_feature(fname, val)
            for fname, val in feat_sorted[:8]
            if abs(to_scalar(val)) > 0.01  # Filter negligible features
        ]
    except Exception as e:
        why = [f"Explanation generation failed: {e}"]
        feat_sorted = []

    # Extract user skills
    user_skills = skills_engine.extract_from_row(original_row)
    
    # IMPROVEMENT 3: Structured skill gaps
    gap_analysis = skills_engine.compute_gap_for_role(user_skills, pred_role)
    skill_gaps = structure_skill_gaps(gap_analysis)
    
    # IMPROVEMENT 4: Learning recommendations
    learning_recs = generate_learning_recommendations(skill_gaps, skills_engine)
    
    # IMPROVEMENT 5: Better alternatives
    alternatives_raw = skills_engine.recommend_alternatives(user_skills, top_n=3)
    alternatives = format_alternatives(alternatives_raw, user_skills, skills_engine)
    
    # IMPROVEMENT 6: Summary
    summary = generate_summary(pred_role, pred_prob, user_skills, gap_analysis)

    # Build final output
    output = {
        "summary": summary,
        
        "prediction": {
            "role": pred_role,
            "confidence": f"{pred_prob*100:.1f}%",
            "why_this_role": why
        },
        
        "your_skills": {
            "all_skills": sorted(list(user_skills))[:15],  # Top 15 for brevity
            "matching_critical": skill_gaps["critical"]["have"],
            "matching_important": skill_gaps["important"]["have"]
        },
        
        "skill_gaps": skill_gaps,
        
        "learning_roadmap": learning_recs,
        
        "alternative_paths": alternatives
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return output

# -----------------------------------------------------------
# CLI ENTRY
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int, help="Row index in dataset")
    g.add_argument("--user_id", type=str, help="User ID to explain")
    args = parser.parse_args()
    
    run_explain(index=args.index, user_id=args.user_id)