# explain.py
"""
Production-ready explanation engine with:
1. Clean, human-readable explanations
2. No garbage data
3. Structured skill gaps with priorities
4. Learning roadmap with duration
5. Better alternatives (excludes predicted role)
6. Actionable summary
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

from skills_engine import (
    SkillsEngine,
    ROLE_SKILLS,
    extract_skills_from_row,
    compute_role_gap,
    suggest_alternatives,
    make_learning_paths,
    estimate_effort,
    calculate_match_score
)

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data.csv"
MODEL_DIR = ROOT / "models"

MODEL_PATHS = [
    MODEL_DIR / "tuned_model.joblib",
    MODEL_DIR / "final_model.joblib",
]

LABEL_ENCODER_PATHS = [
    MODEL_DIR / "label_encoder.joblib",
]

TARGET_COL = "Target Job Role"


# ============================================================
# LOADERS
# ============================================================
def load_model():
    for p in MODEL_PATHS:
        if p.exists():
            print(f"Loaded model from: {p}")
            return joblib.load(p)
    raise FileNotFoundError("No model found. Run train.py first.")

def load_label_encoder():
    for p in LABEL_ENCODER_PATHS:
        if p.exists():
            print(f"Loaded label encoder from: {p}")
            return joblib.load(p)
    return None

def load_data():
    return pd.read_csv(DATA_PATH)


# ============================================================
# FEATURE NAME EXTRACTION
# ============================================================
def get_feature_names(preprocessor):
    """Extract feature names from fitted preprocessor."""
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


# ============================================================
# SHAP EXPLAINER
# ============================================================
def build_shap_explainer(clf, preprocessor, Xraw, bg_size=50):
    """Build SHAP KernelExplainer."""
    bg_raw = Xraw.sample(n=min(bg_size, len(Xraw)), random_state=42)
    bg_trans = preprocessor.transform(bg_raw)

    if sparse.issparse(bg_trans):
        bg_trans = bg_trans.toarray()
    bg_trans = np.asarray(bg_trans)

    feature_names = get_feature_names(preprocessor)
    
    # Align length
    if len(feature_names) != bg_trans.shape[1]:
        diff = bg_trans.shape[1] - len(feature_names)
        if diff > 0:
            feature_names += [f"feature_{i}" for i in range(diff)]
        else:
            feature_names = feature_names[:bg_trans.shape[1]]

    def predict_fn(x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return clf.predict_proba(x)

    explainer = shap.KernelExplainer(predict_fn, bg_trans, link="logit")
    return explainer, feature_names


def to_scalar(v):
    """Safely convert to scalar."""
    try:
        return float(np.array(v).astype(float).ravel()[0])
    except:
        return 0.0


# ============================================================
# HUMAN-READABLE EXPLANATIONS
# ============================================================
def generate_why_explanations(feat_sorted, pred_role, top_n=6):
    """Generate clean, meaningful explanations."""
    explanations = []
    
    for fname, val in feat_sorted[:top_n * 2]:  # Check more, filter later
        score = to_scalar(val)
        if abs(score) < 0.01:
            continue
        
        is_positive = score > 0
        impact = "supports" if is_positive else "weakly affects"
        
        # Parse feature name
        if "Preferred Roles_" in fname:
            role = fname.replace("Preferred Roles_", "")
            if role.lower() == pred_role.lower() and is_positive:
                explanations.append(f"Your preference for '{role}' strongly matches this prediction")
            elif role.lower() != pred_role.lower():
                continue  # Skip non-matching roles
                
        elif "Technical Skills_" in fname:
            skill = fname.split("_")[-1]
            if is_positive:
                explanations.append(f"Your '{skill}' skill is valuable for this role")
            
        elif "Job Level_" in fname:
            level = fname.replace("Job Level_", "")
            explanations.append(f"Your experience level ({level}) {impact} this role")
            
        elif "Graduate Major_" in fname or "PG Major_" in fname:
            major = fname.split("_")[-1]
            if is_positive:
                explanations.append(f"Your educational background ({major}) aligns with this role")
                
        elif "Interest" in fname:
            interest = fname.replace("Interest ", "").replace("_", " ")
            if is_positive:
                explanations.append(f"Your interest in {interest} supports this career path")
                
        elif "Experience Months" in fname:
            if is_positive:
                explanations.append("Your work experience duration is relevant")
                
        elif "Project Count" in fname:
            if is_positive:
                explanations.append("Your project experience strengthens this prediction")
        
        if len(explanations) >= top_n:
            break
    
    # Fallback if no explanations generated
    if not explanations:
        explanations = [
            f"Your profile characteristics align with {pred_role}",
            "Based on your skills and experience combination"
        ]
    
    return explanations


# ============================================================
# OUTPUT FORMATTERS
# ============================================================
def format_skill_gaps(gap_analysis):
    """Format skill gaps cleanly."""
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


def format_learning_roadmap(skill_gaps, skills_engine):
    """Generate prioritized learning roadmap."""
    critical_missing = skill_gaps["critical"]["missing"]
    important_missing = skill_gaps["important"]["missing"]
    
    # Prioritize: critical first, then important
    priority_skills = critical_missing[:3] + important_missing[:2]
    
    paths = skills_engine.make_learning_paths(priority_skills, top_n=5)
    
    roadmap = []
    for skill, info in paths.items():
        priority = "Critical" if skill in critical_missing else "Important"
        roadmap.append({
            "skill": skill,
            "priority": priority,
            "duration": info["duration"],
            "difficulty": info["difficulty"],
            "resources": info["resources"]
        })
    
    return roadmap


def format_alternatives(alternatives_raw, user_skills, skills_engine):
    """Format alternative career paths."""
    formatted = []
    
    for role, matched, total in alternatives_raw:
        gap = skills_engine.compute_gap_for_role(user_skills, role)
        
        critical_missing = gap["critical"]["missing"]
        important_missing = gap["important"]["missing"]
        all_missing = critical_missing + important_missing
        
        critical_have = gap["critical"]["have"]
        important_have = gap["important"]["have"]
        all_have = critical_have + important_have
        
        match_score = skills_engine.calculate_match_score(user_skills, role)
        effort = skills_engine.estimate_effort(len(all_missing))
        
        formatted.append({
            "role": role,
            "match_score": f"{match_score}%",
            "skills_you_have": all_have[:5],
            "skills_to_learn": all_missing[:5],
            "effort_required": effort
        })
    
    return formatted


def generate_summary(pred_role, pred_prob, skill_gaps, user_skills):
    """Generate actionable summary."""
    
    # Confidence level
    if pred_prob >= 0.85:
        confidence = "High"
    elif pred_prob >= 0.65:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # Strengths
    strengths = (
        skill_gaps["critical"]["have"][:2] +
        skill_gaps["important"]["have"][:2] +
        skill_gaps["nice_to_have"]["have"][:1]
    )
    
    if not strengths:
        # Fallback to any detected skills
        strengths = list(user_skills)[:3] if user_skills else ["Adaptable learner"]
    
    # Top priority
    critical_missing = skill_gaps["critical"]["missing"]
    if critical_missing:
        top_priority = f"Focus on: {', '.join(critical_missing[:2])}"
    elif skill_gaps["important"]["missing"]:
        top_priority = f"Improve: {', '.join(skill_gaps['important']['missing'][:2])}"
    else:
        top_priority = "You're well-prepared! Start applying."
    
    # Readiness assessment
    critical_have = len(skill_gaps["critical"]["have"])
    critical_total = critical_have + len(skill_gaps["critical"]["missing"])
    
    if critical_total == 0:
        readiness = "Good foundation - continue building skills"
    else:
        readiness_pct = (critical_have / critical_total) * 100
        if readiness_pct >= 80:
            readiness = "Strong match - Start applying now!"
        elif readiness_pct >= 60:
            readiness = "Good match - Focus on 2-3 key skills"
        elif readiness_pct >= 40:
            readiness = "Developing - Need 3-6 months of learning"
        else:
            readiness = "Early stage - Build foundational skills first"
    
    return {
        "predicted_role": pred_role,
        "confidence": f"{confidence} ({pred_prob*100:.1f}%)",
        "your_strengths": strengths[:5],
        "top_priority": top_priority,
        "readiness": readiness
    }


# ============================================================
# MAIN EXPLAIN FUNCTION
# ============================================================
def run_explain(index=None, user_id=None):
    """Generate comprehensive career explanation."""
    
    # Load resources
    df = load_data()
    model = load_model()
    le = load_label_encoder()
    skills_engine = SkillsEngine()

    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["clf"]

    Xraw = df.drop(columns=[TARGET_COL], errors="ignore")

    # Select user row
    if index is not None:
        row = Xraw.iloc[[index]].reset_index(drop=True)
        original_row = df.iloc[index]
    else:
        sel = df[df["User ID"] == user_id]
        if sel.empty:
            raise ValueError(f"User ID '{user_id}' not found")
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

    pred_role = le.inverse_transform([pred_idx])[0] if le else clf.classes_[pred_idx]

    # SHAP explanations
    explainer, feat_names = build_shap_explainer(clf, pre, Xraw)
    
    try:
        shap_vals = explainer.shap_values(x_t, nsamples=200)
        if isinstance(shap_vals, list):
            sv = np.array(shap_vals[pred_idx])[0]
        else:
            sv = np.array(shap_vals)[0]
        
        feat_shap = list(zip(feat_names, sv.tolist()))
        feat_sorted = sorted(feat_shap, key=lambda x: abs(to_scalar(x[1])), reverse=True)
        
        why_explanations = generate_why_explanations(feat_sorted, pred_role)
    except Exception as e:
        why_explanations = [f"Profile analysis complete for {pred_role}"]

    # Extract user skills (clean)
    user_skills = skills_engine.extract_from_row(original_row)
    
    # Skill gap analysis
    gap_analysis = skills_engine.compute_gap_for_role(user_skills, pred_role)
    skill_gaps = format_skill_gaps(gap_analysis)
    
    # Learning roadmap
    learning_roadmap = format_learning_roadmap(skill_gaps, skills_engine)
    
    # Alternative paths (excluding predicted role)
    alternatives_raw = skills_engine.recommend_alternatives(
        user_skills, 
        exclude_role=pred_role, 
        top_n=3
    )
    alternative_paths = format_alternatives(alternatives_raw, user_skills, skills_engine)
    
    # Summary
    summary = generate_summary(pred_role, pred_prob, skill_gaps, user_skills)

    # Build output
    output = {
        "summary": summary,
        
        "prediction": {
            "role": pred_role,
            "confidence": f"{pred_prob*100:.1f}%",
            "why": why_explanations
        },
        
        "your_profile": {
            "detected_skills": sorted(list(user_skills)) if user_skills else ["No specific skills detected"],
            "matching_skills": {
                "critical": skill_gaps["critical"]["have"],
                "important": skill_gaps["important"]["have"]
            }
        },
        
        "skill_gaps": {
            "critical_missing": skill_gaps["critical"]["missing"],
            "important_missing": skill_gaps["important"]["missing"],
            "nice_to_have_missing": skill_gaps["nice_to_have"]["missing"]
        },
        
        "learning_roadmap": learning_roadmap,
        
        "alternative_careers": alternative_paths
    }

    # Print result
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return output


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain career prediction")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int, help="Row index in dataset")
    group.add_argument("--user_id", type=str, help="User ID to explain")
    
    args = parser.parse_args()
    run_explain(index=args.index, user_id=args.user_id)