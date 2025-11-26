"""
api.py — CORS FULLY ENABLED - Works with ANY domain

CRITICAL: This allows ALL origins for development/testing.
For production, restrict origins to your specific domains.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import joblib
import pandas as pd
import numpy as np
import shap
from scipy import sparse
import warnings

warnings.filterwarnings("ignore")

# Import upgraded Skills Engine
from skills_engine import (
    SkillsEngine,
    ROLE_SKILLS,
)

# Import training preprocessing utilities
from skill_features import extract_skill_flags
from pipeline import ensure_full_schema

# ============================================================
# APP INIT
# ============================================================
app = FastAPI(title="Career Prediction API", version="2.0")

# ============================================================
# CORS - ABSOLUTELY PERMISSIVE (ALLOWS EVERYTHING)
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow ALL origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow ALL methods
    allow_headers=["*"],  # Allow ALL headers
)

print("=" * 60)
print("CORS ENABLED: All origins allowed")
print("=" * 60)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

model = None
label_encoder = None
engine = SkillsEngine()
shap_explainer = None
explainer_mode = None


# ============================================================
# LOAD TRAINED MODELS
# ============================================================
def load_models():
    global model, label_encoder

    for p in [MODEL_DIR / "tuned_model.joblib", MODEL_DIR / "final_model.joblib"]:
        if p.exists():
            model = joblib.load(p)
            print(f"✅ Loaded model: {p}")
            break

    le_path = MODEL_DIR / "label_encoder.joblib"
    if le_path.exists():
        label_encoder = joblib.load(le_path)
        print("✅ Loaded label encoder")


load_models()


# ============================================================
# FEATURE NAME EXTRACTOR (from explain.py)
# ============================================================
def get_feature_names(pre):
    """Extract feature names from ColumnTransformer preprocessor"""
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
# SHAP EXPLAINER BUILDER (from explain.py)
# ============================================================
def build_shap_explainer(clf, pre, df_sample):
    """Build SHAP explainer with TreeExplainer fallback to KernelExplainer"""
    global shap_explainer, explainer_mode
    
    if shap_explainer is not None:
        return shap_explainer, explainer_mode
    
    print("\n🔍 Building SHAP explainer...")

    try:
        shap_explainer = shap.TreeExplainer(clf)
        explainer_mode = "tree"
        print("✅ Using TreeExplainer")
        return shap_explainer, explainer_mode
    except Exception as e:
        print(f"⚠️  TreeExplainer failed → Using KernelExplainer (slow). Error: {e}")

        # Sample background data
        bg = df_sample.sample(n=min(40, len(df_sample)), random_state=42)
        bg_t = pre.transform(bg)

        if sparse.issparse(bg_t):
            bg_t = bg_t.toarray()

        def predict_fn(x):
            x = np.array(x)
            return clf.predict_proba(x)

        shap_explainer = shap.KernelExplainer(predict_fn, bg_t)
        explainer_mode = "kernel"
        return shap_explainer, explainer_mode


# ============================================================
# GENERATE SHAP REASONS
# ============================================================
def generate_shap_reasons(clf, pre, df_transformed, pred_idx, pred_role):
    """Generate prediction reasons using SHAP values"""
    
    try:
        # Load sample data for explainer initialization
        DATA_PATH = ROOT / "data" / "final_training_dataset.csv"
        if not DATA_PATH.exists():
            print("⚠️  Training data not found, using fallback explanation")
            return [f"The feature profile suggests alignment with {pred_role}."]
        
        df_full = pd.read_csv(DATA_PATH)
        df_full = extract_skill_flags(df_full)
        df_full = ensure_full_schema(df_full)
        
        # Drop target column
        if "Target Job Role" in df_full.columns:
            df_full = df_full.drop(columns=["Target Job Role"])
        
        # Build explainer
        explainer, mode = build_shap_explainer(clf, pre, df_full)
        feature_names = get_feature_names(pre)

        # Get SHAP values
        vals = explainer.shap_values(df_transformed)
        
        if isinstance(vals, list):
            sv = vals[pred_idx][0]
        else:
            sv = vals[0]

        # Rank features by importance
        ranked = sorted(
            zip(feature_names, sv.tolist()),
            key=lambda z: abs(z[1]),
            reverse=True
        )

        top_reasons = [
            f"{name} (impact {round(val, 3)})"
            for name, val in ranked[:5]
        ]
        
        print(f"✅ Generated {len(top_reasons)} SHAP reasons")
        return top_reasons

    except Exception as e:
        print(f"❌ SHAP explanation failed: {e}")
        import traceback
        traceback.print_exc()
        return [f"The feature profile suggests alignment with {pred_role}."]


# ============================================================
# INPUT MODEL
# ============================================================
class UserProfile(BaseModel):
    age: int
    gender: str
    location: str
    languages_spoken: str

    class_10_percentage: float
    class_12_percentage: float
    class_12_stream: str

    graduate_major: str
    graduate_cgpa: float
    pg_major: str
    pg_cgpa: float
    highest_education: str

    academic_consistency: float = 0.7

    technical_skills: str
    tech_skill_proficiency: float = 0.7

    soft_skills: str
    soft_skill_proficiency: float = 0.7

    courses_completed: int
    avg_course_difficulty: float
    total_hours_learning: int

    project_count: int
    avg_project_complexity: float

    experience_months: int
    experience_types: str
    job_level: str

    interest_stem: float
    interest_business: float
    interest_arts: float
    interest_design: float
    interest_medical: float
    interest_social_science: float

    career_preference: str
    work_preference: str
    preferred_industries: str
    preferred_roles: str

    conscientiousness: int
    extraversion: int
    openness: int
    agreeableness: int
    emotional_stability: int

    current_status: str


# ============================================================
# CONVERT PROFILE → DATAFRAME (matching training schema)
# ============================================================
def profile_to_df(profile: UserProfile):

    d = {
        "Age": profile.age,
        "Gender": profile.gender,
        "Location": profile.location,
        "Languages Spoken": profile.languages_spoken,

        "Class 10 Percentage": profile.class_10_percentage,
        "Class 12 Percentage": profile.class_12_percentage,
        "Class 12 Stream": profile.class_12_stream,

        "Graduate Major": profile.graduate_major,
        "Graduate CGPA": profile.graduate_cgpa,
        "PG Major": profile.pg_major,
        "PG CGPA": profile.pg_cgpa,
        "Highest Education": profile.highest_education,

        "Academic Consistency": profile.academic_consistency,

        "Technical Skills": profile.technical_skills,
        "Tech Skill Proficiency": profile.tech_skill_proficiency,

        "Soft Skills": profile.soft_skills,
        "Soft Skill Proficiency": profile.soft_skill_proficiency,

        "Courses Completed": profile.courses_completed,
        "Avg Course Difficulty": profile.avg_course_difficulty,
        "Total Hours Learning": profile.total_hours_learning,

        "Project Count": profile.project_count,
        "Avg Project Complexity": profile.avg_project_complexity,

        "Experience Months": profile.experience_months,
        "Experience Types": profile.experience_types,
        "Job Level": profile.job_level,

        "Interest STEM": profile.interest_stem,
        "Interest Business": profile.interest_business,
        "Interest Arts": profile.interest_arts,
        "Interest Design": profile.interest_design,
        "Interest Medical": profile.interest_medical,
        "Interest Social Science": profile.interest_social_science,

        "Career Preference": profile.career_preference,
        "Work Preference": profile.work_preference,
        "Preferred Industries": profile.preferred_industries,
        "Preferred Roles": profile.preferred_roles,

        "Conscientiousness": profile.conscientiousness,
        "Extraversion": profile.extraversion,
        "Openness": profile.openness,
        "Agreeableness": profile.agreeableness,
        "Emotional Stability": profile.emotional_stability,

        "Current Status": profile.current_status,
    }

    return pd.DataFrame([d])


# ============================================================
# EXPLANATION BUILDER (MATCHING explain.py FORMAT)
# ============================================================
def build_explanation(profile: UserProfile, pred_role: str, pred_prob: float, df_transformed):

    print(f"\n📊 Building explanation for: {pred_role}")

    # Extract user skills
    detected_tech = engine.extract_from_text(profile.technical_skills)
    detected_soft = engine.extract_from_text(profile.soft_skills)
    detected = detected_tech | detected_soft

    print(f"✅ Detected {len(detected)} skills")

    # Seniority
    seniority = engine.seniority_estimate(detected)

    # Gap analysis
    gaps = engine.compute_gap(detected, pred_role)
    total_missing = (
        len(gaps["critical"]["missing"]) +
        len(gaps["important"]["missing"])
    )

    print(f"✅ Gap analysis complete: {total_missing} missing skills")

    # Learning resources
    missing_skills = (
        gaps["critical"]["missing"] +
        gaps["important"]["missing"])
    learning_roadmap = engine.learning_path(missing_skills)

    # Role match score
    match_score = engine.compute_role_match(detected, pred_role)

    # Project recommendation
    flagship_project = engine.recommend_project(pred_role)

    # Alternatives
    alternatives = engine.alternatives(detected, exclude=pred_role)

    # Effort estimation
    effort_required = engine.estimate_effort(total_missing)

    # Formal paragraph
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

    # SHAP REASONS
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["clf"]
    pred_idx = int(np.argmax(model.predict_proba(df_transformed)[0]))
    
    prediction_reasons = generate_shap_reasons(clf, pre, df_transformed, pred_idx, pred_role)

    # JSON OUTPUT
    output = {
        "summary": {
            "predicted_role": pred_role,
            "confidence": f"{pred_prob * 100:.1f}%",
            "match_score": f"{match_score}%",
            "seniority": seniority,
            "formal_explanation": paragraph,
        },

        "prediction_reasons": prediction_reasons,

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

    print("✅ Explanation built successfully")
    return output


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def root():
    return {
        "status": "Career Prediction API running",
        "version": "2.0",
        "docs": "/docs",
        "cors_enabled": True,
        "message": "CORS is enabled for all origins"
    }


@app.post("/predict")
def predict(profile: UserProfile):
    print("\n" + "="*60)
    print("📥 PREDICT REQUEST RECEIVED")
    print("="*60)

    try:
        df = profile_to_df(profile)

        # Apply training preprocessing
        df = extract_skill_flags(df)
        df = ensure_full_schema(df)

        proba = model.predict_proba(df)[0]

        idx = int(np.argmax(proba))
        role = label_encoder.inverse_transform([idx])[0]
        score = float(proba[idx])

        # Return sorted probabilities
        all_probs = {
            label_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(proba)
        }
        all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))

        print(f"✅ Prediction: {role} ({score*100:.1f}%)")
        print("="*60 + "\n")

        return {
            "predicted_role": role,
            "confidence": f"{score*100:.1f}%",
            "confidence_score": score,
            "all_probabilities": all_probs,
        }
    
    except Exception as e:
        print(f"❌ PREDICT ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
def explain(profile: UserProfile):
    print("\n" + "="*60)
    print("📥 EXPLAIN REQUEST RECEIVED")
    print("="*60)

    try:
        df = profile_to_df(profile)

        # Apply training preprocessing
        df = extract_skill_flags(df)
        df = ensure_full_schema(df)

        # Transform for prediction
        pre = model.named_steps["preprocessor"]
        df_transformed = pre.transform(df)
        
        if sparse.issparse(df_transformed):
            df_transformed = df_transformed.toarray()

        proba = model.predict_proba(df_transformed)[0]

        idx = int(np.argmax(proba))
        role = label_encoder.inverse_transform([idx])[0]
        score = float(proba[idx])

        print(f"✅ Prediction: {role} ({score*100:.1f}%)")

        result = build_explanation(profile, role, score, df_transformed)
        
        print("="*60 + "\n")
        return result
    
    except Exception as e:
        print(f"❌ EXPLAIN ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/roles")
def roles():
    return {"roles": list(ROLE_SKILLS.keys())}


@app.get("/skills/{role}")
def get_role_skills(role: str):
    if role not in ROLE_SKILLS:
        raise HTTPException(404, "Invalid role")
    return ROLE_SKILLS[role]


@app.get("/learning-path/{skill}")
def learning_path(skill: str):
    roadmap = engine.learning_path([skill])
    if not roadmap:
        return {"skill": skill, "error": "No resources found"}
    return roadmap[0]


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "encoder_loaded": label_encoder is not None,
        "cors_enabled": True
    }


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Career AI API Server...")
    print("📍 CORS: Enabled for ALL origins")
    print("📍 Docs: http://localhost:8000/docs\n")
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)