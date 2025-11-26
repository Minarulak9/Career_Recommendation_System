"""
api.py — Fully Updated (Compatible With New SkillsEngine + Final Pipeline)

Provides:
- /predict → returns predicted role
- /explain → returns detailed explanation, gaps, roadmap
- /roles, /skills, /learning-path endpoints
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

model = None
label_encoder = None
engine = SkillsEngine()


# ============================================================
# LOAD TRAINED MODELS
# ============================================================
def load_models():
    global model, label_encoder

    for p in [MODEL_DIR / "tuned_model.joblib", MODEL_DIR / "final_model.joblib"]:
        if p.exists():
            model = joblib.load(p)
            print(f"Loaded model: {p}")
            break

    le_path = MODEL_DIR / "label_encoder.joblib"
    if le_path.exists():
        label_encoder = joblib.load(le_path)
        print("Loaded label encoder")


load_models()


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
# EXPLANATION BUILDER (SKILLS + GAPS + ROADMAP)
# ============================================================
def build_explanation(profile: UserProfile, pred_role: str, pred_prob: float):

    # Extract user skills
    detected_tech = engine.extract_from_text(profile.technical_skills)
    detected_soft = engine.extract_from_text(profile.soft_skills)
    detected = detected_tech | detected_soft

    # Gap analysis
    gaps = engine.compute_gap(detected, pred_role)
    missing = gaps["critical"]["missing"] + gaps["important"]["missing"]

    # Roadmap (NEW engine)
    roadmap = engine.learning_path(missing)

    # Seniority
    seniority = engine.seniority_estimate(detected)

    # Alternatives
    alternatives = engine.alternatives(detected, exclude=pred_role)

    return {
        "summary": {
            "role": pred_role,
            "confidence": f"{pred_prob * 100:.1f}%",
            "match_score": f"{engine.compute_role_match(detected, pred_role)}%",
            "seniority": seniority,
        },

        "skills_detected": sorted(list(detected)),

        "skill_gaps": gaps,

        "learning_path": {
            "roadmap": roadmap,
            "effort_required": engine.estimate_effort(len(missing)),
            "recommended_project": engine.recommend_project(pred_role),
        },

        "alternative_roles": [
            {"role": r, "match_score": f"{m}%"} for r, m in alternatives
        ],
    }


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def root():
    return {"status": "Career Prediction API running", "docs": "/docs"}


@app.post("/predict")
def predict(profile: UserProfile):

    df = profile_to_df(profile)

    # === NEW: apply training preprocessing ===
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

    return {
        "predicted_role": role,
        "confidence": f"{score*100:.1f}%",
        "confidence_score": score,
        "all_probabilities": all_probs,
    }


@app.post("/explain")
def explain(profile: UserProfile):

    df = profile_to_df(profile)

    # === NEW: apply training preprocessing ===
    df = extract_skill_flags(df)
    df = ensure_full_schema(df)

    proba = model.predict_proba(df)[0]

    idx = int(np.argmax(proba))
    role = label_encoder.inverse_transform([idx])[0]
    score = float(proba[idx])

    return build_explanation(profile, role, score)


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


# ============================================================
# RUN SERVER (LOCAL DEV)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
