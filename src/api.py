"""
api.py — FIXED Prediction vs Skill Match Issue

The issue: Model predicts "AI Engineer" with 15.5% confidence when user has Frontend skills
Root cause: Disconnection between model prediction and skill-based role matching

This version:
- Uses skill-based role matching when model confidence is low
- Provides more reliable career recommendations
- Maintains consistency between predicted role and skill gap analysis
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
app = FastAPI(title="Career Prediction API", version="2.1")

# ============================================================
# CORS - FULLY ENABLED
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("=" * 60)
print("CORS ENABLED: All origins allowed")
print("=" * 60)

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
            print(f"✅ Loaded model: {p}")
            break

    if model is None:
        raise FileNotFoundError("No model file found!")

    le_path = MODEL_DIR / "label_encoder.joblib"
    if le_path.exists():
        label_encoder = joblib.load(le_path)
        print("✅ Loaded label encoder")
    else:
        raise FileNotFoundError("Label encoder not found!")

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
    """Convert UserProfile to DataFrame matching training format EXACTLY"""
    
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
# ENHANCED EXPLANATION WITH SKILL-BASED FALLBACK
# ============================================================
def build_smart_explanation(profile: UserProfile, pred_role: str, pred_prob: float, all_probabilities: dict):
    """
    Build explanation with intelligent fallback:
    - If model confidence is low (<30%) AND skill match is poor (<20%),
      use the best skill-matched role instead
    """
    
    print(f"\n📊 Building explanation for model prediction: {pred_role} ({pred_prob*100:.1f}%)")

    # Extract user skills
    detected_tech = engine.extract_from_text(profile.technical_skills)
    detected_soft = engine.extract_from_text(profile.soft_skills)
    detected = detected_tech | detected_soft

    print(f"✅ Detected {len(detected)} skills: {sorted(list(detected))[:10]}...")

    # Seniority
    seniority = engine.seniority_estimate(detected)

    # CRITICAL FIX: Calculate skill match scores for ALL roles
    role_skill_matches = {}
    for role in ROLE_SKILLS.keys():
        score = engine.compute_role_match(detected, role)
        role_skill_matches[role] = score
    
    # Sort by skill match score
    sorted_skill_matches = sorted(role_skill_matches.items(), key=lambda x: x[1], reverse=True)
    
    print("\n🔍 Skill Match Scores:")
    for role, score in sorted_skill_matches:
        print(f"   {role}: {score}%")
    
    # Get model's confidence for predicted role
    model_confidence = pred_prob * 100  # Convert to percentage
    skill_match_for_predicted = role_skill_matches[pred_role]
    
    print(f"\n📊 Model Prediction: {pred_role}")
    print(f"   - Model Confidence: {model_confidence:.1f}%")
    print(f"   - Skill Match Score: {skill_match_for_predicted}%")
    
    # Decision logic: Use skill-based recommendation if model is unreliable
    USE_SKILL_MATCH_THRESHOLD = 30  # If model confidence < 30%
    POOR_SKILL_MATCH_THRESHOLD = 20  # If skill match < 20%
    
    final_role = pred_role
    final_confidence = pred_prob
    final_match_score = skill_match_for_predicted
    recommendation_method = "model"
    
    if model_confidence < USE_SKILL_MATCH_THRESHOLD and skill_match_for_predicted < POOR_SKILL_MATCH_THRESHOLD:
        # Use best skill-matched role instead
        best_skill_role = sorted_skill_matches[0][0]
        best_skill_score = sorted_skill_matches[0][1]
        
        print(f"\n⚠️ Low model confidence ({model_confidence:.1f}%) with poor skill match ({skill_match_for_predicted}%)")
        print(f"✅ Using best skill-matched role: {best_skill_role} (skill match: {best_skill_score}%)")
        
        final_role = best_skill_role
        final_match_score = best_skill_score
        # Use skill match as confidence when using skill-based recommendation
        final_confidence = best_skill_score / 100.0
        recommendation_method = "skill_match"
    
    print(f"\n🎯 Final Recommendation: {final_role}")
    print(f"   - Method: {recommendation_method}")
    print(f"   - Confidence: {final_confidence*100:.1f}%")
    print(f"   - Skill Match: {final_match_score}%")

    # Gap analysis for FINAL role
    gaps = engine.compute_gap(detected, final_role)
    total_missing = (
        len(gaps["critical"]["missing"]) +
        len(gaps["important"]["missing"])
    )

    print(f"✅ Gap analysis complete: {total_missing} missing skills")

    # Learning resources
    missing_skills = (
        gaps["critical"]["missing"] +
        gaps["important"]["missing"]
    )
    learning_roadmap = engine.learning_path(missing_skills)

    # Project recommendation
    flagship_project = engine.recommend_project(final_role)

    # Alternatives (exclude final role)
    alternatives = engine.alternatives(detected, exclude=final_role)

    # Effort estimation
    effort_required = engine.estimate_effort(total_missing)

    # Formal paragraph
    confidence_text = "high" if final_confidence > 0.7 else "moderate" if final_confidence > 0.4 else "preliminary"
    
    paragraph = (
        f"Based on a formal evaluation of your technical profile, skill indicators, "
        f"and experience attributes, the recommended role is '{final_role}' with a "
        f"{confidence_text} confidence level of {final_confidence * 100:.1f}% "
        f"(skill match score: {final_match_score}%). "
        f"The assessment identifies notable strengths in several foundational areas; however, development is "
        f"recommended in crucial skills such as "
        f"{', '.join(gaps['critical']['missing'][:2]) if gaps['critical']['missing'] else 'core domain fundamentals'}. "
        f"Your current competency level is classified as '{seniority}', and the "
        f"proposed learning roadmap provides a structured path to strengthen readiness "
        f"for this career direction."
    )

    # Simple prediction reasons (based on profile characteristics)
    prediction_reasons = []
    
    # Add recommendation method note
    if recommendation_method == "skill_match":
        prediction_reasons.append(f"Recommendation based on skill profile analysis (your skills align {final_match_score}% with {final_role})")
    else:
        prediction_reasons.append(f"AI model prediction with {final_confidence*100:.1f}% confidence")
    
    if len(detected) > 10:
        prediction_reasons.append(f"Strong skill portfolio with {len(detected)} identified competencies")
    elif len(detected) > 5:
        prediction_reasons.append(f"Moderate skill set with {len(detected)} core competencies")
    else:
        prediction_reasons.append(f"Developing skill set with {len(detected)} current competencies")
    
    if profile.experience_months > 24:
        prediction_reasons.append(f"Significant experience ({profile.experience_months} months)")
    elif profile.experience_months > 0:
        prediction_reasons.append(f"Relevant experience ({profile.experience_months} months)")
    
    if profile.project_count > 5:
        prediction_reasons.append(f"Extensive project portfolio ({profile.project_count} projects)")
    elif profile.project_count > 0:
        prediction_reasons.append(f"Practical project experience ({profile.project_count} projects)")
    
    if profile.graduate_cgpa > 8.0:
        prediction_reasons.append(f"Strong academic background (CGPA: {profile.graduate_cgpa})")
    
    if profile.courses_completed > 10:
        prediction_reasons.append(f"Continuous learning commitment ({profile.courses_completed} courses)")

    # JSON OUTPUT
    output = {
        "summary": {
            "predicted_role": final_role,
            "confidence": f"{final_confidence * 100:.1f}%",
            "match_score": f"{final_match_score}%",
            "seniority": seniority,
            "recommendation_method": recommendation_method,
            "formal_explanation": paragraph,
        },

        "prediction_reasons": prediction_reasons[:5],  # Top 5 reasons

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
        
        # Additional debug info
        "model_info": {
            "original_model_prediction": pred_role,
            "original_model_confidence": f"{pred_prob * 100:.1f}%",
            "all_skill_matches": {
                role: f"{score}%" 
                for role, score in sorted_skill_matches
            }
        }
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
        "version": "2.1 - Fixed",
        "docs": "/docs",
        "cors_enabled": True,
        "message": "Skill-based fallback enabled for low-confidence predictions"
    }


@app.post("/predict")
def predict(profile: UserProfile):
    print("\n" + "="*60)
    print("📥 PREDICT REQUEST RECEIVED")
    print("="*60)

    try:
        # Convert to DataFrame
        df = profile_to_df(profile)
        print(f"✅ Created DataFrame with shape: {df.shape}")

        # Apply SAME preprocessing as training
        print("📊 Applying feature extraction...")
        df = extract_skill_flags(df)
        print(f"   After skill flags: {df.shape}")
        
        df = ensure_full_schema(df)
        print(f"   After schema check: {df.shape}")

        # Make prediction
        print("🔮 Running prediction...")
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
        # Convert to DataFrame
        df = profile_to_df(profile)
        print(f"✅ Created DataFrame with shape: {df.shape}")

        # Apply SAME preprocessing as training
        print("📊 Applying feature extraction...")
        df = extract_skill_flags(df)
        print(f"   After skill flags: {df.shape}")
        
        df = ensure_full_schema(df)
        print(f"   After schema check: {df.shape}")

        # Make prediction
        print("🔮 Running prediction...")
        proba = model.predict_proba(df)[0]

        idx = int(np.argmax(proba))
        role = label_encoder.inverse_transform([idx])[0]
        score = float(proba[idx])
        
        # Get all probabilities
        all_probs = {
            label_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(proba)
        }

        print(f"✅ Model Prediction: {role} ({score*100:.1f}%)")

        # Build SMART explanation with skill-based fallback
        result = build_smart_explanation(profile, role, score, all_probs)
        
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
        "cors_enabled": True,
        "version": "2.1-fixed"
    }


# ============================================================
# DEBUG ENDPOINT
# ============================================================
@app.post("/debug/shape")
def debug_shape(profile: UserProfile):
    """Debug endpoint to check shape at each preprocessing step"""
    try:
        df = profile_to_df(profile)
        shapes = {
            "initial": df.shape,
            "columns_initial": list(df.columns)
        }
        
        df = extract_skill_flags(df)
        shapes["after_skill_flags"] = df.shape
        shapes["columns_after_flags"] = list(df.columns)
        
        df = ensure_full_schema(df)
        shapes["after_schema"] = df.shape
        shapes["columns_final"] = list(df.columns)
        shapes["total_columns"] = len(df.columns)
        
        return shapes
    except Exception as e:
        return {"error": str(e)}


@app.post("/debug/skill-analysis")
def debug_skill_analysis(profile: UserProfile):
    """Debug endpoint to analyze skill extraction and matching"""
    try:
        detected_tech = engine.extract_from_text(profile.technical_skills)
        detected_soft = engine.extract_from_text(profile.soft_skills)
        detected = detected_tech | detected_soft
        
        role_matches = {}
        for role in ROLE_SKILLS.keys():
            score = engine.compute_role_match(detected, role)
            role_matches[role] = score
        
        sorted_matches = sorted(role_matches.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "detected_skills": sorted(list(detected)),
            "skill_count": len(detected),
            "seniority": engine.seniority_estimate(detected),
            "role_skill_matches": dict(sorted_matches),
            "best_match": sorted_matches[0][0] if sorted_matches else None,
            "best_match_score": sorted_matches[0][1] if sorted_matches else 0
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Career AI API Server (FIXED VERSION)...")
    print("📍 CORS: Enabled for ALL origins")
    print("📍 Docs: http://localhost:8000/docs\n")
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)