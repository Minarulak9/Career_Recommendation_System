# src/api.py
"""
FastAPI backend for Career Prediction Model.
Endpoints:
- GET  /                → Health check
- GET  /health          → API status
- POST /predict         → Predict career from user data
- POST /explain         → Full explanation with skills gap
- GET  /roles           → List all possible roles
- GET  /skills/{role}   → Get required skills for a role
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import joblib
import pandas as pd
import numpy as np
from scipy import sparse

from skills_engine import (
    SkillsEngine,
    ROLE_SKILLS,
    compute_role_gap,
    suggest_alternatives,
    make_learning_paths,
    estimate_effort
)

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(
    title="Career Prediction API",
    description="AI-powered career recommendation system",
    version="1.0.0"
)

# CORS (allow frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# LOAD MODEL
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

model = None
label_encoder = None
skills_engine = SkillsEngine()

def load_models():
    """Load ML models on startup."""
    global model, label_encoder
    
    model_paths = [
        MODEL_DIR / "tuned_model.joblib",
        MODEL_DIR / "final_model.joblib",
    ]
    
    for p in model_paths:
        if p.exists():
            model = joblib.load(p)
            print(f"✅ Loaded model: {p}")
            break
    
    le_path = MODEL_DIR / "label_encoder.joblib"
    if le_path.exists():
        label_encoder = joblib.load(le_path)
        print(f"✅ Loaded label encoder: {le_path}")

# Load on startup
load_models()

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================
class UserProfile(BaseModel):
    """User profile for prediction."""
    age: int = Field(..., ge=18, le=60, example=25)
    gender: str = Field(..., example="Male")
    location: str = Field(..., example="Bangalore")
    languages_spoken: str = Field(..., example="English, Hindi")
    
    class_10_percentage: float = Field(..., ge=0, le=100, example=85.5)
    class_12_percentage: float = Field(..., ge=0, le=100, example=78.0)
    class_12_stream: str = Field(..., example="Science")
    
    graduate_major: str = Field(..., example="BTech CSE")
    graduate_cgpa: float = Field(..., ge=0, le=10, example=8.5)
    pg_major: Optional[str] = Field(default="None", example="None")
    pg_cgpa: Optional[float] = Field(default=0.0, ge=0, le=10)
    highest_education: str = Field(..., example="BTech")
    
    technical_skills: str = Field(..., example="Python, JavaScript, React")
    soft_skills: str = Field(..., example="Communication, Teamwork")
    
    courses_completed: int = Field(..., ge=0, example=15)
    avg_course_difficulty: float = Field(..., ge=1, le=5, example=3.5)
    total_hours_learning: int = Field(..., ge=0, example=200)
    
    project_count: int = Field(..., ge=0, example=5)
    avg_project_complexity: float = Field(..., ge=1, le=5, example=3.0)
    
    experience_months: int = Field(..., ge=0, example=12)
    experience_types: str = Field(..., example="Internship")
    job_level: str = Field(..., example="Entry Level")
    
    interest_stem: float = Field(..., ge=0, le=1, example=0.8)
    interest_business: float = Field(..., ge=0, le=1, example=0.3)
    interest_arts: float = Field(..., ge=0, le=1, example=0.2)
    interest_design: float = Field(..., ge=0, le=1, example=0.4)
    interest_medical: float = Field(..., ge=0, le=1, example=0.1)
    interest_social_science: float = Field(..., ge=0, le=1, example=0.2)
    
    career_preference: str = Field(..., example="Technical")
    work_preference: str = Field(..., example="Hybrid")
    preferred_industries: str = Field(..., example="Technology, Finance")
    preferred_roles: str = Field(..., example="Software Engineer")
    
    conscientiousness: int = Field(..., ge=1, le=5, example=4)
    extraversion: int = Field(..., ge=1, le=5, example=3)
    openness: int = Field(..., ge=1, le=5, example=4)
    agreeableness: int = Field(..., ge=1, le=5, example=4)
    emotional_stability: int = Field(..., ge=1, le=5, example=3)
    
    current_status: str = Field(..., example="Student")
    
    # Optional fields with defaults
    academic_consistency: Optional[float] = Field(default=0.7)
    tech_skill_proficiency: Optional[float] = Field(default=0.7)
    soft_skill_proficiency: Optional[float] = Field(default=0.7)

class PredictionResponse(BaseModel):
    """Prediction response."""
    predicted_role: str
    confidence: str
    confidence_score: float
    all_probabilities: Dict[str, float]

class ExplanationResponse(BaseModel):
    """Full explanation response."""
    summary: Dict[str, Any]
    prediction: Dict[str, Any]
    your_profile: Dict[str, Any]
    skill_gaps: Dict[str, Any]
    learning_roadmap: List[Dict[str, Any]]
    alternative_careers: List[Dict[str, Any]]

class RoleSkillsResponse(BaseModel):
    """Skills required for a role."""
    role: str
    critical: List[str]
    important: List[str]
    nice_to_have: List[str]

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def profile_to_dataframe(profile: UserProfile) -> pd.DataFrame:
    """Convert UserProfile to DataFrame for prediction."""
    data = {
        "Age": profile.age,
        "Gender": profile.gender,
        "Location": profile.location,
        "Languages Spoken": profile.languages_spoken,
        "Class 10 Percentage": profile.class_10_percentage,
        "Class 12 Percentage": profile.class_12_percentage,
        "Class 12 Stream": profile.class_12_stream,
        "Graduate Major": profile.graduate_major,
        "Graduate CGPA": profile.graduate_cgpa,
        "PG Major": profile.pg_major or "None",
        "PG CGPA": profile.pg_cgpa or 0.0,
        "Highest Education": profile.highest_education,
        "Academic Consistency": profile.academic_consistency or 0.7,
        "Technical Skills": profile.technical_skills,
        "Tech Skill Proficiency": profile.tech_skill_proficiency or 0.7,
        "Soft Skills": profile.soft_skills,
        "Soft Skill Proficiency": profile.soft_skill_proficiency or 0.7,
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
    return pd.DataFrame([data])

def generate_explanation(profile: UserProfile, pred_role: str, pred_prob: float) -> dict:
    """Generate full explanation for prediction."""
    
    # Extract skills from profile
    user_skills = skills_engine.extract_skills_from_text(profile.technical_skills)
    user_skills |= skills_engine.extract_skills_from_text(profile.soft_skills)
    
    # Skill gap analysis
    gap = skills_engine.compute_gap_for_role(user_skills, pred_role)
    
    # Learning roadmap
    critical_missing = gap["critical"]["missing"]
    important_missing = gap["important"]["missing"]
    priority_skills = critical_missing[:3] + important_missing[:2]
    learning_paths = skills_engine.make_learning_paths(priority_skills, top_n=5)
    
    roadmap = []
    for skill, info in learning_paths.items():
        roadmap.append({
            "skill": skill,
            "priority": "Critical" if skill in critical_missing else "Important",
            "duration": info["duration"],
            "difficulty": info["difficulty"],
            "resources": info["resources"]
        })
    
    # Alternatives
    alternatives_raw = skills_engine.recommend_alternatives(user_skills, exclude_role=pred_role, top_n=3)
    alternatives = []
    for role, matched, total in alternatives_raw:
        alt_gap = skills_engine.compute_gap_for_role(user_skills, role)
        match_score = skills_engine.calculate_match_score(user_skills, role)
        all_missing = alt_gap["critical"]["missing"] + alt_gap["important"]["missing"]
        all_have = alt_gap["critical"]["have"] + alt_gap["important"]["have"]
        
        alternatives.append({
            "role": role,
            "match_score": f"{match_score}%",
            "skills_you_have": all_have[:5],
            "skills_to_learn": all_missing[:5],
            "effort_required": skills_engine.estimate_effort(len(all_missing))
        })
    
    # Summary
    confidence_level = "High" if pred_prob >= 0.85 else "Medium" if pred_prob >= 0.65 else "Low"
    strengths = gap["critical"]["have"][:2] + gap["important"]["have"][:2]
    if not strengths:
        strengths = list(user_skills)[:3] if user_skills else ["Adaptable learner"]
    
    if critical_missing:
        top_priority = f"Focus on: {', '.join(critical_missing[:2])}"
    elif important_missing:
        top_priority = f"Improve: {', '.join(important_missing[:2])}"
    else:
        top_priority = "You're well-prepared! Start applying."
    
    # Readiness
    critical_have = len(gap["critical"]["have"])
    critical_total = critical_have + len(gap["critical"]["missing"])
    if critical_total == 0:
        readiness = "Good foundation"
    else:
        pct = (critical_have / critical_total) * 100
        if pct >= 80:
            readiness = "Strong match - Start applying!"
        elif pct >= 60:
            readiness = "Good match - Focus on 2-3 skills"
        elif pct >= 40:
            readiness = "Developing - Need 3-6 months"
        else:
            readiness = "Early stage - Build foundations"
    
    return {
        "summary": {
            "predicted_role": pred_role,
            "confidence": f"{confidence_level} ({pred_prob*100:.1f}%)",
            "your_strengths": strengths[:5],
            "top_priority": top_priority,
            "readiness": readiness
        },
        "prediction": {
            "role": pred_role,
            "confidence": f"{pred_prob*100:.1f}%",
            "why": [
                f"Your skills and background align with {pred_role}",
                f"Your interest areas support this career path"
            ]
        },
        "your_profile": {
            "detected_skills": sorted(list(user_skills)) if user_skills else ["No specific skills detected"],
            "matching_skills": {
                "critical": gap["critical"]["have"],
                "important": gap["important"]["have"]
            }
        },
        "skill_gaps": {
            "critical_missing": gap["critical"]["missing"],
            "important_missing": gap["important"]["missing"],
            "nice_to_have_missing": gap["nice_to_have"]["missing"]
        },
        "learning_roadmap": roadmap,
        "alternative_careers": alternatives
    }

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/", tags=["Health"])
def root():
    """Root endpoint - health check."""
    return {
        "status": "online",
        "message": "Career Prediction API is running!",
        "docs": "/docs"
    }

@app.get("/health", tags=["Health"])
def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "encoder_loaded": label_encoder is not None,
        "available_roles": list(ROLE_SKILLS.keys())
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(profile: UserProfile):
    """
    Predict career role for a user profile.
    Returns predicted role and confidence score.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = profile_to_dataframe(profile)
        
        # Predict
        proba = model.predict_proba(df)[0]
        pred_idx = int(np.argmax(proba))
        pred_prob = float(proba[pred_idx])
        
        # Get role name
        pred_role = label_encoder.inverse_transform([pred_idx])[0]
        
        # All probabilities
        all_probs = {
            label_encoder.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(proba)
        }
        all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        
        return PredictionResponse(
            predicted_role=pred_role,
            confidence=f"{pred_prob*100:.1f}%",
            confidence_score=round(pred_prob, 4),
            all_probabilities=all_probs
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain", response_model=ExplanationResponse, tags=["Explanation"])
def explain(profile: UserProfile):
    """
    Get full career explanation with skill gaps and learning roadmap.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert and predict
        df = profile_to_dataframe(profile)
        proba = model.predict_proba(df)[0]
        pred_idx = int(np.argmax(proba))
        pred_prob = float(proba[pred_idx])
        pred_role = label_encoder.inverse_transform([pred_idx])[0]
        
        # Generate explanation
        explanation = generate_explanation(profile, pred_role, pred_prob)
        
        return ExplanationResponse(**explanation)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/roles", tags=["Info"])
def list_roles():
    """Get all available career roles."""
    return {
        "roles": list(ROLE_SKILLS.keys()),
        "count": len(ROLE_SKILLS)
    }

@app.get("/skills/{role}", response_model=RoleSkillsResponse, tags=["Info"])
def get_role_skills(role: str):
    """Get required skills for a specific role."""
    if role not in ROLE_SKILLS:
        raise HTTPException(
            status_code=404,
            detail=f"Role '{role}' not found. Available: {list(ROLE_SKILLS.keys())}"
        )
    
    role_data = ROLE_SKILLS[role]
    return RoleSkillsResponse(
        role=role,
        critical=role_data.get("critical", []),
        important=role_data.get("important", []),
        nice_to_have=role_data.get("nice_to_have", [])
    )

@app.get("/learning-path/{skill}", tags=["Info"])
def get_learning_path(skill: str):
    """Get learning resources for a specific skill."""
    paths = skills_engine.make_learning_paths([skill], top_n=1)
    
    if skill.lower() in paths:
        return {
            "skill": skill,
            **paths[skill.lower()]
        }
    else:
        return {
            "skill": skill,
            "resources": ["Search online courses", "Official documentation", "Build projects"],
            "duration": "1-2 months",
            "difficulty": "Varies"
        }

# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)