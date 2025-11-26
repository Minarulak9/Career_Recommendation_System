# src/skill_features.py
"""
Generate stable binary skill features for ML training.
Uses SkillsEngine for extraction + expanded canonical skill vocabulary.
"""

import pandas as pd
from skills_engine import SkillsEngine, ROLE_SKILLS

engine = SkillsEngine()

# ---------------------------------------------------------
# 1. Build a canonical vocabulary for binary features
# ---------------------------------------------------------
def _get_all_skills():
    vocab = set()
    # include ROLE_SKILLS canonical tokens
    for role, data in ROLE_SKILLS.items():
        for priority in ["critical", "important", "nice_to_have"]:
            for s in data.get(priority, []):
                vocab.add(s.lower())

    # Add common aliases / extra tokens not present in ROLE_SKILLS
    extra = {
        # backend
        "express", "expressjs", "fastapi", "jwt", "redis", "rest", "restapi", "rest apis",
        "spring", "springboot", "laravel", "php", "mysql", "postgres", "postgresql",
        "mongodb", "nosql", "api", "microservices", "docker", "kubernetes",
        # frontend / web
        "webpack", "tailwind", "bootstrap", "typescript", "vue", "next.js", "next",
        # design
        "ui", "ux", "ux research", "user research", "wireframe", "mockup", "prototyping",
        "typography", "persona", "usability",
        # data / ml
        "scikit-learn", "sklearn", "pytorch", "keras", "tensorflow", "nlp", "computer vision",
        "cv", "deep learning", "deeplearning", "machine learning", "ml",
        # other useful tokens
        "git", "ci/cd", "restful", "restful api", "http", "oauth", "jwt", "graphql"
    }

    for e in extra:
        vocab.add(e.lower())

    return sorted(vocab)

CANONICAL_SKILLS = _get_all_skills()


# ---------------------------------------------------------
# 2. Extract final skill flags
# ---------------------------------------------------------
def extract_skill_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - skill_python
    - skill_react
    - skill_sql
    - ...
    - total_skill_hits
    """

    df = df.copy()

    # Use SkillsEngine to extract cleaned canonical skills per row
    extracted = df.apply(engine.extract_from_row, axis=1)

    # Create binary columns for canonical skills
    for skill in CANONICAL_SKILLS:
        safe_col = f"skill_{skill.replace(' ', '_').replace('/', '_').replace('.', '').replace('-', '_')}"
        df[safe_col] = extracted.apply(
            lambda s: 1 if any(skill == x.lower() or skill in x.lower() or x.lower() in skill for x in s) else 0
        )

    # Add a simple skill count
    df["total_skill_hits"] = extracted.apply(len)

    return df
