# src/skill_features.py
"""
Generate stable binary skill features for ML training.
Uses SkillsEngine for extraction + canonical skill vocabulary.
"""

import pandas as pd
from skills_engine import SkillsEngine, ROLE_SKILLS

engine = SkillsEngine()

# ---------------------------------------------------------
# 1. Build a canonical vocabulary for binary skills
# ---------------------------------------------------------
def _get_all_skills():
    vocab = set()
    for role, data in ROLE_SKILLS.items():
        for priority in ["critical", "important", "nice_to_have"]:
            for s in data.get(priority, []):
                vocab.add(s.lower())
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

    extracted = df.apply(engine.extract_from_row, axis=1)

    for skill in CANONICAL_SKILLS:
        safe = f"skill_{skill.replace(' ', '_')}"
        df[safe] = extracted.apply(
            lambda s: 1 if skill in {x.lower() for x in s} else 0
        )

    df["total_skill_hits"] = extracted.apply(len)
    return df
