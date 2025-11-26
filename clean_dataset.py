import pandas as pd
import re

# --------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------
df = pd.read_csv("data/upscaled_data.csv")

# --------------------------------------------------------------------
# 2. Normalize technical skills
# --------------------------------------------------------------------
def clean_skills(txt):
    if pd.isna(txt): 
        return []
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9+ ]+", " ", txt)
    tokens = txt.split()
    return tokens

df["skill_tokens"] = df["Technical Skills"].apply(clean_skills)

# --------------------------------------------------------------------
# 3. Skill clusters
# --------------------------------------------------------------------
SKILL_MAP = {
    "AI Engineer": {"tensorflow","pytorch","keras","sklearn","ml","deep","nlp","vision","pandas","numpy"},
    "Data Analyst": {"sql","excel","tableau","powerbi","analytics","pandas","numpy"},
    "Backend Developer": {"node","django","spring","php","express","laravel","mysql","postgres","mongodb","api"},
    "Frontend Developer": {"react","javascript","html","css","bootstrap","angular","vue"},
    "UX Designer": {"figma","xd","photoshop","illustrator","ui","prototyping","wireframe"},
    "Project Manager": {"jira","scrum","kanban","planning","management","leadership"},
}

# --------------------------------------------------------------------
# 4. Decide final role with controlled auto-correction
# --------------------------------------------------------------------
def decide_role(row):
    tokens = set(row["skill_tokens"])
    true_label = row["Target Job Role"]

    # compute match scores
    scores = {role: len(tokens & skills) for role, skills in SKILL_MAP.items()}
    best_role = max(scores, key=scores.get)
    best_score = scores[best_role]

    # Remove completely irrelevant rows
    if best_score == 0:
        return None

    # If matches original → keep
    if best_role == true_label:
        return true_label

    # Strong confidence threshold → auto-correct
    if best_score >= 3:
        return best_role

    # Borderline case → keep original target
    return true_label


df["FinalRole"] = df.apply(decide_role, axis=1)

# --------------------------------------------------------------------
# 5. Keep rows with valid roles
# --------------------------------------------------------------------
clean_df = df[df["FinalRole"].notna()].copy()
clean_df["Target Job Role"] = clean_df["FinalRole"]

# remove helper columns
clean_df = clean_df.drop(columns=["FinalRole", "skill_tokens"], errors="ignore")

# --------------------------------------------------------------------
# 6. Compute corrected label count SAFELY
# --------------------------------------------------------------------
# Restrict original df to same indexes as clean_df
original_aligned = df.loc[clean_df.index, "Target Job Role"]
new_aligned = clean_df["Target Job Role"]

corrected_count = (original_aligned != new_aligned).sum()
removed_count = len(df) - len(clean_df)

# --------------------------------------------------------------------
# 7. Save cleaned dataset
# --------------------------------------------------------------------
clean_df.to_csv("data/cleaned_dataset_v2.csv", index=False)

print("👉 CLEANING COMPLETE")
print("--------------------------------")
print("Original rows:", len(df))
print("Cleaned rows:", len(clean_df))
print("Corrected labels:", corrected_count)
print("Removed rows:", removed_count)
print("--------------------------------")
print("Saved as: cleaned_dataset_v2.csv")
