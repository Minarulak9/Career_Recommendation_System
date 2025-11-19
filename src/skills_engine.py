# src/skills_engine.py
"""
Skills engine and learning-path generator.
- Normalize and fuzzy-match user skill tokens to canonical skills.
- Expanded ROLE_SKILLS library (practical, real-world skills).
- Compute skill gaps and produce short learning paths per missing skill.

Usage:
    from skills_engine import SkillsEngine
    se = SkillsEngine()
    user_set = se.extract_skills_from_row(row)   # row = a pandas Series for a user
    gap = se.compute_role_gap(user_set, "Software Engineer")
    paths = se.make_learning_paths(gap["missing"])
"""

from pathlib import Path
import re
import json
import math

# Try to import rapidfuzz for better fuzzy matching; fallback to difflib
try:
    from rapidfuzz import process as rf_process
    from rapidfuzz import fuzz as rf_fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    import difflib
    _HAS_RAPIDFUZZ = False

# ---------- Canonical skill library (expanded) ----------
# Add or edit as you like.
ROLE_SKILLS = {
    "Software Engineer": [
        "data structures", "algorithms", "object oriented programming", "system design",
        "rest apis", "http", "version control", "git", "github", "unit testing",
        "integration testing", "debugging", "clean code", "design patterns",
        "python", "java", "c++", "c#", "sql", "mysql", "postgresql",
        "nosql", "mongodb", "linux", "bash", "docker", "kubernetes",
        "ci/cd", "aws", "gcp", "azure", "redis", "performance tuning",
        "security basics", "api design", "microservices", "oauth"
    ],

    "Frontend Developer": [
        "html", "css", "javascript", "typescript", "react", "vue", "angular",
        "responsive design", "accessibility", "web performance", "webpack",
        "state management", "redux", "mobx", "next.js", "vite", "tailwind",
        "sass", "figma", "ui design", "cross-browser"
    ],

    "Backend Developer": [
        "node.js", "express", "python", "django", "flask", "java", "spring",
        "rest apis", "graphql", "authentication", "authorization", "jwt",
        "sql", "postgresql", "mysql", "mongodb", "redis", "docker", "kubernetes",
        "microservices", "message queues", "rabbitmq", "kafka", "ci/cd"
    ],

    "AI Engineer": [
        "python", "machine learning", "deep learning", "tensorflow", "pytorch",
        "nlp", "computer vision", "data preprocessing", "feature engineering",
        "model evaluation", "hyperparameter tuning", "mlops", "sklearn",
        "pandas", "numpy", "model deployment", "docker"
    ],

    "Data Analyst": [
        "excel", "sql", "tableau", "power bi", "data visualization",
        "statistics", "pandas", "numpy", "data cleaning", "dashboarding",
        "business intelligence", "storytelling"
    ],

    "Data Scientist": [
        "python", "machine learning", "statistics", "feature engineering",
        "modeling", "sql", "pandas", "numpy", "visualization", "mlops"
    ],

    "Product Manager": [
        "product strategy", "roadmap", "user research", "communication",
        "stakeholder management", "analytics", "prioritization", "okrs",
        "user testing", "wireframing"
    ]
}

# Flatten unique set of canonical skills
_ALL_CANONICAL_SKILLS = sorted({s.lower() for skills in ROLE_SKILLS.values() for s in skills})

# ---------- Aliases (common variants) ----------
# Map messy tokens -> canonical skill
SKILL_ALIASES = {
    "js": "javascript",
    "javascript": "javascript",
    "react.js": "react",
    "reactjs": "react",
    "react native": "react",
    "nodejs": "node.js",
    "node": "node.js",
    "py": "python",
    "pyhton": "python",
    "ml": "machine learning",
    "dl": "deep learning",
    "data structure": "data structures",
    "ds": "data structures",
    "algos": "algorithms",
    "db": "databases",
    "sqlserver": "sql",
    "postgres": "postgresql",
    "aws cloud": "aws",
    "gcp cloud": "gcp",
    "k8s": "kubernetes",
    "ci cd": "ci/cd",
    "ci/cd": "ci/cd",
    "unit test": "unit testing",
    "ut": "unit testing",
    "ux": "ui design",
    "ui ux": "ui design"
}

# ---------- Learning resources (short suggestions) ----------
# For each canonical skill, small pointers (course/book/article)
LEARNING_RESOURCES = {
    "data structures": ["Grokking DS & Algorithms", "CS50 / free courses", "Practice on LeetCode (easy→hard)"],
    "algorithms": ["CLRS (book)", "Algorithms by Robert Sedgewick", "LeetCode guided paths"],
    "python": ["Official Python docs", "Automate the Boring Stuff", "Python for Everybody"],
    "java": ["Java Programming", "Effective Java (book)"],
    "git": ["Git official docs", "Pro Git book", "Practice branching & PR flow"],
    "docker": ["Docker docs", "Deploy simple app with Docker"],
    "kubernetes": ["Kubernetes basics (k8s.io)", "Play with Kubernetes tutorials"],
    "machine learning": ["Andrew Ng Coursera", "Hands-on ML with Scikit-Learn & TF"],
    "deep learning": ["Deep Learning Specialization (Coursera)", "PyTorch tutorials"],
    "sql": ["Mode SQL tutorials", "SQLBolt"],
    "react": ["Official React docs", "Build small apps with CRA or Vite"],
    "html": ["MDN HTML guide"],
    "css": ["MDN CSS guide", "Flexbox & Grid courses"],
    "rest apis": ["Designing HTTP APIs", "Postman practice"],
    "system design": ["Grokking System Design", "System design interview videos"]
}
# fallback generic resource
_GENERIC_RES = ["Online course (Coursera/edX/Pluralsight/Udemy)", "Hands-on projects", "Documentation & small projects"]


# ---------- Skills Engine Class ----------
class SkillsEngine:
    def __init__(self, canonical_skills=None, aliases=None, resources=None, fuzzy_threshold=80):
        self.canonical = canonical_skills or _ALL_CANONICAL_SKILLS
        self.aliases = aliases or SKILL_ALIASES
        self.resources = resources or LEARNING_RESOURCES
        self.threshold = fuzzy_threshold

    # normalize token: remove non-alphanum, trim, lowercase
    def _clean_token(self, tok):
        if not tok:
            return ""
        tok = str(tok).strip().lower()
        tok = re.sub(r"[^a-z0-9\+\-\.\s#]", " ", tok)
        tok = re.sub(r"\s+", " ", tok).strip()
        return tok

    def _alias_map(self, token):
        return self.aliases.get(token, token)

    def fuzzy_match(self, token):
        tok = self._clean_token(token)
        if not tok:
            return None
        # direct alias
        if tok in self.aliases:
            return self.aliases[tok]

        # exact match against canonical
        if tok in self.canonical:
            return tok

        # rapidfuzz if available
        if _HAS_RAPIDFUZZ:
            choice, score, _ = rf_process.extractOne(tok, self.canonical, scorer=rf_fuzz.token_sort_ratio)
            if score >= self.threshold:
                return choice
            # also try partial ratio
            choice2, score2, _ = rf_process.extractOne(tok, self.canonical, scorer=rf_fuzz.partial_ratio)
            if score2 >= self.threshold:
                return choice2
            return None
        else:
            # fallback difflib
            matches = difflib.get_close_matches(tok, self.canonical, n=1, cutoff=self.threshold/100.0)
            return matches[0] if matches else None

    def normalize_and_map(self, raw_token):
        tok = self._clean_token(raw_token)
        if not tok:
            return None
        tok = self._alias_map(tok)
        mapped = self.fuzzy_match(tok)
        return mapped or tok  # return mapped canonical skill or cleaned token

    def extract_skills_from_text(self, text):
        """
        Extract tokens from free text (commas, slashes, 'and', spaces),
        map to canonical skills where possible.
        """
        if text is None:
            return set()
        if isinstance(text, (list, set)):
            tokens = list(text)
        else:
            s = str(text)
            for sep in [",", ";", "|", "/", "\n", " and "]:
                s = s.replace(sep, ",")
            tokens = [t.strip() for t in s.split(",") if t.strip()]

        out = set()
        for t in tokens:
            mapped = self.normalize_and_map(t)
            if mapped:
                out.add(mapped.lower())
        return out

    def extract_from_row(self, row):
        """
        row: pandas Series for a user
        It combines Technical Skills, Soft Skills, Course Keywords, Project Keywords, Preferred Roles, Work Keywords, Graduate Major, PG Major, Highest Education
        """
        fields = [
            "Technical Skills", "Soft Skills", "Course Keywords", "Project Keywords",
            "Preferred Roles", "Work Keywords", "Skill Embedding", "Graduate Major",
            "PG Major", "Highest Education"
        ]
        s = set()
        for f in fields:
            if f in row:
                s |= self.extract_skills_from_text(row.get(f, ""))

        # also split role-like fields into tokens (e.g., "Software Engineer" -> "software engineer")
        # and find canonical skills in skill tokens
        return s

    def compute_gap_for_role(self, user_skills_set, role_name):
        role_reqs = ROLE_SKILLS.get(role_name, [])
        role_reqs_norm = [r.lower() for r in role_reqs]
        have = [r for r in role_reqs_norm if any((r in u) or (u in r) for u in user_skills_set)]
        missing = [r for r in role_reqs_norm if r not in have]
        return {"role": role_name, "have": sorted(have), "missing": sorted(missing)}

    def recommend_alternatives(self, user_skills_set, top_n=3):
        scores = []
        for role, reqs in ROLE_SKILLS.items():
            reqs_norm = [r.lower() for r in reqs]
            overlap = sum(1 for r in reqs_norm if any((r in u) or (u in r) for u in user_skills_set))
            scores.append((role, overlap))
        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
        return [r for r, s in scores_sorted[:top_n]]

    def make_learning_paths(self, missing_skills):
        """
        For each missing skill, return a short learning path (list of string suggestions).
        """
        out = {}
        for ms in missing_skills:
            ms_lower = ms.lower()
            if ms_lower in self.resources:
                out[ms_lower] = self.resources[ms_lower]
            else:
                # heuristic mapping: find related resource by keyword
                found = None
                for k in self.resources.keys():
                    if k in ms_lower or ms_lower in k:
                        found = self.resources[k]
                        break
                out[ms_lower] = found or _GENERIC_RES
        return out


# convenience instance
_DEFAULT_ENGINE = SkillsEngine()

# exported helpers
def normalize_token(token):
    return _DEFAULT_ENGINE.normalize_and_map(token)

def extract_skills_from_row(row):
    return _DEFAULT_ENGINE.extract_from_row(row)

def compute_role_gap(user_skills_set, role):
    return _DEFAULT_ENGINE.compute_gap_for_role(user_skills_set, role)

def suggest_alternatives(user_skills_set, top_n=3):
    return _DEFAULT_ENGINE.recommend_alternatives(user_skills_set, top_n=top_n)

def make_learning_paths(missing_skills):
    return _DEFAULT_ENGINE.make_learning_paths(missing_skills)
