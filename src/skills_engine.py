# skills_engine.py (ENHANCED VERSION)
"""
Enhanced Skills Engine with:
- Skill priority levels (critical/important/nice-to-have)
- Learning paths with duration estimates
- Better skill matching and gap analysis
"""

from pathlib import Path
import re
import json

try:
    from rapidfuzz import process as rf_process
    from rapidfuzz import fuzz as rf_fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    import difflib
    _HAS_RAPIDFUZZ = False

# ---------- EXPANDED ROLE SKILLS WITH PRIORITIES ----------
ROLE_SKILLS = {
    "Software Engineer": {
        "critical": ["python", "data structures", "algorithms", "git", "debugging"],
        "important": ["object oriented programming", "rest apis", "sql", "unit testing", "system design"],
        "nice_to_have": ["docker", "kubernetes", "ci/cd", "aws", "design patterns"]
    },
    
    "Frontend Developer": {
        "critical": ["html", "css", "javascript", "react"],
        "important": ["responsive design", "state management", "webpack", "typescript"],
        "nice_to_have": ["next.js", "tailwind", "figma", "accessibility"]
    },
    
    "Backend Developer": {
        "critical": ["python", "node.js", "sql", "rest apis"],
        "important": ["databases", "authentication", "docker", "mongodb"],
        "nice_to_have": ["graphql", "microservices", "kafka", "redis"]
    },
    
    "AI Engineer": {
        "critical": ["python", "machine learning", "statistics", "tensorflow"],
        "important": ["deep learning", "nlp", "data preprocessing", "model evaluation"],
        "nice_to_have": ["pytorch", "mlops", "computer vision", "hyperparameter tuning"]
    },
    
    "Data Analyst": {
        "critical": ["sql", "excel", "statistics", "data visualization"],
        "important": ["tableau", "power bi", "pandas", "data cleaning"],
        "nice_to_have": ["python", "dashboarding", "business intelligence"]
    },
    
    "Data Scientist": {
        "critical": ["python", "machine learning", "statistics", "sql"],
        "important": ["feature engineering", "pandas", "numpy", "modeling"],
        "nice_to_have": ["mlops", "visualization", "deep learning"]
    },
    
    "Project Manager": {
        "critical": ["communication", "product strategy", "roadmap"],
        "important": ["stakeholder management", "user research", "prioritization"],
        "nice_to_have": ["analytics", "okrs", "wireframing", "user testing"]
    },
    "Designer": {
        "critical": ["ui design", "figma", "wireframing", "user research"],
        "important": ["prototyping", "design systems", "accessibility", "visual design"],
        "nice_to_have": ["animation", "illustration", "branding", "sketch"]
    }
}

# Flatten all skills for matching
_ALL_CANONICAL_SKILLS = []
for role_data in ROLE_SKILLS.values():
    for priority in ["critical", "important", "nice_to_have"]:
        _ALL_CANONICAL_SKILLS.extend([s.lower() for s in role_data.get(priority, [])])
_ALL_CANONICAL_SKILLS = sorted(set(_ALL_CANONICAL_SKILLS))

# ---------- SKILL ALIASES ----------
SKILL_ALIASES = {
    "js": "javascript",
    "reactjs": "react",
    "react.js": "react",
    "react native": "react",
    "nodejs": "node.js",
    "node": "node.js",
    "py": "python",
    "ml": "machine learning",
    "dl": "deep learning",
    "ds": "data structures",
    "algos": "algorithms",
    "db": "databases",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "ci/cd": "ci cd",
    "ui/ux": "ui design",
    "ux": "ui design"
}

# ---------- LEARNING RESOURCES WITH DURATION ----------
LEARNING_RESOURCES = {
    # Critical Skills (2-3 months each)
    "python": {
        "resources": ["Python Crash Course (book)", "Automate the Boring Stuff", "100 Days of Code"],
        "duration": "2-3 months",
        "difficulty": "Beginner"
    },
    "data structures": {
        "resources": ["Grokking DS & Algorithms", "LeetCode Easy→Medium (50 problems)", "CS50"],
        "duration": "3-4 months",
        "difficulty": "Intermediate"
    },
    "algorithms": {
        "resources": ["CLRS (book)", "LeetCode pattern-based practice", "AlgoExpert"],
        "duration": "3-4 months",
        "difficulty": "Intermediate"
    },
    "javascript": {
        "resources": ["JavaScript.info", "Eloquent JavaScript", "FreeCodeCamp JS"],
        "duration": "2-3 months",
        "difficulty": "Beginner"
    },
    "react": {
        "resources": ["Official React Docs", "React Tutorial (scrimba)", "Build 3 projects"],
        "duration": "1-2 months",
        "difficulty": "Intermediate"
    },
    "sql": {
        "resources": ["Mode SQL Tutorial", "SQLBolt", "HackerRank SQL practice"],
        "duration": "1-2 months",
        "difficulty": "Beginner"
    },
    "machine learning": {
        "resources": ["Andrew Ng Coursera", "Hands-on ML with Scikit-Learn", "Kaggle competitions"],
        "duration": "3-4 months",
        "difficulty": "Advanced"
    },
    "git": {
        "resources": ["Pro Git (free book)", "GitHub Learning Lab", "Practice branching"],
        "duration": "2-3 weeks",
        "difficulty": "Beginner"
    },
    "docker": {
        "resources": ["Docker Docs", "Docker Mastery (Udemy)", "Deploy 2-3 apps"],
        "duration": "1 month",
        "difficulty": "Intermediate"
    },
    "html": {
        "resources": ["MDN HTML Guide", "FreeCodeCamp Responsive Web Design"],
        "duration": "2-3 weeks",
        "difficulty": "Beginner"
    },
    "css": {
        "resources": ["CSS Grid & Flexbox", "Tailwind CSS Tutorial", "Build 5 layouts"],
        "duration": "1-2 months",
        "difficulty": "Beginner"
    },
    "rest apis": {
        "resources": ["RESTful API Design", "Postman practice", "Build CRUD API"],
        "duration": "1 month",
        "difficulty": "Intermediate"
    },
    "node.js": {
        "resources": ["Node.js Docs", "Express.js Tutorial", "Build REST API"],
        "duration": "1-2 months",
        "difficulty": "Intermediate"
    },
    "tensorflow": {
        "resources": ["TensorFlow Official Tutorials", "Deep Learning Specialization", "Kaggle kernels"],
        "duration": "2-3 months",
        "difficulty": "Advanced"
    },
    "statistics": {
        "resources": ["Khan Academy Statistics", "StatQuest YouTube", "Think Stats (book)"],
        "duration": "2-3 months",
        "difficulty": "Intermediate"
    },
    "debugging": {
        "resources": ["Debugging techniques", "Chrome DevTools", "Practice on real bugs"],
        "duration": "1 month",
        "difficulty": "Beginner"
    },
    "excel": {
        "resources": ["Excel Essential Training", "PivotTables & VLOOKUP", "Practice datasets"],
        "duration": "1 month",
        "difficulty": "Beginner"
    },
    "tableau": {
        "resources": ["Tableau Public Gallery", "Tableau Desktop Tutorial", "Build 5 dashboards"],
        "duration": "1-2 months",
        "difficulty": "Intermediate"
    },
    "communication": {
        "resources": ["Public speaking course", "Writing skills workshop", "Practice presentations"],
        "duration": "2-3 months",
        "difficulty": "Beginner"
    },
    "product strategy": {
        "resources": ["Inspired (book)", "Product School", "Case studies"],
        "duration": "2-3 months",
        "difficulty": "Intermediate"
    },
    "ui design": {
        "resources": ["Refactoring UI", "Daily UI Challenge", "Dribbble inspiration"],
        "duration": "2-3 months",
        "difficulty": "Intermediate"
    },
    "figma": {
        "resources": ["Figma Tutorial", "Design system practice", "Build 10 screens"],
        "duration": "1 month",
        "difficulty": "Beginner"
    }
}

# Generic fallback
_GENERIC_RESOURCE = {
    "resources": ["Online course (Coursera/Udemy)", "Official documentation", "Hands-on projects"],
    "duration": "1-2 months",
    "difficulty": "Intermediate"
}


# ---------- SKILLS ENGINE CLASS ----------
class SkillsEngine:
    def __init__(self, canonical_skills=None, aliases=None, resources=None, fuzzy_threshold=80):
        self.canonical = canonical_skills or _ALL_CANONICAL_SKILLS
        self.aliases = aliases or SKILL_ALIASES
        self.resources = resources or LEARNING_RESOURCES
        self.threshold = fuzzy_threshold

    def _clean_token(self, tok):
        if not tok:
            return ""
        tok = str(tok).strip().lower()
        tok = re.sub(r"[^a-z0-9\+\-\.\s#/]", " ", tok)
        tok = re.sub(r"\s+", " ", tok).strip()
        return tok

    def _alias_map(self, token):
        return self.aliases.get(token, token)

    def fuzzy_match(self, token):
        tok = self._clean_token(token)
        if not tok:
            return None
        
        # Direct alias
        if tok in self.aliases:
            return self.aliases[tok]

        # Exact match
        if tok in self.canonical:
            return tok

        # Fuzzy matching
        if _HAS_RAPIDFUZZ:
            choice, score, _ = rf_process.extractOne(tok, self.canonical, scorer=rf_fuzz.token_sort_ratio)
            if score >= self.threshold:
                return choice
            choice2, score2, _ = rf_process.extractOne(tok, self.canonical, scorer=rf_fuzz.partial_ratio)
            if score2 >= self.threshold:
                return choice2
            return None
        else:
            matches = difflib.get_close_matches(tok, self.canonical, n=1, cutoff=self.threshold/100.0)
            return matches[0] if matches else None

    def normalize_and_map(self, raw_token):
        tok = self._clean_token(raw_token)
        if not tok:
            return None
        tok = self._alias_map(tok)
        mapped = self.fuzzy_match(tok)
        return mapped or tok

    def extract_skills_from_text(self, text):
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
        fields = [
            "Technical Skills", "Soft Skills", "Course Keywords", "Project Keywords",
            "Preferred Roles", "Work Keywords", "Skill Embedding", "Graduate Major",
            "PG Major", "Highest Education"
        ]
        s = set()
        for f in fields:
            if f in row:
                s |= self.extract_skills_from_text(row.get(f, ""))
        return s

    def compute_gap_for_role(self, user_skills_set, role_name):
        """Enhanced gap analysis with priorities"""
        role_data = ROLE_SKILLS.get(role_name, {})
        
        result = {
            "role": role_name,
            "critical": {"have": [], "missing": []},
            "important": {"have": [], "missing": []},
            "nice_to_have": {"have": [], "missing": []}
        }
        
        for priority in ["critical", "important", "nice_to_have"]:
            reqs = [r.lower() for r in role_data.get(priority, [])]
            have = [r for r in reqs if any((r in u) or (u in r) for u in user_skills_set)]
            missing = [r for r in reqs if r not in have]
            
            result[priority]["have"] = sorted(have)
            result[priority]["missing"] = sorted(missing)
        
        return result

    def recommend_alternatives(self, user_skills_set, top_n=3):
        scores = []
        for role, role_data in ROLE_SKILLS.items():
            all_reqs = []
            for priority in ["critical", "important", "nice_to_have"]:
                all_reqs.extend([r.lower() for r in role_data.get(priority, [])])
            
            overlap = sum(1 for r in all_reqs if any((r in u) or (u in r) for u in user_skills_set))
            scores.append((role, overlap, len(all_reqs)))
        
        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
        return [(r, o, t) for r, o, t in scores_sorted[:top_n]]

    def make_learning_paths(self, missing_skills):
        """Generate learning paths with duration estimates"""
        out = {}
        for ms in missing_skills:
            ms_lower = ms.lower()
            if ms_lower in self.resources:
                out[ms_lower] = self.resources[ms_lower]
            else:
                # Find related resource
                found = None
                for k in self.resources.keys():
                    if k in ms_lower or ms_lower in k:
                        found = self.resources[k]
                        break
                out[ms_lower] = found or _GENERIC_RESOURCE
        return out

    def estimate_effort(self, missing_count):
        """Estimate total learning effort"""
        if missing_count <= 2:
            return "Low (1-2 months)"
        elif missing_count <= 5:
            return "Medium (3-6 months)"
        else:
            return "High (6-12 months)"


# Convenience instance
_DEFAULT_ENGINE = SkillsEngine()

# Exported helpers
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

def estimate_effort(missing_count):
    return _DEFAULT_ENGINE.estimate_effort(missing_count)