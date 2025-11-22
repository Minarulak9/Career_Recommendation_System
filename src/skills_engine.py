# skills_engine.py
"""
Production-ready skills engine with:
- Clean skill extraction (no garbage)
- Fuzzy matching
- Skill priorities
- Learning paths with duration
"""

import re
from pathlib import Path

try:
    from rapidfuzz import process as rf_process
    from rapidfuzz import fuzz as rf_fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    import difflib
    _HAS_RAPIDFUZZ = False


# ============================================================
# ROLE SKILLS WITH PRIORITIES
# ============================================================
ROLE_SKILLS = {
    "Software Engineer": {
        "critical": ["python", "data structures", "algorithms", "git", "debugging"],
        "important": ["object oriented programming", "rest apis", "sql", "unit testing", "system design"],
        "nice_to_have": ["docker", "kubernetes", "ci/cd", "aws", "design patterns"]
    },
    
    "Frontend Developer": {
        "critical": ["html", "css", "javascript", "react"],
        "important": ["responsive design", "typescript", "state management", "webpack"],
        "nice_to_have": ["next.js", "tailwind", "figma", "accessibility", "vue"]
    },
    
    "Backend Developer": {
        "critical": ["python", "node.js", "sql", "rest apis"],
        "important": ["databases", "authentication", "docker", "mongodb"],
        "nice_to_have": ["graphql", "microservices", "kafka", "redis", "kubernetes"]
    },
    
    "AI Engineer": {
        "critical": ["python", "machine learning", "statistics", "tensorflow"],
        "important": ["deep learning", "nlp", "data preprocessing", "model evaluation"],
        "nice_to_have": ["pytorch", "mlops", "computer vision", "kubernetes"]
    },
    
    "Data Analyst": {
        "critical": ["sql", "excel", "statistics", "data visualization"],
        "important": ["tableau", "power bi", "pandas", "data cleaning"],
        "nice_to_have": ["python", "dashboarding", "business intelligence"]
    },
    
    "Data Scientist": {
        "critical": ["python", "machine learning", "statistics", "sql"],
        "important": ["feature engineering", "pandas", "numpy", "modeling"],
        "nice_to_have": ["mlops", "deep learning", "visualization"]
    },
    
    "Project Manager": {
        "critical": ["communication", "stakeholder management", "roadmap"],
        "important": ["product strategy", "user research", "prioritization"],
        "nice_to_have": ["analytics", "okrs", "wireframing", "agile"]
    },
    
    "Designer": {
        "critical": ["ui design", "figma", "wireframing", "user research"],
        "important": ["prototyping", "design systems", "accessibility", "visual design"],
        "nice_to_have": ["animation", "illustration", "branding", "sketch"]
    }
}

# Flatten all canonical skills
_ALL_CANONICAL_SKILLS = set()
for role_data in ROLE_SKILLS.values():
    for priority in ["critical", "important", "nice_to_have"]:
        _ALL_CANONICAL_SKILLS.update([s.lower() for s in role_data.get(priority, [])])
_ALL_CANONICAL_SKILLS = sorted(_ALL_CANONICAL_SKILLS)


# ============================================================
# GARBAGE TOKENS TO FILTER OUT
# ============================================================
GARBAGE_TOKENS = {
    # Generic/meaningless
    "auto", "nan", "none", "null", "missing", "na", "n/a", "",
    "course list", "project description", "work detail",
    
    # Education (not skills)
    "bcom", "btech", "bca", "mca", "bba", "mba", "bsc", "msc",
    "ba", "ma", "be", "me", "phd", "diploma", "btech cse",
    "btech mechanical", "bsc math", "bsc physics", "ba english",
    "btecch", "mtecch", "mtech",
    
    # Job roles (not skills)
    "software engineer", "data analyst", "ai engineer",
    "frontend developer", "backend developer", "designer",
    "project manager", "data scientist",
    
    # Locations
    "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad",
    "chennai", "kolkata", "pune", "ahmedabad", "jaipur",
    "lucknow", "patna", "bhopal", "indore", "nagpur",
    "kochi", "guwahati",
    
    # Random noise
    "male", "female", "other", "student", "working professional",
    "entry level", "mid level", "senior level", "fresher",
    "hybrid", "remote", "office", "flexible"
}


# ============================================================
# SKILL ALIASES
# ============================================================
SKILL_ALIASES = {
    "js": "javascript",
    "reactjs": "react",
    "react.js": "react",
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
    "ci cd": "ci/cd",
    "ui ux": "ui design",
    "ux": "ui design",
    "aws cloud": "aws",
    "amazon web services": "aws",
    "gcp": "google cloud",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "scikit-learn": "sklearn",
    "scikit learn": "sklearn"
}


# ============================================================
# LEARNING RESOURCES
# ============================================================
LEARNING_RESOURCES = {
    "python": {
        "resources": ["Python Crash Course (book)", "Automate the Boring Stuff", "100 Days of Code"],
        "duration": "2-3 months",
        "difficulty": "Beginner"
    },
    "data structures": {
        "resources": ["Grokking DS & Algorithms", "LeetCode (50 Easy problems)", "CS50"],
        "duration": "3-4 months",
        "difficulty": "Intermediate"
    },
    "algorithms": {
        "resources": ["CLRS (book)", "LeetCode patterns", "AlgoExpert"],
        "duration": "3-4 months",
        "difficulty": "Intermediate"
    },
    "javascript": {
        "resources": ["JavaScript.info", "Eloquent JavaScript", "FreeCodeCamp"],
        "duration": "2-3 months",
        "difficulty": "Beginner"
    },
    "react": {
        "resources": ["Official React Docs", "Scrimba React Course", "Build 3 projects"],
        "duration": "1-2 months",
        "difficulty": "Intermediate"
    },
    "sql": {
        "resources": ["Mode SQL Tutorial", "SQLBolt", "HackerRank SQL"],
        "duration": "1-2 months",
        "difficulty": "Beginner"
    },
    "machine learning": {
        "resources": ["Andrew Ng Coursera", "Hands-on ML with Scikit-Learn", "Kaggle"],
        "duration": "3-4 months",
        "difficulty": "Advanced"
    },
    "git": {
        "resources": ["Pro Git (free book)", "GitHub Learning Lab", "Daily practice"],
        "duration": "2-3 weeks",
        "difficulty": "Beginner"
    },
    "docker": {
        "resources": ["Docker Docs", "Docker Mastery (Udemy)", "Deploy 2 apps"],
        "duration": "1 month",
        "difficulty": "Intermediate"
    },
    "html": {
        "resources": ["MDN HTML Guide", "FreeCodeCamp Responsive Web"],
        "duration": "2-3 weeks",
        "difficulty": "Beginner"
    },
    "css": {
        "resources": ["CSS Grid & Flexbox", "Tailwind CSS", "Build 5 layouts"],
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
        "resources": ["TensorFlow Tutorials", "Deep Learning Specialization"],
        "duration": "2-3 months",
        "difficulty": "Advanced"
    },
    "statistics": {
        "resources": ["Khan Academy Statistics", "StatQuest YouTube", "Think Stats"],
        "duration": "2-3 months",
        "difficulty": "Intermediate"
    },
    "debugging": {
        "resources": ["Debugging techniques", "Chrome DevTools", "Practice daily"],
        "duration": "1 month",
        "difficulty": "Beginner"
    },
    "communication": {
        "resources": ["Public speaking course", "Technical writing", "Practice presentations"],
        "duration": "2-3 months",
        "difficulty": "Beginner"
    },
    "ui design": {
        "resources": ["Refactoring UI", "Daily UI Challenge", "Dribbble study"],
        "duration": "2-3 months",
        "difficulty": "Intermediate"
    },
    "figma": {
        "resources": ["Figma Tutorial", "Design system practice", "Recreate 10 screens"],
        "duration": "1 month",
        "difficulty": "Beginner"
    }
}

_GENERIC_RESOURCE = {
    "resources": ["Online course (Coursera/Udemy)", "Official documentation", "Build projects"],
    "duration": "1-2 months",
    "difficulty": "Intermediate"
}


# ============================================================
# SKILLS ENGINE CLASS
# ============================================================
class SkillsEngine:
    def __init__(self, fuzzy_threshold=75):
        self.canonical = _ALL_CANONICAL_SKILLS
        self.aliases = SKILL_ALIASES
        self.resources = LEARNING_RESOURCES
        self.threshold = fuzzy_threshold
        self.garbage = GARBAGE_TOKENS

    def _clean_token(self, tok):
        """Clean and normalize a token."""
        if not tok:
            return ""
        tok = str(tok).strip().lower()
        tok = re.sub(r"[^a-z0-9\+\-\.\s#/]", " ", tok)
        tok = re.sub(r"\s+", " ", tok).strip()
        return tok

    def _is_garbage(self, token):
        """Check if token is garbage."""
        return token.lower() in self.garbage

    def _alias_map(self, token):
        """Map aliases to canonical names."""
        return self.aliases.get(token, token)

    def fuzzy_match(self, token):
        """Fuzzy match token to canonical skill."""
        tok = self._clean_token(token)
        if not tok or len(tok) < 2:
            return None
        
        # Check garbage
        if self._is_garbage(tok):
            return None
        
        # Direct alias
        if tok in self.aliases:
            return self.aliases[tok]

        # Exact match
        if tok in self.canonical:
            return tok

        # Fuzzy matching
        if _HAS_RAPIDFUZZ:
            result = rf_process.extractOne(
                tok, self.canonical,
                scorer=rf_fuzz.token_sort_ratio
            )
            if result and result[1] >= self.threshold:
                return result[0]
            return None
        else:
            matches = difflib.get_close_matches(
                tok, self.canonical,
                n=1, cutoff=self.threshold / 100.0
            )
            return matches[0] if matches else None

    def extract_skills_from_text(self, text):
        """Extract clean skills from text."""
        if text is None or pd.isna(text) if 'pd' in dir() else text is None:
            return set()
        
        if isinstance(text, (list, set)):
            tokens = list(text)
        else:
            s = str(text)
            # Split by various delimiters
            for sep in [",", ";", "|", "/", "\n", " and "]:
                s = s.replace(sep, ",")
            tokens = [t.strip() for t in s.split(",") if t.strip()]

        skills = set()
        for t in tokens:
            clean = self._clean_token(t)
            
            # Skip garbage
            if self._is_garbage(clean):
                continue
            
            # Try to match
            mapped = self.fuzzy_match(clean)
            if mapped:
                skills.add(mapped)
        
        return skills

    def extract_from_row(self, row):
        """Extract skills from a data row."""
        fields = [
            "Technical Skills", "Soft Skills"
        ]
        
        skills = set()
        for f in fields:
            if f in row.index:
                val = row.get(f, "")
                skills |= self.extract_skills_from_text(val)
        
        return skills

    def compute_gap_for_role(self, user_skills, role_name):
        """Compute skill gap with priorities."""
        role_data = ROLE_SKILLS.get(role_name, {})
        
        result = {
            "role": role_name,
            "critical": {"have": [], "missing": []},
            "important": {"have": [], "missing": []},
            "nice_to_have": {"have": [], "missing": []}
        }
        
        user_skills_lower = {s.lower() for s in user_skills}
        
        for priority in ["critical", "important", "nice_to_have"]:
            reqs = [r.lower() for r in role_data.get(priority, [])]
            
            have = []
            missing = []
            
            for r in reqs:
                # Check if user has this skill (exact or partial match)
                matched = any(
                    r in u or u in r
                    for u in user_skills_lower
                )
                if matched:
                    have.append(r)
                else:
                    missing.append(r)
            
            result[priority]["have"] = sorted(have)
            result[priority]["missing"] = sorted(missing)
        
        return result

    def recommend_alternatives(self, user_skills, exclude_role=None, top_n=3):
        """Recommend alternative career paths."""
        user_skills_lower = {s.lower() for s in user_skills}
        scores = []
        
        for role, role_data in ROLE_SKILLS.items():
            if role == exclude_role:
                continue
            
            # Count matching skills
            all_skills = []
            for priority in ["critical", "important", "nice_to_have"]:
                all_skills.extend([s.lower() for s in role_data.get(priority, [])])
            
            total = len(all_skills)
            matched = sum(
                1 for s in all_skills
                if any(s in u or u in s for u in user_skills_lower)
            )
            
            scores.append((role, matched, total))
        
        # Sort by match count
        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores_sorted[:top_n]

    def make_learning_paths(self, missing_skills, top_n=5):
        """Generate learning paths for missing skills."""
        paths = {}
        
        for skill in missing_skills[:top_n]:
            skill_lower = skill.lower()
            
            if skill_lower in self.resources:
                paths[skill_lower] = self.resources[skill_lower]
            else:
                # Try partial match
                found = None
                for k in self.resources.keys():
                    if k in skill_lower or skill_lower in k:
                        found = self.resources[k]
                        break
                paths[skill_lower] = found or _GENERIC_RESOURCE
        
        return paths

    def estimate_effort(self, missing_count):
        """Estimate learning effort based on missing skills."""
        if missing_count == 0:
            return "Ready to apply!"
        elif missing_count <= 2:
            return "Low (1-2 months)"
        elif missing_count <= 5:
            return "Medium (3-6 months)"
        else:
            return "High (6-12 months)"

    def calculate_match_score(self, user_skills, role_name):
        """Calculate percentage match for a role."""
        role_data = ROLE_SKILLS.get(role_name, {})
        user_skills_lower = {s.lower() for s in user_skills}
        
        all_skills = []
        for priority in ["critical", "important", "nice_to_have"]:
            all_skills.extend([s.lower() for s in role_data.get(priority, [])])
        
        if not all_skills:
            return 0
        
        matched = sum(
            1 for s in all_skills
            if any(s in u or u in s for u in user_skills_lower)
        )
        
        return int((matched / len(all_skills)) * 100)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================
_DEFAULT_ENGINE = SkillsEngine()

def extract_skills_from_row(row):
    return _DEFAULT_ENGINE.extract_from_row(row)

def compute_role_gap(user_skills, role):
    return _DEFAULT_ENGINE.compute_gap_for_role(user_skills, role)

def suggest_alternatives(user_skills, exclude_role=None, top_n=3):
    return _DEFAULT_ENGINE.recommend_alternatives(user_skills, exclude_role, top_n)

def make_learning_paths(missing_skills, top_n=5):
    return _DEFAULT_ENGINE.make_learning_paths(missing_skills, top_n)

def estimate_effort(missing_count):
    return _DEFAULT_ENGINE.estimate_effort(missing_count)

def calculate_match_score(user_skills, role_name):
    return _DEFAULT_ENGINE.calculate_match_score(user_skills, role_name)