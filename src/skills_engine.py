"""
skills_engine.py — Standard Upgrade (real courses + projects)

- Adds realistic, academic-friendly learning resources for ~50 skills.
- Each skill entry contains:
    - courses (short list of recommended courses/books/playlists)
    - projects (practical project ideas)
    - duration (typical time to reach competence)
    - difficulty (Beginner / Intermediate / Advanced)
- Keeps all previous engine features: extraction, gap analysis, seniority,
  role-match scoring, project recommendation, learning_path, effort estimation.
"""

import re
import difflib

# ============================================================
# ROLE SKILLS (unchanged)
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

# ============================================================
# PROJECT_LIBRARY (unchanged structure)
# ============================================================
PROJECT_LIBRARY = {
    "Software Engineer": [
        "Build a Todo App with backend + authentication",
        "Implement a URL shortener using REST API",
        "Build a CLI tool (password manager)"
    ],
    "Frontend Developer": [
        "Build a responsive React portfolio",
        "Create a SaaS dashboard UI",
        "Clone a popular web UI"
    ],
    "Backend Developer": [
        "Build REST API with JWT Auth",
        "E-commerce backend with payments",
        "Role-based authentication system"
    ],
    "AI Engineer": [
        "Image Classifier (CIFAR-10)",
        "Sentiment Analysis NLP model",
        "ML training pipeline project"
    ],
    "Data Analyst": [
        "Sales Insights Dashboard",
        "EDA on Kaggle datasets",
        "SQL + BI reporting pipeline"
    ],
}

# ============================================================
# LEARNING_RESOURCES (STANDARD UPGRADE: ~50 skills)
# Each key: courses (list), projects (list), duration, difficulty
# ============================================================
LEARNING_RESOURCES = {
    # Core programming + CS
    "python": {
        "courses": ["Python for Everybody (Coursera - Dr. Charles Severance)", "Automate the Boring Stuff (Al Sweigart)"],
        "projects": ["CLI tool (file manager)", "Web scraper + CSV export"],
        "duration": "1–2 months",
        "difficulty": "Beginner"
    },
    "data structures": {
        "courses": ["Grokking the Coding Interview (educative)", "Data Structures - UC San Diego (Coursera)"],
        "projects": ["Implement linked lists, stacks, queues", "Build a binary tree visualizer"],
        "duration": "2–3 months",
        "difficulty": "Intermediate"
    },
    "algorithms": {
        "courses": ["Algorithms (Princeton / Coursera)", "LeetCode problem patterns"],
        "projects": ["Sorting visualizer", "Solve 50 LeetCode problems (easy→medium)"],
        "duration": "2–3 months",
        "difficulty": "Intermediate"
    },
    "object oriented programming": {
        "courses": ["OOP in Python/Java (Udemy/edX)", "CS50 object-oriented modules"],
        "projects": ["Design a simple banking system (classes + persistence)"],
        "duration": "1–2 months",
        "difficulty": "Beginner→Intermediate"
    },
    "git": {
        "courses": ["Pro Git (book) + Git & GitHub (Coursera)"],
        "projects": ["Host projects on GitHub, use branches and PRs"],
        "duration": "2–3 weeks",
        "difficulty": "Beginner"
    },
    "debugging": {
        "courses": ["Debugging techniques (YouTube series)", "Practical Debugging (Pluralsight)"],
        "projects": ["Debug a provided broken codebase, write tests"],
        "duration": "1 month",
        "difficulty": "Beginner→Intermediate"
    },

    # Backend / APIs / DevOps
    "rest apis": {
        "courses": ["REST API Design (Pluralsight)", "Building APIs with Flask/Django (Udemy)"],
        "projects": ["CRUD REST API with authentication", "API rate-limiting demo"],
        "duration": "1 month",
        "difficulty": "Intermediate"
    },
    "restapi": {  # alias / alternative key
        "courses": ["REST fundamentals (YouTube)"],
        "projects": ["Simple REST service"],
        "duration": "1 month",
        "difficulty": "Intermediate"
    },
    "http": {
        "courses": ["HTTP Fundamentals (MDN + free resources)"],
        "projects": ["Inspect HTTP traffic and build request/response examples"],
        "duration": "2 weeks",
        "difficulty": "Beginner"
    },
    "authentication": {
        "courses": ["Auth in Web Apps (Auth0 docs)", "JWT Essentials (YouTube)"],
        "projects": ["Implement JWT login + refresh tokens"],
        "duration": "2–3 weeks",
        "difficulty": "Intermediate"
    },
    "docker": {
        "courses": ["Docker Mastery (Udemy)", "Docker Official Docs"],
        "projects": ["Containerize a Flask/Node app", "Docker Compose for multi-service app"],
        "duration": "1 month",
        "difficulty": "Intermediate"
    },
    "kubernetes": {
        "courses": ["Kubernetes Basics (Katacoda)", "Kubernetes for Developers (Udemy)"],
        "projects": ["Deploy app to local k8s (minikube)"],
        "duration": "1–2 months",
        "difficulty": "Advanced"
    },
    "ci/cd": {
        "courses": ["CI/CD with GitHub Actions (Docs)", "Jenkins basics (YouTube)"],
        "projects": ["Build CI pipeline for tests + deployment"],
        "duration": "1 month",
        "difficulty": "Intermediate"
    },
    "aws": {
        "courses": ["AWS Cloud Practitioner + Developer fundamentals", "AWS hands-on labs"],
        "projects": ["Deploy app to EC2 / use S3 for storage"],
        "duration": "1–2 months",
        "difficulty": "Intermediate"
    },

    # Databases
    "sql": {
        "courses": ["SQLBolt", "Mode Analytics SQL Tutorial"],
        "projects": ["Design a normalized DB schema + complex joins"],
        "duration": "1 month",
        "difficulty": "Beginner"
    },
    "mysql": {
        "courses": ["MySQL for Beginners (Udemy)"],
        "projects": ["Build a sample app with MySQL backend"],
        "duration": "1 month",
        "difficulty": "Beginner"
    },
    "postgresql": {
        "courses": ["Introduction to PostgreSQL (edX/YouTube)"],
        "projects": ["Use Postgres features (indexes, views)"],
        "duration": "1 month",
        "difficulty": "Intermediate"
    },
    "mongodb": {
        "courses": ["MongoDB University free courses"],
        "projects": ["Build a document-store-backed app"],
        "duration": "3–4 weeks",
        "difficulty": "Beginner→Intermediate"
    },
    "databases": {
        "courses": ["Database systems (Stanford/edX)"],
        "projects": ["ER modeling + implement schema"],
        "duration": "1–2 months",
        "difficulty": "Intermediate"
    },

    # Frontend / UI
    "html": {
        "courses": ["MDN HTML Guide", "FreeCodeCamp HTML"],
        "projects": ["Static landing page, semantic HTML"],
        "duration": "2–3 weeks",
        "difficulty": "Beginner"
    },
    "css": {
        "courses": ["CSS Grid & Flexbox (FreeCodeCamp)", "Advanced CSS (Udemy)"],
        "projects": ["Responsive landing page, CSS-only components"],
        "duration": "1 month",
        "difficulty": "Beginner→Intermediate"
    },
    "javascript": {
        "courses": ["JavaScript.info", "Eloquent JavaScript"],
        "projects": ["To-do app, fetch API exercises"],
        "duration": "1–2 months",
        "difficulty": "Beginner"
    },
    "react": {
        "courses": ["Official React Docs", "Scrimba React Course"],
        "projects": ["Portfolio with React + routing + state"],
        "duration": "1–2 months",
        "difficulty": "Intermediate"
    },
    "typescript": {
        "courses": ["TypeScript Basics (Microsoft docs)", "TypeScript Deep Dive (book)"],
        "projects": ["Convert a JS project to TypeScript"],
        "duration": "1 month",
        "difficulty": "Intermediate"
    },
    "webpack": {
        "courses": ["Webpack fundamentals (YouTube)"],
        "projects": ["Custom build pipeline"],
        "duration": "2–3 weeks",
        "difficulty": "Intermediate"
    },
    "tailwind": {
        "courses": ["Tailwind Docs + Crash Courses"],
        "projects": ["Design a dashboard with Tailwind CSS"],
        "duration": "2–3 weeks",
        "difficulty": "Beginner"
    },

    # Data stack / Analytics / Visualization
    "pandas": {
        "courses": ["Data Analysis with Pandas (Kaggle/YouTube)"],
        "projects": ["Data cleaning pipeline & summary report"],
        "duration": "1 month",
        "difficulty": "Beginner→Intermediate"
    },
    "numpy": {
        "courses": ["NumPy Quickstart (Official)"],
        "projects": ["Matrix operations & small ML helpers"],
        "duration": "2–3 weeks",
        "difficulty": "Beginner"
    },
    "tableau": {
        "courses": ["Tableau Public tutorials"],
        "projects": ["Create interactive dashboard with sample data"],
        "duration": "2–3 weeks",
        "difficulty": "Beginner"
    },
    "power bi": {
        "courses": ["Power BI Guided Learning (Microsoft)"],
        "projects": ["Build a sales dashboard"],
        "duration": "2–3 weeks",
        "difficulty": "Beginner"
    },
    "data cleaning": {
        "courses": ["Data Cleaning (Kaggle micro-courses)"],
        "projects": ["Clean a messy dataset & document steps"],
        "duration": "2–3 weeks",
        "difficulty": "Beginner"
    },
    "data visualization": {
        "courses": ["Storytelling with Data", "Matplotlib / Seaborn tutorials"],
        "projects": ["Create charts & explain insights (report)"],
        "duration": "2–3 weeks",
        "difficulty": "Beginner→Intermediate"
    },

    # Machine learning / AI
    "machine learning": {
        "courses": ["Machine Learning by Andrew Ng (Coursera)", "Hands-On ML with Scikit-Learn (book)"],
        "projects": ["Iris / Titanic classifier, end-to-end pipeline"],
        "duration": "3–4 months",
        "difficulty": "Advanced"
    },
    "deep learning": {
        "courses": ["Deep Learning Specialization (Andrew Ng)", "FastAI courses"],
        "projects": ["Image classifier (transfer learning)"],
        "duration": "3–4 months",
        "difficulty": "Advanced"
    },
    "tensorflow": {
        "courses": ["TensorFlow in Practice (Coursera)"],
        "projects": ["Keras-based image classifier"],
        "duration": "2–3 months",
        "difficulty": "Advanced"
    },
    "pytorch": {
        "courses": ["Deep Learning with PyTorch (Udacity)"],
        "projects": ["Build PyTorch models and trainers"],
        "duration": "2–3 months",
        "difficulty": "Advanced"
    },
    "sklearn": {
        "courses": ["scikit-learn tutorials (official)"],
        "projects": ["Build classical ML models and evaluate"],
        "duration": "1–2 months",
        "difficulty": "Intermediate"
    },
    "model evaluation": {
        "courses": ["ML evaluation metrics (Coursera/YouTube)"],
        "projects": ["ROC, PR curves, cross-validation exercises"],
        "duration": "2–3 weeks",
        "difficulty": "Intermediate"
    },
    "feature engineering": {
        "courses": ["Feature Engineering for ML (Course/YouTube)"],
        "projects": ["Create engineered features and compare models"],
        "duration": "1 month",
        "difficulty": "Intermediate"
    },
    "nlp": {
        "courses": ["NLP with Deep Learning (Stanford CS224n videos)", "Hugging Face tutorials"],
        "projects": ["Sentiment analysis, text classification pipeline"],
        "duration": "2–3 months",
        "difficulty": "Advanced"
    },
    "computer vision": {
        "courses": ["Intro to CV (Coursera/YouTube)"],
        "projects": ["Image classification with augmentation"],
        "duration": "2–3 months",
        "difficulty": "Advanced"
    },

    # Software quality / testing / architecture
    "unit testing": {
        "courses": ["Unit Testing in Python (Real Python)", "Testing JavaScript (Jest)"],
        "projects": ["Add tests to an existing project, CI integration"],
        "duration": "2–3 weeks",
        "difficulty": "Beginner→Intermediate"
    },
    "system design": {
        "courses": ["System Design Primer (GitHub)", "Grokking System Design (educative)"],
        "projects": ["Design a URL shortener / scaled service blueprint"],
        "duration": "1–2 months",
        "difficulty": "Advanced"
    },
    "design patterns": {
        "courses": ["Design Patterns in OOP (book/course)"],
        "projects": ["Implement common patterns in small projects"],
        "duration": "1–2 months",
        "difficulty": "Intermediate"
    },

    # Soft / product skills
    "communication": {
        "courses": ["Technical Communication (Coursera)", "Toastmasters resources"],
        "projects": ["Prepare a project demo video & write docs"],
        "duration": "1 month",
        "difficulty": "Beginner"
    },
    "teamwork": {
        "courses": ["Collaboration courses (LinkedIn Learning)"],
        "projects": ["Contribute to an open-source repo"],
        "duration": "1 month",
        "difficulty": "Beginner"
    },
    "product strategy": {
        "courses": ["Product Management basics (Coursera)"],
        "projects": ["Write a product spec for your flagship project"],
        "duration": "1 month",
        "difficulty": "Intermediate"
    },

    # UX / Design
    "ui design": {
        "courses": ["Refactoring UI (book)", "Figma Crash Course (YouTube)"],
        "projects": ["Design a 5-screen mobile app in Figma"],
        "duration": "1–2 months",
        "difficulty": "Intermediate"
    },
    "figma": {
        "courses": ["Figma official tutorials", "UI design bootcamps (YouTube)"],
        "projects": ["Design and prototype a landing page"],
        "duration": "3–4 weeks",
        "difficulty": "Beginner→Intermediate"
    },
    "prototyping": {
        "courses": ["Prototyping in Figma tutorials"],
        "projects": ["Clickable prototype of a product idea"],
        "duration": "2–3 weeks",
        "difficulty": "Beginner"
    },
    "accessibility": {
        "courses": ["Web Accessibility (WAI, MDN)"],
        "projects": ["Make an accessible form and audit it"],
        "duration": "2–3 weeks",
        "difficulty": "Beginner"
    },

    # Analytics / BI extras
    "excel": {
        "courses": ["Excel Skills for Business (Coursera)", "Excel tutorials (YouTube)"],
        "projects": ["Build financial model / pivot tables"],
        "duration": "2–3 weeks",
        "difficulty": "Beginner"
    },
    "business intelligence": {
        "courses": ["Intro to BI tools (Power BI / Tableau)"],
        "projects": ["Create BI dashboards from raw data"],
        "duration": "1 month",
        "difficulty": "Intermediate"
    },

    # Networking / misc
    "kafka": {
        "courses": ["Kafka Basics (Confluent tutorials)"],
        "projects": ["Simple producer-consumer pipeline"],
        "duration": "1 month",
        "difficulty": "Advanced"
    },
    "redis": {
        "courses": ["Redis University free courses"],
        "projects": ["Cache integration for an API"],
        "duration": "2–3 weeks",
        "difficulty": "Intermediate"
    }
}

# Generic fallback resource
GENERIC_RESOURCE = {
    "courses": ["Official documentation", "YouTube guided playlist"],
    "projects": ["Build a small practical project"],
    "duration": "1–2 months",
    "difficulty": "Intermediate"
}

# ============================================================
# NORMALIZATION / ALIASES / NOISE
# ============================================================
SKILL_ALIASES = {
    "js": "javascript",
    "reactjs": "react",
    "nodejs": "node.js",
    "py": "python",
    "ml": "machine learning",
    "dl": "deep learning",
    "ds": "data structures",
    "expressjs": "express",
}

NOISE = {"", "none", "null", "nan", "student", "fresher"}

# ============================================================
# FINAL SKILLS ENGINE (unchanged APIs)
# ============================================================
class SkillsEngine:
    def _clean(self, text):
        if not text:
            return ""
        text = str(text).lower().strip()
        text = re.sub(r"[^a-z0-9\s.+-]", " ", text)
        return " ".join(text.split())

    def extract_from_text(self, txt):
        txt = self._clean(txt)
        tokens = set(re.split(r"[ ,;/\n]+", txt))
        return {SKILL_ALIASES.get(t, t) for t in tokens if t not in NOISE}

    def extract_from_row(self, row):
        skills = set()
        for col in ["Technical Skills", "Soft Skills"]:
            if col in row:
                skills |= self.extract_from_text(row[col])
        return skills

    def compute_gap(self, user_skills, role):
        mapping = ROLE_SKILLS[role]
        gap = {}
        for priority, items in mapping.items():
            have = [s for s in items if s in user_skills]
            missing = [s for s in items if s not in user_skills]
            gap[priority] = {"have": have, "missing": missing}
        return gap

    def seniority_estimate(self, skills):
        c = len(skills)
        if c >= 15:
            return "Advanced"
        if c >= 8:
            return "Intermediate"
        return "Beginner"

    def compute_role_match(self, skills, role):
        mapping = ROLE_SKILLS[role]
        score, total = 0, 0
        for pri, items in mapping.items():
            w = 3 if pri == "critical" else 2 if pri == "important" else 1
            for s in items:
                total += w
                if s in skills:
                    score += w
        return int((score / total) * 100)

    def recommend_project(self, role):
        return PROJECT_LIBRARY.get(role, ["Build a portfolio project"])[0]

    def alternatives(self, skills, exclude):
        scores = []
        for role in ROLE_SKILLS:
            if role == exclude:
                continue
            m = self.compute_role_match(skills, role)
            scores.append((role, m))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:3]

    def learning_path(self, missing_skills):
        roadmap = []
        for s in missing_skills[:7]:
            info = LEARNING_RESOURCES.get(s, GENERIC_RESOURCE)
            roadmap.append({
                "skill": s,
                "courses": info.get("courses", GENERIC_RESOURCE["courses"]),
                "projects": info.get("projects", GENERIC_RESOURCE["projects"]),
                "duration": info.get("duration", GENERIC_RESOURCE["duration"]),
                "difficulty": info.get("difficulty", GENERIC_RESOURCE["difficulty"])
            })
        return roadmap

    def estimate_effort(self, missing_count):
        if missing_count == 0:
            return "Ready to apply immediately"
        elif missing_count <= 2:
            return "Low (1–2 months)"
        elif missing_count <= 5:
            return "Moderate (2–4 months)"
        elif missing_count <= 8:
            return "Significant (4–6 months)"
        else:
            return "High (6–12 months)"
