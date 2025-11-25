# Machine Learning Career Prediction Project — Full Technical Report

## 1. Project Overview
This project is a **career prediction and explanation system** built using:
- Scikit-learn pipelines  
- XGBoost classifier  
- TF‑IDF & OneHotEncoding preprocessing  
- SHAP explanations  
- A custom Skills Engine for skill‑gap analysis  
- Structured reporting for insights and recommendations  

The goal is to predict a **target job role** for a user based on academic, skill, and profile data, and then generate a detailed explanation including:
- Why the model predicted that role  
- The user’s current skills  
- Missing skills  
- A learning roadmap  
- Alternative suitable career paths  

---

## 2. Project Folder Structure

```
project/
├── data/
│   └── data.csv
├── models/
│   ├── final_model.joblib
│   ├── tuned_model.joblib
│   └── label_encoder.joblib
├── reports/
│   └── training_report.txt
├── notebooks/
├── src/
│   ├── pipeline.py
│   ├── train.py
│   ├── explain.py
│   ├── skills_engine.py
│   └── api.py
├── requirements.txt
├── Procfile
└── railway.json
```

---

## 3. Data Processing Pipeline (`pipeline.py`)
- Drops irrelevant columns  
- Numeric: median impute + StandardScaler  
- Categorical: impute "Missing" + OneHotEncoder(min_frequency=5)  
- Text: TF‑IDF (max_features=500, bigrams enabled)  
- Combined using ColumnTransformer  
- Produces ~1500–3000 features  
- Saved inside final model pipeline  

---

## 4. Model Training (`train.py`)
- Loads and cleans dataset  
- Encode target using LabelEncoder  
- Stratified split (80/20)  
- XGBoost classifier with tuned hyperparameters  
- Pipeline = preprocessing + model  
- Saves:
  - `final_model.joblib`
  - `label_encoder.joblib`
  - training report  

---

## 5. Skills Engine (`skills_engine.py`)
- Extracts clean skills from text  
- Garbage filtering  
- Alias mapping (`js` → `javascript`)  
- Fuzzy matching using RapidFuzz  
- Computes:
  - Skill gaps  
  - Alternative career matches  
  - Match scores  
  - Learning roadmap  
  - Effort estimation  
- Supports 8 predefined career roles  

---

## 6. Explanation Engine (`explain.py`)
Generates a full structured explanation:
- Predicted role + probability  
- SHAP feature importance analysis  
- Cleaned user skills  
- Skill gap analysis  
- Learning roadmap  
- Alternative careers  
- Readiness + strengths summary  

Uses KernelExplainer → slow but model‑agnostic.

---

## 7. Strengths
- Production-ready ML pipeline  
- Reproducible training  
- Skills intelligence + rule-based logic  
- SHAP explainability  
- Detailed JSON outputs  
- Modular and clean architecture  

---

## 8. Limitations
- SHAP KernelExplainer is slow  
- Skills Engine role catalog fixed to 8 roles  
- High feature dimensionality (TF‑IDF + OHE)  
- Garbage filtering requires periodic updates  

---

## 9. Summary
A complete ML + NLP + Explainability system for career recommendations with:
- Predictive accuracy  
- Skill reasoning  
- Roadmap generation  
- Alternative career paths  

Clean, modular, and ready for integration into apps or dashboards.
