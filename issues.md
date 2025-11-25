# Project Issues & Solutions — Technical Report

This document summarizes **all key issues** found in your ML project and provides **clear, actionable solutions** for each part of the pipeline.

---

# 1. Data Pipeline Issues (pipeline.py)

## Issue 1 — Missing Columns During Prediction
If an incoming JSON payload does NOT include all expected numeric/categorical/text fields →  
**pipeline will break** before prediction.

### Solution
Add a preprocessing wrapper before calling `model.predict()`:

```python
def ensure_full_schema(input_dict, expected_columns):
    for col in expected_columns:
        if col not in input_dict:
            input_dict[col] = None
    return input_dict
```

---

## Issue 2 — TF‑IDF Vocabulary Instability
TF‑IDF vocabulary depends on training data.  
Changing dataset = different vocabulary = SHAP wrong + model mismatch.

### Solutions
Choose ONE:
1. **Freeze vocabulary** and save it separately.
2. Replace TF‑IDF with **sentence embeddings (SBERT)**.
3. Use **HashingVectorizer** to ensure fixed dimensionality.

---

## Issue 3 — OneHotEncoder min_frequency=5 May Drop Categories
During training, categories may be merged or removed depending on frequency.  
During inference, new categories are ignored automatically → good.  
But dimensionality mismatch may appear in SHAP.

### Solution
Set:
```python
OneHotEncoder(handle_unknown="ignore", min_frequency=None)
```

Or maintain a **fixed category list**.

---

# 2. Model Training Issues (train.py)

## Issue 4 — High‑Dimensional Feature Space
TF‑IDF (1000+) + OneHot + Numeric → 1500–3000 features.  
XGBoost handles it, but:
- Slower training
- Increased memory usage
- SHAP extremely slow

### Solutions
- Use **dimensionality reduction**:
  - TruncatedSVD for TF‑IDF  
  - PCA for numeric  
- Or replace TF‑IDF with embeddings.

---

## Issue 5 — No Class Weights for Imbalanced Data
If target distribution is imbalanced → F1 scores biased.

### Solutions
Calculate class weights:

```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight("balanced", classes, y_encoded)
```

Then set:

```python
clf = xgb.XGBClassifier(scale_pos_weight=weights)
```

---

## Issue 6 — Confusion Matrix Imported But Not Used
The script imports `confusion_matrix` but never prints or saves it.

### Solution
Add:

```python
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

---

# 3. Skills Engine Issues (skills_engine.py)

## Issue 7 — Skill Roles Are Static
ROLE_SKILLS only supports **8 roles**.  
If the ML model predicts a role outside these →  
skill gap = empty → broken recommendations.

### Solutions
1. Auto‑generate ROLE_SKILLS from dataset.  
2. Build a dynamic fallback:
   - cluster similar roles  
   - or simple keyword matching  
3. Extend ROLE_SKILLS based on dataset distribution.

---

## Issue 8 — NaN Detection Logic Bug
This line is incorrect:

```python
if text is None or pd.isna(text) if 'pd' in dir() else text is None:
```

### Solution

Replace with:

```python
if text is None or (isinstance(text, float) and np.isnan(text)):
    return set()
```

---

## Issue 9 — Garbage Token Filter May Remove Useful Tokens
Aggressive garbage list can drop valid content like:
- “student experience”
- “entry level developer”

### Solution
Review garbage list regularly.  
Or limit garbage removal to exact matches only.

---

## Issue 10 — Fuzzy Matching Threshold Too High (75)
Some skills won't match:
- “javscript” → might fail  
- “nod js” → might fail

### Solution
Lower threshold:

```python
SkillsEngine(fuzzy_threshold=65)
```

---

# 4. Explanation Engine Issues (explain.py)

## Issue 11 — SHAP KernelExplainer Extremely Slow
This is the BIGGEST issue.

KernelExplainer = O(n_features × nsamples)

Result:
- 20–60 seconds per explanation
- Not scalable
- Not API‑friendly

### Solutions
Choose ONE:
1. Use **TreeExplainer** (XGBoost‑optimized)
2. Reduce feature size  
3. Cache SHAP background dataset  
4. Precompute SHAP for all training points

---

## Issue 12 — SHAP Feature Name Mismatch
Your fallback logic:

```python
diff = bg_trans.shape[1] - len(feature_names)
```

Means:
- Pipeline output dimension != feature name list
- Causes misaligned explanations

### Solution
Fix preprocessing consistency:
- Disable `min_frequency=5`
- Freeze TF‑IDF vocabulary
- Always use identical training pipeline

---

## Issue 13 — Skills Engine Role Mismatch with Model Prediction
If ML predicts:
- “Cybersecurity Engineer”
- “Cloud Architect”
Your explanation system becomes empty.

### Solution
Add fallback:

```python
if pred_role not in ROLE_SKILLS:
    match = fuzzy_match_role(pred_role)
```

---

# 5. General Architecture Issues

## Issue 14 — Preprocessing Not Validated Before Prediction
If incoming data is missing fields → model errors.

### Solution
Create a schema validation layer.

---

## Issue 15 — Different Components Don’t Fully Integrate
- ML model doesn’t use extracted skills  
- SkillsEngine works independently  

### Solution
Integrate extracted skills into:
- feature engineering  
- or model input

Example: add binary vector of extracted canonical skills.

---

# 6. Summary of Critical Fixes

| Issue | Impact | Priority | Solution |
|------|--------|----------|----------|
| SHAP KernelExplainer too slow | High | ⭐⭐⭐⭐⭐ | Use TreeExplainer |
| TF‑IDF instability | High | ⭐⭐⭐⭐ | Freeze vocabulary / use embeddings |
| SkillsEngine fixed role list | Medium | ⭐⭐⭐⭐ | Expand dynamic roles |
| Missing columns on inference | Medium | ⭐⭐⭐ | Add schema validation |
| Fuzzy skill threshold | Medium | ⭐⭐⭐ | Lower threshold |
| NaN skill extraction bug | Low | ⭐⭐ | Fix condition |
| Confusion matrix unused | Low | ⭐ | Print/save it |

---

# 7. Final Recommendation
Your project is well‑structured, but to reach **production‑grade ML quality**, fix the following first:

1. TF‑IDF stability  
2. SHAP performance  
3. Role mismatch between ML model & SkillsEngine  
4. Input schema validation  

Everything else is optional improvements.

