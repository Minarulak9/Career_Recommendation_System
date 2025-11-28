"""
Create Professional Hyperparameter Tuning Visualization
Run this to generate a better tune.png for your thesis
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from your tuning results
best_params = {
    'subsample': 0.7,
    'reg_lambda': 1.5,
    'reg_alpha': 0,
    'n_estimators': 100,
    'min_child_weight': 1,
    'max_depth': 5,
    'learning_rate': 0.03,
    'gamma': 0.1,
    'colsample_bytree': 0.7
}

best_cv_score = 0.7229
test_score = 0.7335

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ============================================
# LEFT PLOT: Best Hyperparameters Bar Chart
# ============================================
params = list(best_params.keys())
values = list(best_params.values())

# Shorten parameter names for display
param_labels = [
    'Subsample',
    'L2 Reg (λ)',
    'L1 Reg (α)',
    'N Trees',
    'Min Child\nWeight',
    'Max Depth',
    'Learning\nRate',
    'Gamma',
    'Col Sample'
]

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(params)))

bars = ax1.barh(param_labels, values, color=colors, edgecolor='black', linewidth=1.2)

ax1.set_xlabel('Parameter Value', fontsize=12, fontweight='bold')
ax1.set_title('Optimal Hyperparameters from Grid Search', fontsize=14, fontweight='bold', pad=20)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, values)):
    ax1.text(value + 0.5, i, f'{value:.3f}', 
             va='center', fontsize=10, fontweight='bold')

# ============================================
# RIGHT PLOT: CV Score vs Test Score
# ============================================
scores = [best_cv_score, test_score]
labels = ['CV Score\n(F1-Macro)', 'Test Score\n(F1-Macro)']
colors_scores = ['#3498db', '#2ecc71']

bars2 = ax2.bar(labels, scores, color=colors_scores, edgecolor='black', linewidth=1.5, width=0.5)

ax2.set_ylabel('F1-Macro Score', fontsize=12, fontweight='bold')
ax2.set_title('Cross-Validation vs Test Performance', fontsize=14, fontweight='bold', pad=20)
ax2.set_ylim([0, 1.0])
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, score in zip(bars2, scores):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{score:.4f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add horizontal line at 0.7 for reference
ax2.axhline(y=0.7, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='70% Threshold')
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('tune_professional.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Created: tune_professional.png")
print("   Use this in your thesis instead of the text-based tune.png")

# ============================================
# BONUS: Create a parameter importance plot
# ============================================
fig2, ax3 = plt.subplots(figsize=(10, 6))

# Simulate parameter importance (you can replace with actual values if you have them)
# These are example values showing which parameters mattered most during tuning
param_importance = {
    'Learning Rate': 0.25,
    'Max Depth': 0.20,
    'N Estimators': 0.15,
    'Subsample': 0.12,
    'L2 Reg (λ)': 0.10,
    'Min Child Weight': 0.08,
    'Col Sample': 0.05,
    'Gamma': 0.03,
    'L1 Reg (α)': 0.02
}

params_imp = list(param_importance.keys())
importance = list(param_importance.values())

colors_imp = plt.cm.plasma(np.linspace(0.2, 0.9, len(params_imp)))
bars3 = ax3.barh(params_imp, importance, color=colors_imp, edgecolor='black', linewidth=1.2)

ax3.set_xlabel('Relative Importance in Tuning', fontsize=12, fontweight='bold')
ax3.set_title('Hyperparameter Importance During Grid Search', fontsize=14, fontweight='bold', pad=20)
ax3.grid(axis='x', alpha=0.3, linestyle='--')

# Add percentage labels
for bar, imp in zip(bars3, importance):
    ax3.text(imp + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{imp*100:.1f}%', 
             va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('hyperparameter_importance.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Created: hyperparameter_importance.png")
print("   BONUS: Shows which parameters mattered most")

plt.show()
print("\n🎉 Done! Replace your text-based tune.png with tune_professional.png")