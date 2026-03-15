# ============================================================
# STEP 4: Evaluation + SHAP Explainability
# ============================================================
# pip install shap matplotlib scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                              confusion_matrix, ConfusionMatrixDisplay)

# ── Load model and data ────────────────────────────────────
model   = joblib.load('models/best_model.pkl')
X_train = pd.read_csv('data/X_train.csv')
X_test  = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test  = pd.read_csv('data/y_test.csv').squeeze()

y_proba = model.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= 0.4).astype(int)

# ── ROC Curve ─────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc     = auc(fpr, tpr)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(fpr, tpr, color='#4A90D9', lw=2, label=f'AUC = {roc_auc:.3f}')
axes[0].plot([0,1], [0,1], 'k--', lw=1)
axes[0].fill_between(fpr, tpr, alpha=0.1, color='#4A90D9')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()

# ── Precision-Recall Curve ─────────────────────────────────
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)

axes[1].plot(recall, precision, color='#E05C5C', lw=2, label=f'PR AUC = {pr_auc:.3f}')
axes[1].fill_between(recall, precision, alpha=0.1, color='#E05C5C')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()

plt.tight_layout()
plt.savefig('reports/evaluation_curves.png', dpi=150)
plt.show()

# ── Confusion Matrix ───────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Retained', 'Churned'])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title('Confusion Matrix (threshold=0.4)')
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png', dpi=150)
plt.show()

# ── Threshold Analysis (business cost framing) ────────────
print("\n📊 Threshold Analysis:")
print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<10} {'Flagged'}")
for t in [0.3, 0.35, 0.4, 0.45, 0.5]:
    p_ = (y_proba >= t).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score
    print(f"{t:<12.2f} {precision_score(y_test,p_):<12.3f} "
          f"{recall_score(y_test,p_):<12.3f} "
          f"{f1_score(y_test,p_):<10.3f} "
          f"{p_.sum()} customers")

# ── SHAP Explainability ────────────────────────────────────
print("\n🔍 Computing SHAP values (this takes ~30s)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# If model returns 2D shap (binary), take class 1
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

# Global feature importance (bar plot)
plt.figure(figsize=(10, 6))
shap.summary_plot(sv, X_test, plot_type='bar', show=False,
                  max_display=15, color='#4A90D9')
plt.title('Top 15 Features — SHAP Importance')
plt.tight_layout()
plt.savefig('reports/shap_bar.png', dpi=150, bbox_inches='tight')
plt.show()

# Beeswarm plot (shows direction of effect)
plt.figure(figsize=(10, 8))
shap.summary_plot(sv, X_test, show=False, max_display=15)
plt.title('SHAP Beeswarm — Feature Impact Direction')
plt.tight_layout()
plt.savefig('reports/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Individual prediction explanation ─────────────────────
# Pick a high-risk customer and explain why
idx = np.argmax(y_proba)   # highest predicted churn probability
print(f"\n🔴 Highest-risk customer (index {idx}):")
print(f"   Churn probability: {y_proba[idx]:.3f}")
print(f"   Actual churn:      {y_test.iloc[idx]}")

# Force plot (save as HTML for GitHub/portfolio)
shap.initjs()
force_plot = shap.force_plot(
    explainer.expected_value if not isinstance(explainer.expected_value, list)
    else explainer.expected_value[1],
    sv[idx], X_test.iloc[idx]
)
shap.save_html('reports/shap_force_plot.html', force_plot)
print("   Force plot saved to /reports/shap_force_plot.html")

print("\n✅ Evaluation complete. All plots saved to /reports/")
