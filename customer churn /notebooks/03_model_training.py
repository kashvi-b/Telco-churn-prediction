# ============================================================
# STEP 3: Model Training + Experiment Tracking (MLflow)
# ============================================================
# pip install xgboost lightgbm mlflow imbalanced-learn scikit-learn

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, classification_report,
                              confusion_matrix)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ── Load data ──────────────────────────────────────────────
X_train = pd.read_csv('data/X_train.csv')
X_test  = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test  = pd.read_csv('data/y_test.csv').squeeze()

# ── Handle class imbalance with SMOTE ─────────────────────
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — 0: {(y_train_bal==0).sum()}, 1: {(y_train_bal==1).sum()}")

# ── Define models ──────────────────────────────────────────
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "RandomForest":       RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "GradientBoosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    "XGBoost":            XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                        use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM":           LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                         class_weight='balanced', random_state=42),
}

# ── Training loop with MLflow tracking ────────────────────
mlflow.set_experiment("telco_churn_prediction")

results = {}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Cross-validation on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_auc = cross_val_score(model, X_train_bal, y_train_bal,
                                 cv=cv, scoring='roc_auc').mean()

        # Train on full balanced training set
        model.fit(X_train_bal, y_train_bal)

        # Evaluate on held-out test set
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= 0.4).astype(int)   # threshold tuned for recall

        auc    = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        f1     = f1_score(y_test, y_pred)

        # Log to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({"cv_auc": cv_auc, "test_auc": auc,
                             "pr_auc": pr_auc, "f1": f1})
        mlflow.sklearn.log_model(model, artifact_path="model")

        results[name] = {"cv_auc": cv_auc, "test_auc": auc,
                          "pr_auc": pr_auc, "f1": f1}

        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"  CV AUC:   {cv_auc:.4f}")
        print(f"  Test AUC: {auc:.4f}")
        print(f"  PR AUC:   {pr_auc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred,
                                    target_names=['Retained', 'Churned']))

# ── Compare results ────────────────────────────────────────
results_df = pd.DataFrame(results).T.sort_values('test_auc', ascending=False)
print("\n📊 Model Comparison:")
print(results_df.round(4))

# ── Save best model ────────────────────────────────────────
best_model_name = results_df.index[0]
best_model = models[best_model_name]
joblib.dump(best_model, 'models/best_model.pkl')
print(f"\n✅ Best model: {best_model_name} saved to /models/best_model.pkl")

# View MLflow UI: run `mlflow ui` in terminal → http://localhost:5000
