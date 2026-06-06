# Customer Churn Prediction
## IBM Telco Dataset | XGBoost + SHAP + FastAPI + Streamlit

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green)](https://xgboost.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-orange)](https://mlflow.org)

---

## Project Summary

End-to-end churn prediction pipeline for a telecom company using the IBM Telco dataset.
Achieved **0.87 AUC-ROC** with XGBoost. Includes full SHAP explainability,
a Streamlit business dashboard, and a FastAPI serving endpoint.

## Results

| Model             | CV AUC | Test AUC | PR AUC | F1    |
|-------------------|--------|----------|--------|-------|
| LogisticRegression| 0.836  | 0.841    | 0.631  | 0.597 |
| RandomForest      | 0.861  | 0.858    | 0.672  | 0.618 |
| GradientBoosting  | 0.867  | 0.863    | 0.681  | 0.625 |
| LightGBM          | 0.873  | 0.869    | 0.693  | 0.631 |
| **XGBoost**       | **0.878** | **0.874** | **0.702** | **0.643** |

---

## Project Structure

```
churn_project/
├── data/                   # Raw + processed data (not committed)
├── models/                 # Saved model & scaler
├── notebooks/
│   ├── 01_eda.py           # Exploratory data analysis
│   ├── 02_feature_engineering.py
│   ├── 03_model_training.py  # MLflow experiment tracking
│   └── 04_evaluation_shap.py # SHAP explainability
├── reports/                # Saved plots & HTML reports
├── src/
│   ├── api.py              # FastAPI serving endpoint
│   └── dashboard.py        # Streamlit business dashboard
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/yourname/telco-churn-prediction
cd telco-churn-prediction
pip install -r requirements.txt

# 2. Download dataset
# → https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# → Place CSV in /data/

# 3. Run notebooks in order
python notebooks/01_eda.py
python notebooks/02_feature_engineering.py
python notebooks/03_model_training.py
python notebooks/04_evaluation_shap.py

# 4. Launch dashboard
streamlit run src/dashboard.py

# 5. Launch API
uvicorn src.api:app --reload
# → API docs at http://localhost:8000/docs
```

---

## Key Technical Highlights

- **Class imbalance**: Handled via SMOTE (26.5% → 50% balanced)
- **Feature engineering**: RFM-inspired features, service bundles, contract risk flags
- **Explainability**: SHAP TreeExplainer with global + per-customer force plots
- **Experiment tracking**: MLflow with parameter logging and model registry
- **Threshold tuning**: Threshold set at 0.40 (prioritising recall over precision)
- **Deployment**: FastAPI with Pydantic validation + batch scoring endpoint
- **Dashboard**: Streamlit with Plotly charts, SHAP bar chart, risk tier breakdown

---

## Business Impact Framing

> Targeting the top 20% of highest-scored customers captures **~65% of all actual churners**,
> allowing the retention team to prioritise outreach efficiently rather than contacting all 7,000 customers.

| Risk Tier | Score Range | Customers | Recommended Action              |
|-----------|-------------|-----------|----------------------------------|
| 🔴 HIGH   | ≥ 70%       | ~180      | Immediate call + retention offer |
| 🟡 MEDIUM | 40–70%      | ~520      | Automated email campaign         |
| 🟢 LOW    | < 40%       | ~706      | No action needed                 |
