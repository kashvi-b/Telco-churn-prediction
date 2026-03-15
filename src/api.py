# ============================================================
# STEP 5: Model Serving with FastAPI
# ============================================================
# pip install fastapi uvicorn pydantic joblib pandas
#
# Run with: uvicorn src.api:app --reload
# Test at:  http://localhost:8000/docs

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts customer churn probability for telecom customers.",
    version="1.0.0"
)

# ── Load model and scaler on startup ──────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'models/best_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'models/scaler.pkl')

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ── Request schema ─────────────────────────────────────────
class CustomerFeatures(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract: str           # "Month-to-month" | "One year" | "Two year"
    PaymentMethod: str      # "Electronic check" | "Mailed check" | etc.
    InternetService: str    # "DSL" | "Fiber optic" | "No"
    OnlineSecurity: str     # "Yes" | "No" | "No internet service"
    TechSupport: str        # "Yes" | "No" | "No internet service"
    PaperlessBilling: str   # "Yes" | "No"
    Partner: str            # "Yes" | "No"
    Dependents: str         # "Yes" | "No"
    service_count: int      # count of active services (0-9)

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 5,
                "MonthlyCharges": 85.0,
                "TotalCharges": 425.0,
                "Contract": "Month-to-month",
                "PaymentMethod": "Electronic check",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "TechSupport": "No",
                "PaperlessBilling": "Yes",
                "Partner": "No",
                "Dependents": "No",
                "service_count": 3
            }
        }

def preprocess_input(data: CustomerFeatures) -> pd.DataFrame:
    """Convert input to model-ready feature vector."""
    row = data.dict()

    # Engineered features
    row['charges_per_month'] = row['TotalCharges'] / (row['tenure'] + 1)
    row['is_monthly']        = int(row['Contract'] == 'Month-to-month')
    row['no_support']        = int(row['OnlineSecurity'] == 'No' and
                                   row['TechSupport'] == 'No')
    row['payment_friction']  = int(row['PaperlessBilling'] == 'Yes' and
                                   row['PaymentMethod'] == 'Electronic check')

    # Binary encode
    row['Partner']          = int(row['Partner'] == 'Yes')
    row['Dependents']       = int(row['Dependents'] == 'Yes')
    row['PaperlessBilling'] = int(row['PaperlessBilling'] == 'Yes')

    df = pd.DataFrame([row])

    # One-hot encode categoricals to match training schema
    cat_cols = ['Contract', 'PaymentMethod', 'InternetService',
                'OnlineSecurity', 'TechSupport']
    df = pd.get_dummies(df, columns=cat_cols)

    # Align with training columns (add missing, drop extra)
    training_cols = model.feature_names_in_   # XGBoost/LGBM store this
    for col in training_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[training_cols]

    # Scale numerical
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_month']
    df[num_features] = scaler.transform(df[num_features])

    return df

# ── Endpoints ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Telco Churn Prediction API is running 🚀"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": type(model).__name__}

@app.post("/predict")
def predict(customer: CustomerFeatures):
    try:
        X = preprocess_input(customer)
        churn_proba  = float(model.predict_proba(X)[0][1])
        churn_flag   = churn_proba >= 0.4

        # Risk tier
        if churn_proba >= 0.70:
            risk_tier = "HIGH"
            action    = "Immediate outreach — offer retention discount"
        elif churn_proba >= 0.40:
            risk_tier = "MEDIUM"
            action    = "Automated email campaign with soft incentive"
        else:
            risk_tier = "LOW"
            action    = "No action needed"

        return {
            "churn_probability": round(churn_proba, 4),
            "will_churn":        bool(churn_flag),
            "risk_tier":         risk_tier,
            "recommended_action": action
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(customers: list[CustomerFeatures]):
    """Score a list of customers at once."""
    results = []
    for customer in customers:
        result = predict(customer)
        results.append(result)
    return {"predictions": results, "count": len(results)}
