# ============================================================
# STEP 2: Feature Engineering & Preprocessing
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# ── Basic cleaning (same as Step 1) ───────────────────────
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
df.drop(columns=['customerID'], inplace=True)

# ── Feature Engineering ────────────────────────────────────

# 1. Tenure bands (loyalty segments)
df['tenure_band'] = pd.cut(
    df['tenure'],
    bins=[0, 12, 24, 48, 72],
    labels=['new', 'developing', 'established', 'loyal']
)

# 2. Avg monthly spend trend proxy
df['charges_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)

# 3. Service count (how many services subscribed to)
service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
df['service_count'] = (df[service_cols]
                       .apply(lambda x: x.isin(['Yes', 'Fiber optic', 'DSL']))
                       .sum(axis=1))

# 4. Is month-to-month contract (highest risk flag)
df['is_monthly'] = (df['Contract'] == 'Month-to-month').astype(int)

# 5. Has no online security AND no tech support (risk combo)
df['no_support'] = (
    (df['OnlineSecurity'] == 'No') &
    (df['TechSupport'] == 'No')
).astype(int)

# 6. Paperless + electronic check (payment friction signal)
df['payment_friction'] = (
    (df['PaperlessBilling'] == 'Yes') &
    (df['PaymentMethod'] == 'Electronic check')
).astype(int)

print("New features:", ['tenure_band', 'charges_per_month', 'service_count',
                         'is_monthly', 'no_support', 'payment_friction'])

# ── Encode categoricals ────────────────────────────────────
# Binary Yes/No columns
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
               'PaperlessBilling']
for col in binary_cols:
    df[col] = (df[col] == 'Yes').astype(int)

# Drop 'gender' — generally not a good feature ethically and adds little signal
df.drop(columns=['gender'], inplace=True)

# Multi-class categoricals → one-hot encode
cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract',
            'PaymentMethod', 'tenure_band']

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ── Train/Test Split ───────────────────────────────────────
X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Churn rate train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

# ── Scale numerical features ───────────────────────────────
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'charges_per_month']
scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features]  = scaler.transform(X_test[num_features])

# ── Save processed data ────────────────────────────────────
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv',   index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv',   index=False)
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ Feature engineering complete. Data saved.")
