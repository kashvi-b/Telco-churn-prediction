# ============================================================
# STEP 1: Data Loading & Exploratory Data Analysis
# ============================================================
# Dataset: IBM Telco Customer Churn (Kaggle)
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Download WA_Fn-UseC_-Telco-Customer-Churn.csv and place in /data/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load Data ──────────────────────────────────────────────
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.shape)       # (7043, 21)
print(df.dtypes)
print(df.isnull().sum())

# ── Fix TotalCharges (loaded as string due to whitespace) ──
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# ── Encode target ──────────────────────────────────────────
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
print("Churn rate:", df['Churn'].mean().round(3))  # ~0.265

# ── Drop customerID (not a feature) ───────────────────────
df.drop(columns=['customerID'], inplace=True)

# ── Class distribution ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Churn balance
df['Churn'].value_counts().plot.pie(
    autopct='%1.1f%%', ax=axes[0],
    labels=['Retained', 'Churned'],
    colors=['#4A90D9', '#E05C5C']
)
axes[0].set_title('Churn Distribution')

# Churn by contract type
churn_by_contract = df.groupby('Contract')['Churn'].mean().reset_index()
sns.barplot(data=churn_by_contract, x='Contract', y='Churn', ax=axes[1], palette='Blues_d')
axes[1].set_title('Churn Rate by Contract Type')
axes[1].set_ylabel('Churn Rate')
plt.tight_layout()
plt.savefig('reports/eda_overview.png', dpi=150)
plt.show()

# ── Numerical feature distributions ───────────────────────
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(num_cols):
    sns.histplot(data=df, x=col, hue='Churn', bins=30,
                 ax=axes[i], palette={0: '#4A90D9', 1: '#E05C5C'})
    axes[i].set_title(f'{col} by Churn')
plt.tight_layout()
plt.savefig('reports/eda_distributions.png', dpi=150)
plt.show()

# ── Correlation heatmap ────────────────────────────────────
corr = df[num_cols + ['Churn']].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('reports/eda_correlation.png', dpi=150)
plt.show()

print("✅ EDA complete. Plots saved to /reports/")
