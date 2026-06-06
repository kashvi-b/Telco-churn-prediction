# ============================================================
# STEP 6: Streamlit Dashboard (Portfolio Showpiece)
# ============================================================
# pip install streamlit plotly shap joblib pandas
# Run with: streamlit run src/dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import shap

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Churn Risk Dashboard",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Telco Customer Churn Risk Dashboard")
st.caption("Powered by XGBoost + SHAP — IBM Telco Dataset")

# ── Load model & data ──────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('models/best_model.pkl')

@st.cache_data
def load_test_data():
    X = pd.read_csv('data/X_test.csv')
    y = pd.read_csv('data/y_test.csv').squeeze()
    return X, y

model      = load_model()
X_test, y_test = load_test_data()
y_proba    = model.predict_proba(X_test)[:, 1]

# ── Sidebar: threshold slider ──────────────────────────────
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Risk threshold", 0.2, 0.8, 0.4, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("**Risk tiers**")
st.sidebar.markdown("🔴 HIGH: ≥ 70%")
st.sidebar.markdown("🟡 MEDIUM: 40–70%")
st.sidebar.markdown("🟢 LOW: < 40%")

# ── KPI metrics ────────────────────────────────────────────
y_pred    = (y_proba >= threshold).astype(int)
flagged   = y_pred.sum()
high_risk = (y_proba >= 0.70).sum()
from sklearn.metrics import roc_auc_score, f1_score
auc_score = roc_auc_score(y_test, y_proba)
f1        = f1_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC",         f"{auc_score:.3f}", "↑ Target: 0.85+")
col2.metric("F1 Score",        f"{f1:.3f}",        f"threshold={threshold}")
col3.metric("Customers flagged", f"{flagged}",      f"of {len(y_pred)} total")
col4.metric("High-risk (≥70%)", f"{high_risk}",    "immediate action")

st.markdown("---")

# ── Score distribution ─────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Churn probability distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=y_proba[y_test == 0], name='Retained',
        marker_color='#4A90D9', opacity=0.7, nbinsx=40
    ))
    fig.add_trace(go.Histogram(
        x=y_proba[y_test == 1], name='Churned',
        marker_color='#E05C5C', opacity=0.7, nbinsx=40
    ))
    fig.add_vline(x=threshold, line_dash="dash",
                  line_color="orange", annotation_text=f"threshold={threshold}")
    fig.update_layout(barmode='overlay', height=320,
                      xaxis_title="Churn probability",
                      yaxis_title="Count", legend_title="Actual")
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Risk tier breakdown")
    tiers = pd.cut(y_proba, bins=[0, 0.4, 0.7, 1.0],
                   labels=['Low', 'Medium', 'High'])
    tier_counts = tiers.value_counts().reset_index()
    tier_counts.columns = ['Tier', 'Count']
    fig2 = px.pie(tier_counts, values='Count', names='Tier',
                  color='Tier',
                  color_discrete_map={'Low': '#5CB85C',
                                      'Medium': '#F0AD4E',
                                      'High': '#D9534F'})
    fig2.update_layout(height=320)
    st.plotly_chart(fig2, use_container_width=True)

# ── SHAP feature importance ────────────────────────────────
# ── Feature Importance ────────────────────────────────
st.markdown("---")
st.subheader("🔍 Feature Importance")

st.info(
    "Feature importance visualization is disabled in the cloud deployment version."
)

# ── High-risk customer table ───────────────────────────────
st.markdown("---")
st.subheader("🔴 Top high-risk customers")

top_n = st.slider("Show top N customers", 5, 50, 20)
results = X_test.copy()
results['churn_probability'] = y_proba
results['actual_churn']      = y_test.values
results['risk_tier']         = pd.cut(
    y_proba, bins=[0, 0.4, 0.7, 1.0],
    labels=['Low', 'Medium', 'High']
)

top_risk = (results
            .sort_values('churn_probability', ascending=False)
            .head(top_n)[['tenure', 'MonthlyCharges', 'is_monthly',
                           'no_support', 'service_count',
                           'churn_probability', 'risk_tier', 'actual_churn']])

def colour_risk(val):
    if val == 'High':   return 'background-color: #ffe0e0'
    if val == 'Medium': return 'background-color: #fff5cc'
    return ''

st.dataframe(
    top_risk.style
        .format({'churn_probability': '{:.2%}'})
        .applymap(colour_risk, subset=['risk_tier']),
    use_container_width=True
)

# ── Individual prediction explorer ────────────────────────
st.markdown("---")
st.subheader("🧪 Individual customer explorer")
idx = st.number_input("Customer index (0 to N-1)",
                       0, len(X_test)-1, 0, 1)
cust = X_test.iloc[idx]
prob = y_proba[idx]
actual = y_test.iloc[idx]

c1, c2, c3 = st.columns(3)
c1.metric("Churn probability", f"{prob:.1%}")
c2.metric("Predicted",  "Churn ⚠️" if prob >= threshold else "Retain ✅")
c3.metric("Actual",     "Churned 🔴" if actual == 1 else "Retained 🟢")

with st.expander("Show all features for this customer"):
    st.dataframe(cust.to_frame().T, use_container_width=True)
