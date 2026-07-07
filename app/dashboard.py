"""
Streamlit dashboard for the churn model - single-customer scoring with
SHAP explainability, plus batch scoring from an uploaded CSV.

Run:
    streamlit run app/dashboard.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import shap
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_processing import build_feature_frame  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

st.set_page_config(page_title="Churn Risk Dashboard", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "churn_model.pkl")
    feature_cols = json.loads((MODELS_DIR / "feature_list.json").read_text())
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
    return model, feature_cols, metrics


st.title("📉 Customer Churn Risk Dashboard")

if not (MODELS_DIR / "churn_model.pkl").exists():
    st.warning(
        "No trained model found yet. Run `python src/train.py` first, then reload this page."
    )
    st.stop()

model, feature_cols, metrics = load_artifacts()

with st.sidebar:
    st.header("Model info")
    st.metric("Best model", metrics["best_model"])
    st.metric("ROC-AUC (test)", metrics["candidates"][metrics["best_model"]]["roc_auc"])
    st.caption("Metrics computed on held-out test set during training.")

tab_single, tab_batch = st.tabs(["🔍 Score one customer", "📄 Batch scoring (CSV)"])

with tab_single:
    st.subheader("Customer profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        phone = st.selectbox("Phone service", ["Yes", "No"])

    with col3:
        monthly = st.number_input("Monthly charges ($)", 18.0, 130.0, 75.0)
        total = st.number_input("Total charges ($)", 0.0, 10000.0, float(monthly * tenure))
        tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
        online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

    customer = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": "No",
        "InternetService": internet, "OnlineSecurity": online_security, "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": "No", "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
    }])

    if st.button("Score customer", type="primary"):
        X, _ = build_feature_frame(customer, fit_columns=feature_cols)
        proba = float(model.predict_proba(X)[0, 1])

        risk = "🟢 Low" if proba < 0.3 else ("🟡 Medium" if proba < 0.6 else "🔴 High")
        st.metric("Churn probability", f"{proba:.1%}", delta=None)
        st.markdown(f"### Risk level: {risk}")

        st.subheader("Why? (SHAP explanation)")
        clf = model.named_steps["clf"]
        try:
            if hasattr(clf, "feature_importances_"):
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X)
                sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
            else:
                X_scaled = model.named_steps["scaler"].transform(X)
                explainer = shap.LinearExplainer(clf, X_scaled)
                sv = explainer.shap_values(X_scaled)[0]

            contrib = pd.Series(sv, index=feature_cols).sort_values(key=abs, ascending=False).head(10)
            st.bar_chart(contrib)
            st.caption("Top 10 features by |SHAP value| pushing this prediction up (churn) or down (stay).")
        except Exception as e:  # pragma: no cover - explainability is best-effort
            st.info(f"SHAP explanation unavailable for this model type: {e}")

with tab_batch:
    st.subheader("Score a batch of customers")
    st.caption("Upload a CSV with the same columns as data/raw/telco_churn.csv (customerID/Churn optional).")
    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        X, _ = build_feature_frame(raw, fit_columns=feature_cols)
        proba = model.predict_proba(X)[:, 1]

        out = raw.copy()
        out["churn_probability"] = proba.round(4)
        out["risk_label"] = pd.cut(
            proba, bins=[-0.01, 0.3, 0.6, 1.0], labels=["low", "medium", "high"]
        )
        out = out.sort_values("churn_probability", ascending=False)

        st.dataframe(out, use_container_width=True)
        st.download_button(
            "Download scored CSV",
            out.to_csv(index=False).encode("utf-8"),
            file_name="scored_customers.csv",
            mime="text/csv",
        )
