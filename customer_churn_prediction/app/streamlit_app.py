"""
Streamlit dashboard for Customer Churn Prediction.

Provides customer input form, predictions, risk indicators, SHAP explanations,
and downloadable prediction reports.
"""

import sys
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Add src to path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import ChurnPredictor  # noqa: E402

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .risk-low { color: #28a745; font-weight: bold; font-size: 1.5rem; }
    .risk-medium { color: #ffc107; font-weight: bold; font-size: 1.5rem; }
    .risk-high { color: #dc3545; font-weight: bold; font-size: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> ChurnPredictor:
    """Load and cache the churn predictor."""
    predictor = ChurnPredictor()
    try:
        predictor.load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Please run `python main.py` first to train the model."
        )
        st.stop()
    return predictor


def render_risk_indicator(risk_level: str, probability: float) -> None:
    """Display color-coded risk level indicator."""
    risk_class = {
        "Low": "risk-low",
        "Medium": "risk-medium",
        "High": "risk-high",
    }.get(risk_level, "risk-medium")

    st.markdown(
        f'<p class="{risk_class}">Risk Level: {risk_level} ({probability:.1%})</p>',
        unsafe_allow_html=True,
    )


def plot_shap_chart(shap_values: dict[str, float]) -> None:
    """Render SHAP feature contribution bar chart."""
    if not shap_values:
        st.info("SHAP explanations unavailable. Install shap package for feature contributions.")
        return

    top_features = dict(list(shap_values.items())[:10])
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E94F37" if v > 0 else "#2E86AB" for v in top_features.values()]
    ax.barh(list(top_features.keys())[::-1], list(top_features.values())[::-1], color=colors[::-1])
    ax.set_xlabel("SHAP Value (impact on churn probability)")
    ax.set_title("Feature Contribution to Churn Prediction")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    """Render the Streamlit dashboard."""
    st.markdown('<p class="main-header">AI-Powered Customer Churn Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Predict customer churn risk and identify key retention drivers</p>',
        unsafe_allow_html=True,
    )

    predictor = load_predictor()

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.subheader("Customer Information")
        with st.form("customer_form"):
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["Yes", "No", "No phone service"]
            )
            internet = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0, 1.0)

            submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    with col_result:
        st.subheader("Prediction Results")

        if submitted:
            customer_data = {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple_lines,
                "InternetService": internet,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }

            try:
                result = predictor.predict(customer_data)

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Churn Prediction", result.churn_prediction)
                with m2:
                    st.metric("Churn Probability", f"{result.churn_probability:.1%}")
                with m3:
                    st.metric("Risk Level", result.risk_level)

                render_risk_indicator(result.risk_level, result.churn_probability)

                st.progress(min(result.churn_probability, 1.0))

                st.markdown("---")
                st.subheader("Feature Importance (SHAP)")
                plot_shap_chart(result.shap_values or {})

                report = predictor.generate_report(result, customer_id="CUSTOM")
                st.download_button(
                    label="Download Prediction Report",
                    data=report,
                    file_name="churn_prediction_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
        else:
            st.info("Fill in the customer form and click **Predict Churn** to see results.")

    # Sidebar info
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            This dashboard uses machine learning to predict customer churn
            based on demographics, account information, and service usage.

            **Models used:** Logistic Regression, Random Forest,
            XGBoost, Gradient Boosting (best model selected automatically).

            **Explainability:** SHAP values show which features
            drive each prediction.
            """
        )
        st.markdown("---")
        st.markdown("**Risk Levels**")
        st.markdown("- **Low:** < 30% churn probability")
        st.markdown("- **Medium:** 30–60% churn probability")
        st.markdown("- **High:** > 60% churn probability")


if __name__ == "__main__":
    main()
