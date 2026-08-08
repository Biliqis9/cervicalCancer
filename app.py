"""
app.py
------
Main entrypoint for the Cervical Cancer Risk Predictor Streamlit app.
This file renders the Home page. All other pages live in pages/ and
are picked up automatically by Streamlit's multipage navigation.
"""

import logging

import streamlit as st

from components.cards import card_close, card_open, section_pill
from components.footer import render_footer
from components.sidebar import render_sidebar_brand
from components.styling import inject_css

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Cervical Cancer Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
render_sidebar_brand()

# ------------------------------------------------------------------
# Hero
# ------------------------------------------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    section_pill("AI-Powered Clinical Decision Support")
    st.title("Cervical Cancer Risk Predictor")
    st.markdown(
        """
        A machine learning tool that estimates cervical cancer risk from
        demographic, behavioral, and clinical risk factors -- built to
        support (not replace) clinical judgment for healthcare
        professionals working in screening and early-detection programs.
        """
    )
    st.markdown(
        """
        Use the sidebar to navigate to **Predict Cervical Cancer Risk** to
        run an assessment, or explore the model's performance and
        explainability on the other pages.
        """
    )
with col2:
    card_open()
    st.markdown("#### At a Glance")
    st.markdown(
        """
        - **Model:** XGBoost Classifier
        - **Preprocessing:** MICE (Iterative) Imputation
        - **Explainability:** SHAP
        - **Inputs:** 15 risk factors
        """
    )
    card_close()

st.divider()

# ------------------------------------------------------------------
# Feature highlights
# ------------------------------------------------------------------
st.markdown("### What You Can Do Here")
c1, c2, c3 = st.columns(3)

with c1:
    card_open()
    st.markdown("#### 🩺 Predict Risk")
    st.write(
        "Fill in a patient's demographic, behavioral, and clinical "
        "information to get an instant, color-coded risk estimate with "
        "a plain-language explanation."
    )
    card_close()

with c2:
    card_open()
    st.markdown("#### 🔍 Understand Why")
    st.write(
        "SHAP-based explainability shows exactly which factors pushed "
        "a prediction higher or lower, for a single patient or across "
        "the model as a whole."
    )
    card_close()

with c3:
    card_open()
    st.markdown("#### 📊 Trust the Model")
    st.write(
        "Review accuracy, precision, recall, F1, ROC AUC, and confusion "
        "matrix on the Model Performance page to see how the model "
        "was validated."
    )
    card_close()

st.divider()

# ------------------------------------------------------------------
# Safety notice
# ------------------------------------------------------------------
card_open()
st.markdown("#### ⚠️ Important Notice")
st.write(
    "This application is a **decision-support tool**, not a diagnostic "
    "device. Predictions are estimates derived from statistical patterns "
    "in historical data and should always be interpreted alongside "
    "clinical judgment, physical examination, and standard screening "
    "protocols (e.g. Pap smear, HPV DNA testing, colposcopy)."
)
card_close()

render_footer()
