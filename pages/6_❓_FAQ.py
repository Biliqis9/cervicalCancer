"""FAQ page."""

import streamlit as st

from components.footer import render_footer
from components.sidebar import render_sidebar_brand
from components.styling import inject_css
from components.cards import section_pill

st.set_page_config(page_title="FAQ", page_icon="❓", layout="wide")
inject_css()
render_sidebar_brand()

section_pill("Help")
st.title("Frequently Asked Questions")

faqs = [
    (
        "Is this a diagnostic tool?",
        "No. This tool provides a statistical risk estimate to support "
        "clinical decision-making. It cannot diagnose cervical cancer "
        "and does not replace standard screening (Pap smear, HPV "
        "testing, colposcopy, biopsy).",
    ),
    (
        "How was the model trained?",
        "The model is an XGBoost classifier trained on demographic, "
        "behavioral, and clinical risk factors, with missing values "
        "handled via MICE (Multiple Imputation by Chained Equations) "
        "and class imbalance addressed during training. See the "
        "**About the Project** page for the full pipeline.",
    ),
    (
        "What does 'Risk Level' mean?",
        "Risk Level (Low / Moderate / High) is a simplified, "
        "color-coded translation of the model's predicted probability "
        "relative to its decision threshold, meant to be quickly "
        "readable by non-technical staff.",
    ),
    (
        "Why does the model use a custom threshold instead of 50%?",
        "For screening-style use cases, missing a true positive is "
        "usually costlier than a false alarm, so the threshold is "
        "tuned (rather than defaulted to 0.5) to better balance "
        "sensitivity and specificity for this purpose.",
    ),
    (
        "Is my data stored anywhere?",
        "No. Inputs entered on the Predict page are used only in your "
        "browser session to generate a prediction and are not saved to "
        "a database or shared externally.",
    ),
    (
        "What do the SHAP plots mean?",
        "SHAP values show how much each input pushed a specific "
        "prediction up or down relative to the model's average output. "
        "See the **Feature Importance (SHAP)** page for a plain-language "
        "breakdown alongside the plots.",
    ),
    (
        "Can I trust a 'Low Risk' result completely?",
        "No single model is perfect. A 'Low Risk' result reflects the "
        "patterns in the training data and should not replace routine "
        "screening intervals recommended by clinical guidelines.",
    ),
]

for q, a in faqs:
    with st.expander(q):
        st.write(a)

render_footer()
