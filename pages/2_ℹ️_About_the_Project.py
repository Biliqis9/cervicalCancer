"""About the Project page."""

import streamlit as st

from components.cards import card_close, card_open, section_pill
from components.footer import render_footer
from components.sidebar import render_sidebar_brand
from components.styling import inject_css
from utils.config import FEATURE_DESCRIPTIONS

st.set_page_config(page_title="About the Project", page_icon="ℹ️", layout="wide")
inject_css()
render_sidebar_brand()

section_pill("About")
st.title("About the Project")

st.markdown(
    """
    Cervical cancer is highly preventable and treatable when detected
    early, yet it remains a leading cause of cancer death among women
    in low- and middle-resource settings where screening access is
    limited. This project explores whether routinely collected
    demographic, behavioral, and clinical data can help **prioritize**
    patients for screening and follow-up.
    """
)

card_open()
st.markdown("### Project Goal")
st.write(
    "Build and deploy a machine learning model that estimates a "
    "patient's probability of elevated cervical cancer risk from "
    "15 risk factors, and package it as an accessible tool for "
    "non-technical healthcare staff."
)
card_close()

col1, col2 = st.columns(2)

with col1:
    card_open()
    st.markdown("### Machine Learning Pipeline")
    st.markdown(
        """
        1. **Data preprocessing** -- cleaning and encoding raw records.
        2. **Missing value imputation** -- Multiple Imputation by
           Chained Equations (MICE), via scikit-learn's
           `IterativeImputer`.
        3. **Feature engineering** -- derived features such as
           `Smoking_Index` computed from smoking history.
        4. **Class imbalance handling** -- addressed during training
           since cervical cancer risk is a minority-class outcome.
        5. **Model training & tuning** -- gradient-boosted trees
           (`XGBClassifier`) with hyperparameter search.
        6. **Threshold selection** -- a custom decision threshold
           (rather than the default 0.5) chosen to balance sensitivity
           and specificity for a screening use case.
        7. **Evaluation** -- held-out test performance, see the
           **Model Performance** page.
        """
    )
    card_close()

with col2:
    card_open()
    st.markdown("### Model Inputs (15 Features)")
    for name, desc in FEATURE_DESCRIPTIONS.items():
        st.markdown(f"- **{name}**: {desc}")
    card_close()

card_open()
st.markdown("### Intended Use & Limitations")
st.markdown(
    """
    - Intended as a **triage / decision-support aid** in food-secure and
      resource-limited screening programs alike -- not a replacement
      for cytology, HPV testing, or colposcopy.
    - Trained on a specific historical dataset; performance may not
      generalize to populations that differ substantially from the
      training data.
    - Predictions should always be reviewed by a qualified clinician
      before any action is taken.
    """
)
card_close()

render_footer()
