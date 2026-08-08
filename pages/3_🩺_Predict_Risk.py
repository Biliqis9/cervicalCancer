"""Predict Cervical Cancer Risk page."""

import logging

import streamlit as st

from components.cards import card_close, card_open, metric_row, risk_banner, section_pill
from components.footer import render_footer
from components.sidebar import render_sidebar_brand
from components.styling import inject_css
from utils.config import FIELD_SECTIONS
from utils.model_loader import ArtifactLoadError, load_core_artifacts
from utils.prediction import predict
from utils.preprocessing import run_preprocessing_pipeline
from utils.validation import validate_form

logger = logging.getLogger("cervical_cancer_app")

st.set_page_config(page_title="Predict Cervical Cancer Risk", page_icon="🩺", layout="wide")
inject_css()
render_sidebar_brand()

section_pill("Assessment Tool")
st.title("Predict Cervical Cancer Risk")
st.write(
    "Complete the form below with the patient's information. All fields "
    "are required. Values are used only for this single prediction and "
    "are not stored."
)

# ------------------------------------------------------------------
# Load model artifacts up front so we fail fast with a friendly message
# ------------------------------------------------------------------
try:
    artifacts = load_core_artifacts()
except ArtifactLoadError as exc:
    st.error(
        "⚠️ The prediction model could not be loaded. Please contact the "
        "system administrator. (The application's model files may be "
        "missing or corrupted.)"
    )
    logger.error("Artifact load failure: %s", exc)
    st.stop()

model = artifacts["model"]
imputer = artifacts["imputer"]
threshold = artifacts["threshold"]
feature_names = artifacts["feature_names"]

# ------------------------------------------------------------------
# Form
# ------------------------------------------------------------------
with st.form("risk_form", clear_on_submit=False):
    form_values = {}

    for section_name, fields in FIELD_SECTIONS.items():
        card_open()
        st.markdown(f"#### {section_name}")
        cols = st.columns(2)
        for i, field in enumerate(fields):
            with cols[i % 2]:
                key = field["key"]
                if field["type"] == "int":
                    form_values[key] = st.number_input(
                        field["label"],
                        min_value=field["min"],
                        max_value=field["max"],
                        value=field["default"],
                        step=1,
                        help=field["help"],
                        key=f"input_{key}",
                    )
                elif field["type"] == "number":
                    form_values[key] = st.number_input(
                        field["label"],
                        min_value=field["min"],
                        max_value=field["max"],
                        value=field["default"],
                        step=field.get("step", 1.0),
                        help=field["help"],
                        key=f"input_{key}",
                    )
                elif field["type"] == "select":
                    form_values[key] = st.selectbox(
                        field["label"],
                        options=field["options"],
                        index=field["options"].index(field["default"]),
                        help=field["help"],
                        key=f"input_{key}",
                    )
        card_close()

    submitted = st.form_submit_button("🔎 Predict Risk", use_container_width=True)

# ------------------------------------------------------------------
# Handle submission
# ------------------------------------------------------------------
if submitted:
    errors = validate_form(form_values)

    if errors:
        st.error("Please fix the following before continuing:")
        for e in errors:
            st.markdown(f"- {e}")
    else:
        try:
            with st.spinner("Running preprocessing and prediction..."):
                processed_df = run_preprocessing_pipeline(
                    form_values, imputer, feature_names
                )
                result = predict(model, processed_df, threshold)
        except Exception:  # noqa: BLE001
            logger.exception("Prediction failed")
            st.error(
                "⚠️ Something went wrong while generating the prediction. "
                "Please double-check your inputs and try again. If the "
                "problem persists, contact support."
            )
        else:
            st.session_state["last_prediction"] = {
                "result": result,
                "processed_df": processed_df,
                "form_values": form_values,
            }

# ------------------------------------------------------------------
# Display results (persist across reruns via session_state)
# ------------------------------------------------------------------
if "last_prediction" in st.session_state:
    result = st.session_state["last_prediction"]["result"]

    st.divider()
    st.markdown("### Result")

    risk_banner(result.risk_level, result.color, result.probability, result.threshold)

    metric_row(
        [
            ("Risk Level", result.risk_level),
            ("Probability", f"{result.probability * 100:.1f}%"),
            ("Confidence Score", f"{result.confidence:.0f}%"),
            ("Predicted Class", "At-Risk" if result.predicted_class == 1 else "Not At-Risk"),
        ]
    )

    card_open()
    st.markdown("#### Clinical Interpretation")
    st.write(result.explanation)
    card_close()

    st.info(
        "💡 Want to see which factors drove this result? Head to the "
        "**Feature Importance (SHAP)** page to explore a detailed "
        "breakdown for this patient."
    )

render_footer()
