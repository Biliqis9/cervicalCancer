"""Feature Importance (SHAP) page."""

import logging

import matplotlib.pyplot as plt
import shap
import streamlit as st

from components.cards import card_close, card_open, section_pill
from components.footer import render_footer
from components.sidebar import render_sidebar_brand
from components.styling import inject_css
from utils.model_loader import (
    ArtifactLoadError,
    load_core_artifacts,
    load_optional_shap_explainer,
    load_optional_test_data,
)
from utils.shap_utils import explain_batch, explain_instance, get_explainer, is_probability_space

logger = logging.getLogger("cervical_cancer_app")

st.set_page_config(page_title="Feature Importance (SHAP)", page_icon="🔍", layout="wide")
inject_css()
render_sidebar_brand()

section_pill("Explainability")
st.title("Feature Importance (SHAP)")
st.write(
    "SHAP (SHapley Additive exPlanations) values show how much each "
    "feature pushed a prediction above or below the model's average "
    "output, in the same units as the predicted probability."
)

try:
    artifacts = load_core_artifacts()
except ArtifactLoadError:
    st.error("⚠️ The model could not be loaded. Please contact the system administrator.")
    st.stop()

model = artifacts["model"]

try:
    saved_explainer = load_optional_shap_explainer()
    explainer = get_explainer(model, saved_explainer)
except Exception:  # noqa: BLE001
    logger.exception("Could not build SHAP explainer")
    st.error("⚠️ Explainability could not be initialized for this model.")
    st.stop()

tab1, tab2 = st.tabs(["🧍 This Patient's Prediction", "🌐 Global Feature Importance"])

# ------------------------------------------------------------------
# Per-patient explanation (waterfall / force)
# ------------------------------------------------------------------
with tab1:
    if "last_prediction" not in st.session_state:
        st.info(
            "Run a prediction on the **Predict Cervical Cancer Risk** page "
            "first -- this tab explains that patient's specific result."
        )
    else:
        processed_df = st.session_state["last_prediction"]["processed_df"]
        result = st.session_state["last_prediction"]["result"]

        st.markdown(
            f"Explaining a prediction of **{result.probability * 100:.1f}%** "
            f"risk probability (**{result.risk_level}** risk)."
        )

        try:
            row_values, row_base, row_data = explain_instance(explainer, processed_df)

            card_open()
            st.markdown("#### Waterfall Plot")
            st.caption("How each feature moved the prediction away from the model's baseline.")
            fig, ax = plt.subplots(figsize=(8, 5.5))
            explanation = shap.Explanation(
                values=row_values,
                base_values=row_base,
                data=row_data.values,
                feature_names=list(row_data.index),
            )
            shap.plots.waterfall(explanation, show=False, max_display=15)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            card_close()

            card_open()
            st.markdown("#### Force Plot")
            st.caption("A compact view of the same contributions, in one line.")
            try:
                fig2 = plt.figure(figsize=(10, 3))
                shap.plots.force(
                    row_base,
                    row_values,
                    row_data,
                    matplotlib=True,
                    show=False,
                )
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)
            except Exception:  # noqa: BLE001
                logger.exception("Force plot failed")
                st.caption("Force plot unavailable for this input; see waterfall plot above.")
            card_close()

            top_idx = sorted(
                range(len(row_values)), key=lambda i: abs(row_values[i]), reverse=True
            )[:5]
            prob_space = is_probability_space(explainer)
            card_open()
            st.markdown("#### Plain-Language Summary")
            if not prob_space:
                st.caption(
                    "Contributions below are on the model's internal log-odds "
                    "scale (larger magnitude = stronger influence), not "
                    "directly in percentage points -- use them to rank *which* "
                    "factors mattered most, and in *which direction*."
                )
            for i in top_idx:
                feat = row_data.index[i]
                val = row_data.values[i]
                contrib = row_values[i]
                direction = "increased" if contrib > 0 else "decreased"
                if prob_space:
                    st.markdown(
                        f"- **{feat}** (value: {val:.2f}) **{direction}** the predicted "
                        f"risk by {abs(contrib) * 100:.1f} percentage points."
                    )
                else:
                    st.markdown(
                        f"- **{feat}** (value: {val:.2f}) **{direction}** the predicted "
                        f"risk (log-odds contribution: {contrib:+.3f})."
                    )
            card_close()

        except Exception:  # noqa: BLE001
            logger.exception("SHAP instance explanation failed")
            st.error("⚠️ Could not generate an explanation for this prediction.")

# ------------------------------------------------------------------
# Global explanation (summary plot)
# ------------------------------------------------------------------
with tab2:
    test_data = load_optional_test_data()
    X_test = test_data["X_test"]

    if X_test is None:
        st.warning(
            "**Global SHAP summary needs a background dataset.** This tab "
            "reads `X_test.pkl` from `models/` (per your "
            "`README_MODEL_FOLDER.txt`), which wasn't included in the "
            "upload. Add it to `models/` to unlock the global summary "
            "plot -- or, if you'd like, run a handful of predictions "
            "here to build an ad-hoc sample."
        )
    else:
        try:
            sample = X_test.sample(min(200, len(X_test)), random_state=42) if hasattr(
                X_test, "sample"
            ) else X_test
            with st.spinner("Computing SHAP values across the test set..."):
                batch_explanation = explain_batch(explainer, sample)

            card_open()
            st.markdown("#### Summary Plot")
            st.caption(
                "Each dot is one patient/feature pair. Red = high feature value, "
                "blue = low. Position shows impact on predicted risk."
            )
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.summary_plot(
                batch_explanation.values
                if batch_explanation.values.ndim == 2
                else batch_explanation.values[:, :, 1],
                sample,
                show=False,
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            card_close()
        except Exception:  # noqa: BLE001
            logger.exception("Global SHAP summary failed")
            st.error("⚠️ Could not generate the global feature importance summary.")

render_footer()
