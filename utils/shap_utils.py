"""
shap_utils.py
-------------
SHAP explainability helpers for the tree-based XGBClassifier.

If models/shap_explainer.pkl is present (as train_and_save_model.py
is expected to produce per README_MODEL_FOLDER.txt), it is used
directly. Otherwise a shap.TreeExplainer is built from the model at
runtime -- this is fast and requires no saved background dataset for
tree models, so the app remains fully functional without that file.
"""

import logging

import numpy as np
import shap
import streamlit as st

logger = logging.getLogger("cervical_cancer_app")


@st.cache_resource(show_spinner=False)
def get_explainer(_model, saved_explainer=None):
    """
    Build (or reuse) a SHAP explainer for the model.

    Uses the default tree_path_dependent TreeExplainer, which works
    directly from the trained trees with no background dataset needed.
    Note: this model was saved with enable_categorical=True, which
    means SHAP's interventional/probability-output mode is not
    supported for it (raises NotImplementedError) -- only
    tree_path_dependent mode works. As a result, SHAP values are in
    **margin (log-odds) space**, not probability-point space. The UI
    labels contributions accordingly (see is_probability_space()).
    """
    if saved_explainer is not None:
        return saved_explainer
    try:
        return shap.TreeExplainer(_model, feature_perturbation="tree_path_dependent")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to build SHAP TreeExplainer")
        raise


def is_probability_space(explainer) -> bool:
    """True if the explainer's contributions are in probability-point units."""
    return getattr(explainer, "model_output", None) == "probability"


def explain_instance(explainer, processed_df):
    """
    Returns a shap.Explanation for a single-row DataFrame, with the
    base value and per-feature contributions for the positive class.
    """
    shap_values = explainer(processed_df)

    # Binary classifiers via shap can return a 3D array (samples, features, classes)
    # or an Explanation with .values shaped (samples, features). Normalize to the
    # positive-class contribution for a single row.
    values = shap_values.values
    base_values = shap_values.base_values

    if values.ndim == 3:
        # (1, n_features, n_classes) -> take positive class (index 1)
        row_values = values[0, :, 1]
        row_base = base_values[0, 1] if np.ndim(base_values) > 0 else base_values
    else:
        row_values = values[0]
        row_base = base_values[0] if np.ndim(base_values) > 0 else base_values

    return row_values, row_base, processed_df.iloc[0]


def explain_batch(explainer, processed_df):
    """Returns raw shap.Explanation for a batch (used for summary plots)."""
    return explainer(processed_df)
