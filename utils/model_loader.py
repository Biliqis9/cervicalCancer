"""
model_loader.py
----------------
Loads all serialized artifacts (model, imputer, feature names, threshold)
exactly as produced by train_and_save_model.py. Nothing is retrained here
-- this module performs inference-only loading, cached for performance.
"""

import logging
import pickle

import joblib
import streamlit as st

from utils.config import (
    FEATURE_NAMES_PATH,
    IMPUTER_PATH,
    MODEL_PATH,
    SHAP_EXPLAINER_PATH,
    THRESHOLD_PATH,
    X_TEST_PATH,
    Y_PROB_TEST_PATH,
    Y_TEST_PATH,
)

logger = logging.getLogger("cervical_cancer_app")
logging.basicConfig(level=logging.INFO)


class ArtifactLoadError(Exception):
    """Raised when a required model artifact cannot be loaded."""


def _safe_joblib_load(path):
    if not path.exists():
        raise ArtifactLoadError(f"Missing required file: {path.name}")
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load %s", path)
        raise ArtifactLoadError(f"Could not load {path.name}: {exc}") from exc


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_core_artifacts():
    """
    Load the four required artifacts: model, imputer, feature_names,
    threshold. Raises ArtifactLoadError with a friendly message if
    anything required is missing -- callers should catch this and
    show a clean UI error rather than a raw traceback.
    """
    model = _safe_joblib_load(MODEL_PATH)
    imputer = _safe_joblib_load(IMPUTER_PATH)
    threshold = _safe_joblib_load(THRESHOLD_PATH)

    if not FEATURE_NAMES_PATH.exists():
        raise ArtifactLoadError("Missing required file: feature_names.pkl")
    with open(FEATURE_NAMES_PATH, "rb") as fh:
        feature_names = pickle.load(fh)

    return {
        "model": model,
        "imputer": imputer,
        "threshold": float(threshold),
        "feature_names": list(feature_names),
    }


@st.cache_resource(show_spinner=False)
def load_optional_test_data():
    """
    Load held-out test data for the Model Performance page, if present.
    Returns a dict with keys X_test / y_test / y_prob_test, each None
    if the corresponding file is missing. This lets the app degrade
    gracefully instead of crashing when the notebook's export step
    (X_test.pkl, y_test.pkl, y_prob_test.pkl) hasn't been run yet.
    """
    result = {"X_test": None, "y_test": None, "y_prob_test": None}
    try:
        if X_TEST_PATH.exists():
            result["X_test"] = joblib.load(X_TEST_PATH)
        if Y_TEST_PATH.exists():
            result["y_test"] = joblib.load(Y_TEST_PATH)
        if Y_PROB_TEST_PATH.exists():
            result["y_prob_test"] = joblib.load(Y_PROB_TEST_PATH)
    except Exception:  # noqa: BLE001
        logger.exception("Error loading optional test-data artifacts")
    return result


@st.cache_resource(show_spinner=False)
def load_optional_shap_explainer():
    """
    Load a pre-fitted shap_explainer.pkl if present. If absent, the
    Feature Importance page builds a shap.TreeExplainer directly from
    the model at runtime (safe and fast for tree models, no saved
    background dataset required).
    """
    if SHAP_EXPLAINER_PATH.exists():
        try:
            return joblib.load(SHAP_EXPLAINER_PATH)
        except Exception:  # noqa: BLE001
            logger.exception("Error loading shap_explainer.pkl")
    return None
