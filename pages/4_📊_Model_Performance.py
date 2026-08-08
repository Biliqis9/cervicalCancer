"""Model Performance page."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from components.cards import card_close, card_open, metric_row, section_pill
from components.footer import render_footer
from components.sidebar import render_sidebar_brand
from components.styling import inject_css
from utils.model_loader import ArtifactLoadError, load_core_artifacts, load_optional_test_data

logger = logging.getLogger("cervical_cancer_app")

st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")
inject_css()
render_sidebar_brand()

section_pill("Validation")
st.title("Model Performance")
st.write(
    "Metrics below are computed on the model's held-out test set, "
    "exactly as produced by the training notebook -- no data is "
    "re-evaluated or retrained here."
)

try:
    artifacts = load_core_artifacts()
except ArtifactLoadError:
    st.error("⚠️ The model could not be loaded. Please contact the system administrator.")
    st.stop()

threshold = artifacts["threshold"]
test_data = load_optional_test_data()
X_test, y_test, y_prob_test = (
    test_data["X_test"],
    test_data["y_test"],
    test_data["y_prob_test"],
)

if y_test is None or y_prob_test is None:
    st.warning(
        "**Performance data not available yet.** This page reads "
        "`X_test.pkl`, `y_test.pkl`, and `y_prob_test.pkl` from the "
        "`models/` folder, as listed in your `README_MODEL_FOLDER.txt`, "
        "but those files were not included in the upload. Export them "
        "from your training notebook (held-out test features, labels, "
        "and predicted probabilities) and drop them into `models/` to "
        "populate this page automatically -- no code changes needed."
    )
else:
    y_pred = (np.asarray(y_prob_test) >= threshold).astype(int)
    y_test_arr = np.asarray(y_test)

    acc = accuracy_score(y_test_arr, y_pred)
    prec = precision_score(y_test_arr, y_pred, zero_division=0)
    rec = recall_score(y_test_arr, y_pred, zero_division=0)
    f1 = f1_score(y_test_arr, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test_arr, y_prob_test)
    except ValueError:
        auc = float("nan")

    st.markdown("### Key Metrics")
    metric_row(
        [
            ("Accuracy", f"{acc * 100:.1f}%"),
            ("Precision", f"{prec * 100:.1f}%"),
            ("Recall", f"{rec * 100:.1f}%"),
            ("F1 Score", f"{f1 * 100:.1f}%"),
        ]
    )
    st.write("")
    metric_row([("ROC AUC", f"{auc:.3f}" if not np.isnan(auc) else "N/A"),
                ("Decision Threshold", f"{threshold:.2f}"),
                ("Test Set Size", f"{len(y_test_arr)}"),
                ("Positive Cases", f"{int(np.sum(y_test_arr))}")])

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        card_open()
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4.5, 4.2))
        cm = confusion_matrix(y_test_arr, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Risk", "At Risk"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        st.pyplot(fig, use_container_width=True)
        card_close()

    with col2:
        card_open()
        st.markdown("#### ROC Curve")
        fig2, ax2 = plt.subplots(figsize=(4.5, 4.2))
        RocCurveDisplay.from_predictions(y_test_arr, y_prob_test, ax=ax2)
        ax2.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        st.pyplot(fig2, use_container_width=True)
        card_close()

    card_open()
    st.markdown("#### Precision-Recall Curve")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    PrecisionRecallDisplay.from_predictions(y_test_arr, y_prob_test, ax=ax3)
    st.pyplot(fig3, use_container_width=True)
    card_close()

    st.caption(
        "Metrics use the model's saved decision threshold "
        f"({threshold:.2f}) rather than the default 0.5 cutoff, matching "
        "how predictions are made elsewhere in this app."
    )

render_footer()
