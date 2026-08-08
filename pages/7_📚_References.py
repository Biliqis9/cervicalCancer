"""References page."""

import streamlit as st

from components.cards import card_close, card_open, section_pill
from components.footer import render_footer
from components.sidebar import render_sidebar_brand
from components.styling import inject_css

st.set_page_config(page_title="References", page_icon="📚", layout="wide")
inject_css()
render_sidebar_brand()

section_pill("Sources")
st.title("References")

card_open()
st.markdown("#### Clinical Background")
st.markdown(
    """
    - World Health Organization -- Cervical cancer fact sheet and
      global elimination strategy.
    - International Agency for Research on Cancer (IARC) -- cervical
      cancer screening guidelines.
    - National Cancer Institute -- cervical cancer risk factors and
      HPV information.
    """
)
card_close()

card_open()
st.markdown("#### Methods")
st.markdown(
    """
    - Van Buuren, S., & Groothuis-Oudshoorn, K. -- *mice: Multivariate
      Imputation by Chained Equations in R* (the method underlying
      scikit-learn's `IterativeImputer`).
    - Chen, T., & Guestrin, C. (2016) -- *XGBoost: A Scalable Tree
      Boosting System*.
    - Lundberg, S. M., & Lee, S.-I. (2017) -- *A Unified Approach to
      Interpreting Model Predictions* (SHAP).
    """
)
card_close()

card_open()
st.markdown("#### Software")
st.markdown(
    """
    - [Streamlit](https://streamlit.io/) -- application framework
    - [scikit-learn](https://scikit-learn.org/) -- imputation & metrics
    - [XGBoost](https://xgboost.readthedocs.io/) -- classification model
    - [SHAP](https://shap.readthedocs.io/) -- model explainability
    """
)
card_close()

st.caption(
    "This list is provided for context and does not constitute medical "
    "advice. Always refer to current clinical guidelines in your region."
)

render_footer()
