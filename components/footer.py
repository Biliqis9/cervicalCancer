"""
footer.py
---------
Consistent footer + medical disclaimer shown on every page.
"""

import streamlit as st


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            <strong>Disclaimer:</strong> This tool provides a decision-support
            risk estimate based on a statistical model and is <strong>not</strong>
            a medical diagnosis. Always consult a qualified healthcare
            professional for clinical decisions.
            <br>
            Cervical Cancer Risk Predictor &middot; Built with Streamlit &middot;
            For research / educational use.
        </div>
        """,
        unsafe_allow_html=True,
    )
