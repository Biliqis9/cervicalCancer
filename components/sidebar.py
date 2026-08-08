"""
sidebar.py
----------
Shared sidebar branding shown above Streamlit's automatic page
navigation (built from the files in pages/).
"""

import streamlit as st


def render_sidebar_brand():
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
                <div style="font-size:2rem;">🩺</div>
                <div style="font-weight:800; font-size:1.05rem; color:#EAF2F8;">
                    Cervical Cancer<br>Risk Predictor
                </div>
                <div style="font-size:0.75rem; color:#9FB6CC; margin-top:2px;">
                    Clinical Decision Support
                </div>
            </div>
            <hr style="border-color:rgba(255,255,255,0.15);">
            """,
            unsafe_allow_html=True,
        )
