"""
cards.py
--------
Small reusable UI building blocks: section pills, metric cards,
and the color-coded risk result banner.
"""

import streamlit as st


def section_pill(text: str):
    st.markdown(f'<span class="section-pill">{text}</span>', unsafe_allow_html=True)


def card_open():
    st.markdown('<div class="med-card">', unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_row(items: list):
    """items: list of (label, value) tuples, rendered as equal columns."""
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            metric_card(label, value)


def risk_banner(risk_level: str, color: str, probability: float, threshold: float):
    st.markdown(
        f"""
        <div class="risk-banner {color}">
            <div class="risk-title">Risk Level: {risk_level}</div>
            <div class="risk-sub">
                Predicted probability: {probability * 100:.1f}%
                &nbsp;|&nbsp; Decision threshold: {threshold * 100:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, color: str):
    st.markdown(f'<span class="badge {color}">{text}</span>', unsafe_allow_html=True)
