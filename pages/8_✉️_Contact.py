"""Contact page."""

import streamlit as st

from components.cards import card_close, card_open, section_pill
from components.footer import render_footer
from components.sidebar import render_sidebar_brand
from components.styling import inject_css

st.set_page_config(page_title="Contact", page_icon="✉️", layout="wide")
inject_css()
render_sidebar_brand()

section_pill("Get in Touch")
st.title("Contact")

col1, col2 = st.columns([1.2, 1])

with col1:
    card_open()
    st.markdown("#### Send a Message")
    with st.form("contact_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message", height=150)
        submitted = st.form_submit_button("Send")

        if submitted:
            if not name or not email or not message:
                st.error("Please fill in all fields before sending.")
            elif "@" not in email:
                st.error("Please enter a valid email address.")
            else:
                # NOTE: This demo form does not send email itself -- wire
                # it to an email service (e.g. SMTP, SendGrid) or a
                # connected inbox integration before deploying.
                st.success(
                    "Thanks! Your message has been noted. "
                    "(Connect this form to an email service to make it live.)"
                )
    card_close()

with col2:
    card_open()
    st.markdown("#### Project Contact")
    st.markdown(
        """
        **Email:** bilkisabd@gmail.com
        **Location:** Abuja, FCT, Nigeria
        """
    )
    card_close()

    card_open()
    st.markdown("#### Report an Issue")
    st.write(
        "Found a bug or an inaccurate prediction? Please describe the "
        "inputs used and the unexpected result so it can be investigated."
    )
    card_close()

render_footer()
