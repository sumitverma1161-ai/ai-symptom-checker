"""
app.py  —  AI-Powered Symptom Checker
Streamlit front-end using Google Gemini 3.6 Flash for triage recommendations.

Run:
    streamlit run app.py
"""

import os
import json

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

from prompt_engine import (
    SYSTEM_INSTRUCTION,
    build_prompt,
    parse_response,
    get_triage_meta,
    sort_conditions,
)

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Symptom Checker",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Load env variables (for local development)
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None


# ─────────────────────────────────────────────
# Helper: configure and call Gemini
# ─────────────────────────────────────────────
def run_triage(api_key: str, user_prompt: str) -> dict:
    """Send the prompt to Gemini 2.5 Flash and return parsed JSON result."""
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=2048,
        ),
    )
    return parse_response(response.text)


# ─────────────────────────────────────────────
# Sidebar — API key & about
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🩺 Symptom Checker")
    st.caption("Powered by **Gemini 3.6 Flash**")
    st.divider()

    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_key,
        type="password",
        placeholder="Paste your API key here…",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )

    st.divider()
    st.markdown("### How it works")
    st.markdown(
        """
1. Enter your symptoms below.
2. Optionally add age, gender, and duration.
3. Click **Analyse Symptoms**.
4. The AI returns a triage recommendation and possible conditions.
        """
    )
    st.divider()
    st.markdown("### Triage Levels")
    st.markdown("🟢 **SELF-CARE** — Manage at home")
    st.markdown("🟠 **SEE A DOCTOR** — Book an appointment")
    st.markdown("🔴 **EMERGENCY** — Call 911 / 999 immediately")
    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This tool is for informational purposes only. "
        "It does not replace professional medical advice, diagnosis, or treatment."
    )

    if st.session_state.history:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()


# ─────────────────────────────────────────────
# Main page header
# ─────────────────────────────────────────────
st.title("🩺 AI-Powered Symptom Checker")
st.markdown(
    "> Enter your symptoms and receive an instant AI-powered triage recommendation — "
    "**Self-Care**, **See a Doctor**, or **Emergency** — along with possible conditions."
)
st.divider()

# ─────────────────────────────────────────────
# Input form
# ─────────────────────────────────────────────
with st.form("symptom_form", clear_on_submit=False):
    st.subheader("📋 Describe Your Symptoms")

    symptoms = st.text_area(
        "Symptoms *",
        placeholder="e.g. severe headache, fever 38.5°C, stiff neck, sensitivity to light for the past 24 hours…",
        height=130,
        help="Be as descriptive as possible for a more accurate result.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input("Age", placeholder="e.g. 34", max_chars=3)
    with col2:
        gender = st.selectbox(
            "Biological Sex",
            ["Prefer not to say", "Male", "Female", "Other"],
        )
    with col3:
        duration = st.text_input("Duration", placeholder="e.g. 2 days", max_chars=40)

    extra_context = st.text_area(
        "Additional context (optional)",
        placeholder="e.g. diabetic, currently on metformin, recently travelled abroad…",
        height=80,
    )

    submitted = st.form_submit_button(
        "🔍 Analyse Symptoms", use_container_width=True, type="primary"
    )


# ─────────────────────────────────────────────
# Handle form submission
# ─────────────────────────────────────────────
if submitted:
    if not api_key_input.strip():
        st.error("🔑 Please enter your Gemini API key in the sidebar.")
        st.stop()
    if not symptoms.strip():
        st.error("📝 Please describe your symptoms before submitting.")
        st.stop()

    user_prompt = build_prompt(symptoms, age, gender, duration, extra_context)

    with st.spinner("🤖 Analysing symptoms with Gemini 3.6 Flash…"):
        try:
            result = run_triage(api_key_input.strip(), user_prompt)
            st.session_state.result = result
            st.session_state.history.append(
                {"symptoms": symptoms[:80] + ("…" if len(symptoms) > 80 else ""), "result": result}
            )
        except json.JSONDecodeError:
            st.error(
                "⚠️ The AI returned an unexpected response format. Please try again."
            )
            st.stop()
        except Exception as exc:
            st.error(f"❌ Error calling Gemini API: {exc}")
            st.stop()


# ─────────────────────────────────────────────
# Display result
# ─────────────────────────────────────────────
def display_result(result: dict):
    triage_level = result.get("triage_level", "SEE A DOCTOR").upper()
    meta = get_triage_meta(triage_level)

    # ── Triage Banner ──────────────────────────────────────────────
    st.markdown(
        f"""
<div style="
    background-color:{meta['bg']};
    border:2px solid {meta['border']};
    border-radius:10px;
    padding:20px 24px;
    margin-bottom:16px;
">
    <h2 style="color:{meta['color']};margin:0 0 6px 0;">
        {meta['icon']} {meta['label']}
    </h2>
    <p style="color:{meta['color']};margin:0;font-size:1.05rem;">
        {result.get('triage_summary', '')}
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Tabs for detailed output ────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💊 Conditions", "✅ Recommended Actions", "⚠️ Warning Signs", "📄 Raw JSON"]
    )

    with tab1:
        conditions = sort_conditions(result.get("possible_conditions", []))
        if conditions:
            for cond in conditions:
                likelihood = cond.get("likelihood", "Low")
                badge_colors = {"High": "#dc3545", "Moderate": "#fd7e14", "Low": "#28a745"}
                badge_bg = badge_colors.get(likelihood, "#6c757d")
                st.markdown(
                    f"""
<div style="border:1px solid #dee2e6;border-radius:8px;padding:12px 16px;margin-bottom:10px;background:#fafafa;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
        <strong style="font-size:1rem;">{cond.get('name','')}</strong>
        <span style="
            background:{badge_bg};color:#fff;
            border-radius:12px;padding:2px 10px;
            font-size:0.78rem;font-weight:600;
        ">{likelihood}</span>
    </div>
    <p style="margin:0;color:#555;font-size:0.9rem;">{cond.get('brief_description','')}</p>
</div>
""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No specific conditions identified.")

    with tab2:
        actions = result.get("recommended_actions", [])
        if actions:
            for i, action in enumerate(actions, 1):
                st.markdown(f"**{i}.** {action}")
        else:
            st.info("No specific actions provided.")

    with tab3:
        warnings = result.get("warning_signs", [])
        if warnings:
            for w in warnings:
                st.warning(f"⚠️ {w}")
        else:
            st.success("No immediate red-flag warning signs identified.")

    with tab4:
        st.json(result)

    # ── Disclaimer ──────────────────────────────────────────────────
    st.divider()
    st.caption(
        f"🔒 {result.get('disclaimer', 'This is for informational purposes only. Consult a healthcare professional.')}"
    )


if st.session_state.result:
    st.subheader("📊 Triage Result")
    display_result(st.session_state.result)


# ─────────────────────────────────────────────
# Consultation History
# ─────────────────────────────────────────────
if len(st.session_state.history) > 1:
    st.divider()
    st.subheader("🕑 Previous Checks This Session")
    for i, entry in enumerate(reversed(st.session_state.history[:-1]), 1):
        prev_meta = get_triage_meta(entry["result"].get("triage_level", "SEE A DOCTOR"))
        with st.expander(
            f"{prev_meta['icon']} [{prev_meta['label']}] — {entry['symptoms']}", expanded=False
        ):
            display_result(entry["result"])
