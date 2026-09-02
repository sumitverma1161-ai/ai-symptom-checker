"""
app.py  —  AI-Powered Symptom Checker + Lifestyle Guide
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
if "guide_result" not in st.session_state:
    st.session_state.guide_result = None
if "page" not in st.session_state:
    st.session_state.page = "Symptom Checker"


# ─────────────────────────────────────────────
# Gemini helpers
# ─────────────────────────────────────────────
def run_triage(api_key: str, user_prompt: str) -> dict:
    """Send the symptom prompt to Gemini and return parsed JSON result."""
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


LIFESTYLE_SYSTEM = """You are a medical wellness expert specialising in preventive health and lifestyle medicine.
Generate a comprehensive, practical lifestyle guide based on the health topic provided by the user.

IMPORTANT:
- Always respond with ONLY valid JSON — no markdown fences, no extra text.
- Use plain language anyone can understand.
- Be specific and actionable, not generic.

Response JSON schema (strictly follow this):
{
  "title": "<guide title>",
  "introduction": "<2-3 sentence overview of the topic>",
  "sections": [
    {
      "heading": "<section heading, e.g. Stress Management>",
      "icon": "<a single relevant emoji>",
      "summary": "<1-2 sentence section overview>",
      "tips": [
        {
          "tip": "<short tip title>",
          "detail": "<2-3 sentence practical explanation>"
        }
      ]
    }
  ],
  "daily_routine": {
    "morning": ["<habit 1>", "<habit 2>"],
    "afternoon": ["<habit 1>", "<habit 2>"],
    "evening": ["<habit 1>", "<habit 2>"]
  },
  "trigger_checklist": ["<common trigger 1>", "<common trigger 2>", "..."],
  "when_to_see_doctor": "<1-2 sentences on warning signs that warrant professional consultation>",
  "disclaimer": "This guide is for general wellness information only and does not substitute professional medical advice."
}
"""


def run_lifestyle_guide(api_key: str, topic: str) -> dict:
    """Generate a lifestyle guide for the given health topic."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=f"Generate a comprehensive lifestyle guide on: {topic}",
        config=types.GenerateContentConfig(
            system_instruction=LIFESTYLE_SYSTEM,
            temperature=0.4,
            max_output_tokens=4096,
        ),
    )
    return parse_response(response.text)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🩺 AI Health Assistant")
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
    st.markdown("### 🗂️ Navigation")
    if st.button("🩺 Symptom Checker", use_container_width=True,
                 type="primary" if st.session_state.page == "Symptom Checker" else "secondary"):
        st.session_state.page = "Symptom Checker"
        st.rerun()
    if st.button("📖 Lifestyle Guide", use_container_width=True,
                 type="primary" if st.session_state.page == "Lifestyle Guide" else "secondary"):
        st.session_state.page = "Lifestyle Guide"
        st.rerun()

    st.divider()

    if st.session_state.page == "Symptom Checker":
        st.markdown("### Triage Levels")
        st.markdown("🟢 **SELF-CARE** — Manage at home")
        st.markdown("🟠 **SEE A DOCTOR** — Book an appointment")
        st.markdown("🔴 **EMERGENCY** — Call 911 / 999 immediately")
    else:
        st.markdown("### Guide Topics")
        st.markdown("Try topics like:")
        st.markdown("- Headache prevention")
        st.markdown("- Managing anxiety")
        st.markdown("- Better sleep hygiene")
        st.markdown("- Reducing back pain")
        st.markdown("- Heart health habits")

    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This tool is for informational purposes only. "
        "It does not replace professional medical advice."
    )

    if st.session_state.history:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()


# ═══════════════════════════════════════════════════════════
# PAGE 1 — SYMPTOM CHECKER
# ═══════════════════════════════════════════════════════════
if st.session_state.page == "Symptom Checker":

    st.title("🩺 AI-Powered Symptom Checker")
    st.markdown(
        "> Enter your symptoms and receive an instant AI-powered triage recommendation — "
        "**Self-Care**, **See a Doctor**, or **Emergency** — along with possible conditions."
    )
    st.divider()

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
                st.error("⚠️ The AI returned an unexpected response format. Please try again.")
                st.stop()
            except Exception as exc:
                st.error(f"❌ Error calling Gemini API: {exc}")
                st.stop()

    # ── Display result ─────────────────────────────────────────────
    def display_result(result: dict):
        triage_level = result.get("triage_level", "SEE A DOCTOR").upper()
        meta = get_triage_meta(triage_level)

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

        tab1, tab2, tab3, tab4 = st.tabs(
            ["💊 Conditions", "✅ Recommended Actions", "⚠️ Warning Signs", "🩺 Get Help"]
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
        <strong style="font-size:1rem;color:#000000;">{cond.get('name','')}</strong>
        <span style="background:{badge_bg};color:#fff;border-radius:12px;padding:2px 10px;font-size:0.78rem;font-weight:600;">{likelihood}</span>
    </div>
    <p style="margin:0;color:#000000;font-size:0.9rem;">{cond.get('brief_description','')}</p>
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
            st.markdown("### 🩺 Recommended Specialists & Doctors")
            st.caption("Based on your symptoms, these are the types of doctors you should consider consulting.")
            st.divider()

            doctors_list = result.get("recommended_doctors", [])
            if doctors_list:
                for specialty_group in doctors_list:
                    specialty = specialty_group.get("specialty", "")
                    why = specialty_group.get("why", "")
                    example_doctors = specialty_group.get("example_doctors", [])

                    st.markdown(f"#### 👨‍⚕️ {specialty}")
                    st.info(f"**Why this specialist?** {why}")

                    for doc in example_doctors:
                        st.markdown(
                            f"""
<div style="border:1px solid #dee2e6;border-radius:10px;padding:16px 20px;margin-bottom:12px;background:#f8f9fa;">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
        <span style="font-size:2rem;">👤</span>
        <div>
            <strong style="font-size:1.05rem;color:#1f2328;">{doc.get('name','')}</strong><br>
            <span style="font-size:0.85rem;color:#57606a;">{doc.get('qualification','')}</span>
        </div>
    </div>
    <p style="margin:4px 0;font-size:0.9rem;">🏅 <strong>Experience:</strong> {doc.get('experience','')}</p>
    <p style="margin:4px 0;font-size:0.9rem;">🔬 <strong>Known for:</strong> {doc.get('known_for','')}</p>
</div>
""",
                            unsafe_allow_html=True,
                        )
                    st.divider()
            else:
                st.info("No specific doctor recommendations available. Please consult your local healthcare provider.")

            st.caption("⚠️ The doctors listed are AI-generated examples for guidance only. Please search for verified, licensed practitioners in your area.")

        st.divider()
        st.caption(
            f"🔒 {result.get('disclaimer', 'This is for informational purposes only. Consult a healthcare professional.')}"
        )

    if st.session_state.result:
        st.subheader("📊 Triage Result")
        display_result(st.session_state.result)

    if len(st.session_state.history) > 1:
        st.divider()
        st.subheader("🕑 Previous Checks This Session")
        for i, entry in enumerate(reversed(st.session_state.history[:-1]), 1):
            prev_meta = get_triage_meta(entry["result"].get("triage_level", "SEE A DOCTOR"))
            with st.expander(
                f"{prev_meta['icon']} [{prev_meta['label']}] — {entry['symptoms']}", expanded=False
            ):
                display_result(entry["result"])


# ═══════════════════════════════════════════════════════════
# PAGE 2 — LIFESTYLE GUIDE
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "Lifestyle Guide":

    st.title("📖 AI Lifestyle & Prevention Guide")
    st.markdown(
        "> Get a comprehensive, AI-generated lifestyle guide on any health topic — "
        "covering habits, triggers, daily routines, and prevention strategies."
    )
    st.divider()

    # ── Quick-pick topic buttons ───────────────────────────────────
    st.markdown("**⚡ Quick Topics:**")
    quick_cols = st.columns(4)
    quick_topics = [
        "Headache prevention",
        "Better sleep hygiene",
        "Stress & anxiety management",
        "Heart health habits",
    ]
    for idx, qt in enumerate(quick_topics):
        with quick_cols[idx]:
            if st.button(qt, use_container_width=True):
                st.session_state["guide_topic_prefill"] = qt
                st.rerun()

    st.divider()

    # ── Topic input form ───────────────────────────────────────────
    prefill = st.session_state.get("guide_topic_prefill", "")
    with st.form("guide_form", clear_on_submit=False):
        st.subheader("🔎 Enter a Health Topic")
        guide_topic = st.text_input(
            "Topic *",
            value=prefill,
            placeholder="e.g. Lifestyle changes to prevent headaches, Managing diabetes with diet…",
            help="Be specific for a more targeted guide.",
        )
        guide_submitted = st.form_submit_button(
            "📖 Generate Lifestyle Guide", use_container_width=True, type="primary"
        )

    if guide_submitted:
        if not api_key_input.strip():
            st.error("🔑 Please enter your Gemini API key in the sidebar.")
            st.stop()
        if not guide_topic.strip():
            st.error("📝 Please enter a health topic.")
            st.stop()

        st.session_state["guide_topic_prefill"] = guide_topic

        with st.spinner(f"✍️ Generating guide for **{guide_topic}**…"):
            try:
                guide = run_lifestyle_guide(api_key_input.strip(), guide_topic)
                st.session_state.guide_result = guide
            except json.JSONDecodeError:
                st.error("⚠️ The AI returned an unexpected format. Please try again.")
                st.stop()
            except Exception as exc:
                st.error(f"❌ Error calling Gemini API: {exc}")
                st.stop()

    # ── Display guide ──────────────────────────────────────────────
    def display_guide(guide: dict):
        st.markdown(f"## 📋 {guide.get('title', 'Lifestyle Guide')}")
        st.info(guide.get("introduction", ""))
        st.divider()

        # ── Sections ──────────────────────────────────────────────
        sections = guide.get("sections", [])
        for section in sections:
            icon = section.get("icon", "•")
            heading = section.get("heading", "")
            summary = section.get("summary", "")
            tips = section.get("tips", [])

            st.markdown(f"### {icon} {heading}")
            st.markdown(f"*{summary}*")

            for tip in tips:
                with st.expander(f"💡 {tip.get('tip', '')}"):
                    st.markdown(tip.get("detail", ""))

            st.divider()

        # ── Daily Routine ──────────────────────────────────────────
        routine = guide.get("daily_routine", {})
        if routine:
            st.markdown("### 🗓️ Suggested Daily Routine")
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.markdown("#### 🌅 Morning")
                for item in routine.get("morning", []):
                    st.markdown(f"- {item}")
            with rc2:
                st.markdown("#### ☀️ Afternoon")
                for item in routine.get("afternoon", []):
                    st.markdown(f"- {item}")
            with rc3:
                st.markdown("#### 🌙 Evening")
                for item in routine.get("evening", []):
                    st.markdown(f"- {item}")
            st.divider()

        # ── Trigger Checklist ──────────────────────────────────────
        triggers = guide.get("trigger_checklist", [])
        if triggers:
            st.markdown("### ⚠️ Common Triggers Checklist")
            st.markdown("Track which of these may be affecting you:")
            tcols = st.columns(2)
            for i, trigger in enumerate(triggers):
                with tcols[i % 2]:
                    st.checkbox(trigger, key=f"trigger_{i}")
            st.divider()

        # ── When to see a doctor ───────────────────────────────────
        when_doc = guide.get("when_to_see_doctor", "")
        if when_doc:
            st.markdown("### 🏥 When to See a Doctor")
            st.warning(f"🔔 {when_doc}")
            st.divider()

        # ── Disclaimer ─────────────────────────────────────────────
        st.caption(f"🔒 {guide.get('disclaimer', 'This guide is for general wellness information only.')}")

    if st.session_state.guide_result:
        st.divider()
        display_guide(st.session_state.guide_result)
