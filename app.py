"""
app.py  —  AI-Powered Symptom Checker + Lifestyle Guide + Offline ML Diagnosis
Streamlit front-end using Google Gemini 3.6 Flash (online) and local
scikit-learn / PyTorch models (offline, no API key required).

Run:
    streamlit run app.py
"""

import os
import json
import sys
from pathlib import Path

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

# ── Make ml_models importable regardless of working directory ────
sys.path.insert(0, str(Path(__file__).parent))
from ml_models.ml_engine import (
    predict_from_text, predict_from_image, get_model_status,
    _inject_kaggle_credentials, TextPredictor,
)

# ── On Streamlit Cloud, trigger auto-training at startup ──────────
# Inject Kaggle creds from st.secrets before any predictor loads
_inject_kaggle_credentials()

# ─────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Symptom Checker",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# Global gradient + UI polish
# ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Main app background: soft light-blue → mint-green gradient ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e8f4fd 0%, #d6eef8 30%, #d4f1e8 65%, #c8edd9 100%);
    min-height: 100vh;
}

/* ── Sidebar: frosted white tint ── */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.82) !important;
    backdrop-filter: blur(8px);
    border-right: 1px solid rgba(180, 220, 210, 0.5);
}

/* ── Top header bar ── */
[data-testid="stHeader"] {
    background: rgba(232, 244, 253, 0.75);
    backdrop-filter: blur(6px);
}

/* ── Form / widget containers: glass card feel ── */
[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.75);
    border-radius: 16px;
    border: 1px solid rgba(180, 220, 210, 0.6);
    padding: 8px;
    backdrop-filter: blur(4px);
}

/* ── Expander header ── */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.65);
    border-radius: 10px;
    border: 1px solid rgba(180, 220, 210, 0.5);
}

/* ── Tab list ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255, 255, 255, 0.55);
    border-radius: 10px;
    padding: 4px;
}

/* ── Metric / info boxes ── */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* ── Dividers: subtle teal tint ── */
hr {
    border-color: rgba(100, 190, 170, 0.3) !important;
}

/* ── ML confidence bar ── */
.conf-bar-wrap {
    background: #e9ecef;
    border-radius: 6px;
    height: 10px;
    width: 100%;
    overflow: hidden;
    margin-top: 4px;
}
.conf-bar-fill {
    height: 10px;
    border-radius: 6px;
    transition: width 0.4s ease;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────
# Load env variables (for local development)
# ─────────────────────────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────
for key, default in [
    ("history", []),
    ("result", None),
    ("guide_result", None),
    ("page", "Symptom Checker"),
    ("ml_text_result", None),
    ("ml_image_result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────
# Gemini helpers
# ─────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────
# ML result display helpers
# ─────────────────────────────────────────────────────────────────

def _conf_color(conf: float) -> str:
    """Return a hex colour representing confidence level."""
    if conf >= 0.6:
        return "#28a745"
    if conf >= 0.3:
        return "#fd7e14"
    return "#6c757d"


def _render_prediction_card(rank: int, name: str, confidence: float, label_key: str = "disease"):
    """Render a single prediction result card with confidence bar."""
    pct = round(confidence * 100, 1)
    color = _conf_color(confidence)
    st.markdown(
        f"""
<div style="
    border:1px solid #dee2e6;border-radius:10px;
    padding:12px 16px;margin-bottom:10px;
    background:rgba(255,255,255,0.8);
">
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
            <span style="font-size:0.75rem;color:#6c757d;font-weight:600;">#{rank}</span>
            &nbsp;
            <strong style="font-size:1rem;color:#1f2328;">{name}</strong>
        </div>
        <span style="
            background:{color};color:#fff;
            border-radius:12px;padding:2px 10px;
            font-size:0.82rem;font-weight:700;
        ">{pct}%</span>
    </div>
    <div class="conf-bar-wrap" style="margin-top:8px;">
        <div class="conf-bar-fill" style="width:{pct}%;background:{color};"></div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_model_info(model_info: dict, source_label: str):
    """Render a small model provenance badge."""
    if not model_info:
        return
    acc = model_info.get("accuracy", "N/A")
    mtype = model_info.get("type", "ML Model")
    n_cls = model_info.get("num_classes", "?")
    st.caption(
        f"🤖 **{source_label}** · {mtype} · {n_cls} classes · "
        f"accuracy: **{acc if isinstance(acc, str) else f'{acc*100:.1f}%'}**"
    )


def _render_model_status_badge(ready: bool, label: str):
    if ready:
        st.sidebar.success(f"✅ {label} loaded")
    else:
        st.sidebar.warning(f"⚠️ {label} not trained yet")


# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🩺 AI Health Assistant")
    st.divider()

    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_key,
        type="password",
        placeholder="Paste your API key here…",
        help="Required only for AI Symptom Checker & Lifestyle Guide pages. Not needed for offline ML Diagnosis.",
    )

    st.divider()

    # ── Offline ML model status ────────────────────────────────
    st.markdown("**🧠 Offline ML Models**")
    status = get_model_status()
    text_ready = status["text_model"]["ready"]
    text_error = status["text_model"].get("error", "")

    _render_model_status_badge(text_ready, "Text Disease Model")
    _render_model_status_badge(status["image_model"]["ready"], "Image Classifier")

    # Show auto-training status if model isn't ready yet
    if not text_ready and text_error:
        if "KAGGLE_USERNAME" in text_error or "Auto-training failed" in text_error:
            st.sidebar.error(
                "⚙️ **Text model needs Kaggle credentials.**\n\n"
                "Add to **App Settings → Secrets**:\n"
                "```toml\n"
                "KAGGLE_USERNAME = \"your_username\"\n"
                "KAGGLE_KEY = \"your_api_key\"\n"
                "```\n"
                "Then reboot the app — it will train automatically.",
                icon="🔑",
            )
        else:
            st.sidebar.info("⏳ Text model will train on first use (needs Kaggle key).")

    with st.expander("ℹ️ Training Instructions", expanded=False):
        st.markdown(
            """
**☁️ Streamlit Cloud (automatic):**
1. Go to App Settings → Secrets and add:
```toml
KAGGLE_USERNAME = "your_username"
KAGGLE_KEY = "your_api_key"
```
2. Reboot the app — model trains automatically on first load.

**💻 Local (manual):**
1. Download [Training.csv from Kaggle](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)
2. Place it in `ml_models/data/`
3. Run: `python ml_models/train_text_model.py`

**Image Model (Skin / Medical Imaging):**
1. Download images from one or more of:
   - [Google SCIN](https://github.com/google-research-datasets/scin)
   - [TCIA](https://www.cancerimagingarchive.net/)
   - [MedImg](https://www.cuilab.cn/medimg/)
2. Organise as `ml_models/data/images/<dataset>/<label>/<image>.jpg`
3. Run: `python ml_models/train_image_model.py`
"""
        )

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


# ═══════════════════════════════════════════════════════════════════
# RIGHT-PANEL HEADER — App title left, Nav right
# ═══════════════════════════════════════════════════════════════════
hdr_left, hdr_right = st.columns([1.1, 1], gap="large")

with hdr_left:
    st.markdown(
        """
<div style="padding:24px 0 16px 0;">
    <span style="font-size:3rem;">🩺</span>
    <h1 style="margin:8px 0 4px 0;font-size:2rem;font-weight:800;letter-spacing:-0.5px;color:#1a3c4d;">
        AI Health Assistant
    </h1>
    <p style="color:#4a7a8a;font-size:0.95rem;margin:0;">
        Gemini 3.6 Flash · Offline ML · Image Analysis
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

with hdr_right:
    st.markdown(
        """
<div style="
    background:rgba(255,255,255,0.72);
    border:1.5px solid rgba(100,190,170,0.45);
    border-radius:18px;
    padding:18px 20px 14px 20px;
    margin-top:18px;
    box-shadow:0 2px 12px rgba(80,160,150,0.10);
">
    <div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;color:#4a9b8e;margin-bottom:10px;">
        ◈ NAVIGATE
    </div>
""",
        unsafe_allow_html=True,
    )
    nav_b1, nav_b2, nav_b3 = st.columns(3)
    with nav_b1:
        if st.button(
            "🩺 Symptom\nChecker",
            use_container_width=True,
            type="primary" if st.session_state.page == "Symptom Checker" else "secondary",
            key="nav_symptom",
        ):
            st.session_state.page = "Symptom Checker"
            st.rerun()
    with nav_b2:
        if st.button(
            "🧠 ML\nDiagnosis",
            use_container_width=True,
            type="primary" if st.session_state.page == "ML Diagnosis" else "secondary",
            key="nav_ml",
        ):
            st.session_state.page = "ML Diagnosis"
            st.rerun()
    with nav_b3:
        if st.button(
            "📖 Lifestyle\nGuide",
            use_container_width=True,
            type="primary" if st.session_state.page == "Lifestyle Guide" else "secondary",
            key="nav_lifestyle",
        ):
            st.session_state.page = "Lifestyle Guide"
            st.rerun()

    # ── Triage legend (only on Symptom Checker page) ─────────
    if st.session_state.page == "Symptom Checker":
        st.markdown(
            """
<div style="margin-top:14px;">
    <div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;color:#4a9b8e;margin-bottom:10px;">
        ◈ TRIAGE LEVELS
    </div>
    <div style="display:flex;flex-direction:column;gap:8px;">
        <div style="
            background:#d4edda;border-left:5px solid #28a745;
            border-radius:10px;padding:8px 14px;
            display:flex;align-items:center;gap:10px;
        ">
            <span style="font-size:1.3rem;">🟢</span>
            <div>
                <div style="font-weight:700;color:#1e7e34;font-size:0.82rem;">SELF-CARE</div>
                <div style="color:#1e7e34;font-size:0.72rem;">Manage at home</div>
            </div>
        </div>
        <div style="
            background:#fff3cd;border-left:5px solid #ffc107;
            border-radius:10px;padding:8px 14px;
            display:flex;align-items:center;gap:10px;
        ">
            <span style="font-size:1.3rem;">🟠</span>
            <div>
                <div style="font-weight:700;color:#856404;font-size:0.82rem;">SEE A DOCTOR</div>
                <div style="color:#856404;font-size:0.72rem;">Book an appointment</div>
            </div>
        </div>
        <div style="
            background:#f8d7da;border-left:5px solid #dc3545;
            border-radius:10px;padding:8px 14px;
            display:flex;align-items:center;gap:10px;
        ">
            <span style="font-size:1.3rem;">🔴</span>
            <div>
                <div style="font-weight:700;color:#721c24;font-size:0.82rem;">EMERGENCY</div>
                <div style="color:#721c24;font-size:0.72rem;">Call 911 / 999 now</div>
            </div>
        </div>
    </div>
</div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — SYMPTOM CHECKER  (Gemini AI)
# ═══════════════════════════════════════════════════════════════════
if st.session_state.page == "Symptom Checker":

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


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — OFFLINE ML DIAGNOSIS
# ═══════════════════════════════════════════════════════════════════
elif st.session_state.page == "ML Diagnosis":

    st.markdown(
        """
## 🧠 Offline ML Medical Diagnosis
> Predict likely conditions from **typed symptoms** or an **uploaded medical image** — 
> 100% offline, no API key needed. Powered by locally trained scikit-learn and PyTorch models.
"""
    )

    # ── Dataset provenance notice ──────────────────────────────
    with st.expander("📚 Dataset Sources & Citations", expanded=False):
        st.markdown(
            """
| Model | Dataset | Source |
|-------|---------|--------|
| Text Disease Classifier | Disease Prediction Using ML | [Kaggle — kaushil268](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning) |
| Text Disease Classifier | Heart Disease Dataset | [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| Text Disease Classifier | Pima Indians Diabetes | [UCI ML Repository](https://archive.ics.uci.edu/dataset/34/diabetes) |
| Image Classifier | Skin Condition Image Network | [Google SCIN](https://github.com/google-research-datasets/scin) |
| Image Classifier | Medical Imaging Collections | [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/) |
| Image Classifier | MedImg Dataset | [CUI Lab / MedImg](https://www.cuilab.cn/medimg/) |

> Models are auto-downloaded and trained on Streamlit Cloud if Kaggle credentials are set in Secrets.
"""
        )

    st.divider()

    # ── Auto-train spinner (shown only while training on first load) ──
    _ts = get_model_status()["text_model"]
    if not _ts["ready"] and not _ts.get("error"):
        with st.spinner("⏳ Training ML model from Kaggle dataset… this takes ~2 minutes on first load."):
            TextPredictor.get()   # blocks until training completes
        st.rerun()

    # ════════════════════════════════════════════════════════════
    # TAB A — Text Symptom Prediction
    # TAB B — Image-Based Prediction
    # ════════════════════════════════════════════════════════════
    tab_text, tab_image = st.tabs(["📝 Text Symptom Analysis", "🖼️ Image-Based Diagnosis"])

    # ── TAB A: Text ─────────────────────────────────────────────
    with tab_text:
        st.markdown(
            "Enter your symptoms in plain English. The model will match them against "
            "**132 known symptom indicators** trained on the Kaggle disease dataset."
        )

        text_status = get_model_status()["text_model"]
        if not text_status["ready"]:
            err = text_status.get("error", "")
            if "KAGGLE_USERNAME" in err or "Auto-training failed" in err or "kagglehub" in err:
                st.error(
                    "🔑 **Kaggle credentials required for auto-training.**\n\n"
                    "Go to **App Settings → Secrets** and add:\n"
                    "```toml\n"
                    "KAGGLE_USERNAME = \"your_username\"\n"
                    "KAGGLE_KEY = \"your_api_key\"\n"
                    "```\n"
                    "Then reboot — the model will train automatically (~2 min).",
                )
            else:
                st.warning(f"⚠️ Text model not loaded.\n\n```\n{err}\n```", icon="🚧")
        else:
            trained_rt = text_status["meta"].get("trained_at_runtime", False)
            st.success(
                f"✅ Model ready — {text_status['meta'].get('num_classes', '?')} diseases · "
                f"accuracy {text_status['meta'].get('test_accuracy', 'N/A')}"
                + (" · *(trained at runtime)*" if trained_rt else ""),
                icon="🤖",
            )

        with st.form("ml_text_form", clear_on_submit=False):
            ml_symptoms = st.text_area(
                "Describe your symptoms *",
                placeholder=(
                    "e.g. I have a high fever, itching all over, and fatigue for the past 3 days. "
                    "Also experiencing loss of appetite and mild joint pain."
                ),
                height=140,
                help="Use natural language — the model recognises symptom keywords automatically.",
            )
            top_n_text = st.slider("Number of predictions to show", min_value=1, max_value=10, value=5)
            ml_text_submitted = st.form_submit_button(
                "🔬 Predict Disease", use_container_width=True, type="primary"
            )

        if ml_text_submitted:
            if not ml_symptoms.strip():
                st.error("📝 Please enter your symptoms.")
                st.stop()
            with st.spinner("🧠 Running offline ML inference…"):
                st.session_state.ml_text_result = predict_from_text(ml_symptoms.strip(), top_n=top_n_text)

        # ── Display text prediction result ─────────────────────
        if st.session_state.ml_text_result:
            res = st.session_state.ml_text_result

            if "error" in res and not res.get("predictions"):
                st.error(f"❌ {res['error']}")
            else:
                st.divider()
                st.subheader("🔬 ML Prediction Results")

                # Matched symptoms summary
                matched = res.get("matched_symptoms", [])
                n_matched = res.get("num_symptoms_matched", 0)
                if matched:
                    st.info(
                        f"🎯 **{n_matched} symptom(s) matched:** "
                        + " · ".join(f"`{s}`" for s in matched)
                    )
                else:
                    st.warning(
                        "⚠️ No specific symptom keywords were matched in your text. "
                        "Try using more specific medical terms (e.g. 'high fever', 'joint pain', 'itching')."
                    )

                predictions = res.get("predictions", [])
                if predictions:
                    st.markdown("#### Top Predicted Conditions")
                    for pred in predictions:
                        _render_prediction_card(
                            pred["rank"], pred["disease"], pred["confidence"], label_key="disease"
                        )
                    _render_model_info(res.get("model_info", {}), "Text Disease Classifier")
                else:
                    st.info("No predictions available. Try adding more descriptive symptoms.")

                st.divider()
                st.caption(
                    "⚠️ These predictions are generated by a locally trained machine learning model "
                    "and are for informational purposes only. They do not constitute medical advice. "
                    "Always consult a qualified healthcare professional for diagnosis."
                )

    # ── TAB B: Image ─────────────────────────────────────────────
    with tab_image:
        st.markdown(
            "Upload a medical or skin condition image. The MobileNetV2 model — "
            "trained on Google SCIN, TCIA, and MedImg — will classify it offline."
        )

        image_status = get_model_status()["image_model"]
        if not image_status["ready"]:
            st.warning(
                f"⚠️ Image model not loaded.\n\n```\n{image_status['error']}\n```",
                icon="🚧",
            )
        else:
            st.success(
                f"✅ Model ready — {image_status['meta'].get('num_classes', '?')} classes, "
                f"val. accuracy {image_status['meta'].get('best_val_accuracy', 'N/A')}",
                icon="🤖",
            )

        # ── File uploader ──────────────────────────────────────
        uploaded_file = st.file_uploader(
            "Upload a medical image",
            type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"],
            help=(
                "Accepted formats: JPEG, PNG, BMP, WebP, TIFF. "
                "For best results use clear, well-lit images of skin conditions or medical scans. "
                "Images are processed locally and never sent to external servers."
            ),
            key="ml_image_uploader",
        )

        col_img, col_ctrl = st.columns([1, 1], gap="medium")

        with col_img:
            if uploaded_file is not None:
                st.image(
                    uploaded_file,
                    caption=f"📷 {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)",
                    use_container_width=True,
                )

        with col_ctrl:
            if uploaded_file is not None:
                top_n_img = st.slider(
                    "Number of predictions", min_value=1, max_value=10, value=5, key="top_n_img"
                )
                st.markdown("")  # spacer
                if st.button(
                    "🔬 Classify Image",
                    use_container_width=True,
                    type="primary",
                    key="classify_btn",
                ):
                    with st.spinner("🧠 Running image classification…"):
                        img_bytes = uploaded_file.read()
                        st.session_state.ml_image_result = predict_from_image(
                            img_bytes, top_n=top_n_img
                        )

        # ── Display image prediction result ───────────────────
        if st.session_state.ml_image_result:
            img_res = st.session_state.ml_image_result

            if "error" in img_res and not img_res.get("predictions"):
                st.error(f"❌ {img_res['error']}")
            else:
                st.divider()
                st.subheader("🔬 Image Classification Results")

                predictions = img_res.get("predictions", [])
                if predictions:
                    for pred in predictions:
                        _render_prediction_card(
                            pred["rank"], pred["condition"], pred["confidence"], label_key="condition"
                        )
                    _render_model_info(img_res.get("model_info", {}), "MobileNetV2 Image Classifier")

                    # Dataset sources for the image model
                    datasets_used = img_res.get("model_info", {}).get("datasets", [])
                    if datasets_used:
                        with st.expander("📚 Training Datasets Used for This Model"):
                            for ds in datasets_used:
                                st.markdown(f"- **{ds['name']}** — [{ds['url']}]({ds['url']})")
                else:
                    st.info("No predictions returned. The model may need re-training on more data.")

                st.divider()
                st.caption(
                    "⚠️ Image classification results are generated by a locally trained model "
                    "and are for informational/research purposes only. "
                    "They are not a substitute for professional medical imaging analysis or diagnosis."
                )
        elif uploaded_file is None:
            st.info(
                "👆 Upload an image above to get started. "
                "Supported types: skin conditions, dermatology photos, or any medical image "
                "from the supported dataset categories."
            )


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — LIFESTYLE GUIDE
# ═══════════════════════════════════════════════════════════════════
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
