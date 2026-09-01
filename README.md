# AI-Powered Symptom Checker

An AI-powered medical triage application built with **Streamlit** and **Google Gemini 2.5 Flash**. Users enter symptoms and receive an instant triage recommendation — **Self-Care**, **See a Doctor**, or **Emergency** — along with possible related conditions, recommended actions, and warning signs.

---

## Features

- 🤖 **Gemini 2.5 Flash** AI backbone — fast, structured JSON responses
- 🟢🟠🔴 **3-level triage** system (Self-Care / See a Doctor / Emergency)
- 💊 **Possible conditions** with likelihood ratings (High / Moderate / Low)
- ✅ **Recommended actions** specific to your symptoms
- ⚠️ **Warning signs** to watch out for
- 🕑 **Session history** — review all checks in the current session
- 🔑 **API key via sidebar or `.env`** — no code changes needed

---

## Project Structure

```
symptom-checker/
├── app.py              # Streamlit UI
├── prompt_engine.py    # Prompt builder, response parser, display helpers
├── requirements.txt    # Python dependencies
├── .env.example        # API key template
└── README.md
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API key

**Option A — `.env` file (recommended for local dev):**

```bash
cp .env.example .env
# Edit .env and replace the placeholder with your real key
```

**Option B — Sidebar:** Paste the key directly into the sidebar input at runtime.

Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey).

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set `GEMINI_API_KEY` as a **Secret** in the Streamlit Cloud dashboard (Settings → Secrets):

```toml
GEMINI_API_KEY = "your_key_here"
```

4. Set **Main file path** to `app.py` and deploy.

---

## Disclaimer

This application is for **informational purposes only**. It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for any health concerns.
