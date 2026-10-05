# AI-Powered Symptom Checker

An AI-powered medical triage application built with **Streamlit**, **Google Gemini 3.6 Flash** (online), and **locally trained scikit-learn / PyTorch models** (100% offline, no API key required).

---

## Features

- 🤖 **Gemini 3.6 Flash** AI backbone — fast, structured JSON triage responses
- 🧠 **Offline ML Diagnosis** — predict diseases from typed symptoms or uploaded images with no API key
- 🟢🟠🔴 **3-level triage** system (Self-Care / See a Doctor / Emergency)
- 🖼️ **Image-based diagnosis** — upload a skin/medical photo and get instant local predictions
- 💊 **Possible conditions** with confidence scores and likelihood ratings
- ✅ **Recommended actions** specific to your symptoms
- ⚠️ **Warning signs** to watch out for
- 📖 **Lifestyle Guide** — AI-generated prevention & habit guide for any health topic
- 🕑 **Session history** — review all checks in the current session

---

## Project Structure

```
symptom-checker/
├── app.py                          # Streamlit UI (3 pages)
├── prompt_engine.py                # Gemini prompt builder & response parser
├── requirements.txt                # Python dependencies
├── .env.example                    # API key template
├── README.md
└── ml_models/
    ├── __init__.py                 # Package init
    ├── ml_engine.py                # Offline inference engine (TextPredictor + ImagePredictor)
    ├── train_text_model.py         # Train RF+GB ensemble on symptom-disease data
    ├── train_image_model.py        # Fine-tune MobileNetV2 on medical images
    └── data/
        ├── README.md               # Dataset placement & training instructions
        ├── Training.csv            # ← place Kaggle dataset here
        ├── Testing.csv             # ← place Kaggle dataset here
        └── images/
            ├── scin/               # ← Google SCIN skin images
            ├── tcia/               # ← TCIA medical images (converted from DICOM)
            └── medimg/             # ← MedImg / CUI Lab images
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` and `torchvision` are required for the image classifier. For CPU-only machines, they install without a GPU driver. Installation may take a few minutes.

### 2. (Optional) Train offline ML models

The app runs fine without trained models — the ML Diagnosis page will display setup instructions until models are ready.

#### Text Disease Classifier

1. Download `Training.csv` and `Testing.csv` from  
   [https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)
2. Place both files in `ml_models/data/`
3. Run:

```bash
python ml_models/train_text_model.py
```

Expected output: `ml_models/text_model.pkl`, `symptom_columns.pkl`, `label_encoder.pkl`, `model_meta.json`

#### Image Classifier (MobileNetV2)

1. Download images from one or more of:
   - **Google SCIN** (skin conditions): [https://github.com/google-research-datasets/scin](https://github.com/google-research-datasets/scin)
   - **The Cancer Imaging Archive**: [https://www.cancerimagingarchive.net/](https://www.cancerimagingarchive.net/)
   - **MedImg / CUI Lab**: [https://www.cuilab.cn/medimg/](https://www.cuilab.cn/medimg/)
2. Organise images in ImageFolder layout:  
   `ml_models/data/images/<dataset>/<label>/<image>.jpg`
3. Run:

```bash
# CPU (slower)
python ml_models/train_image_model.py --epochs 20 --batch_size 16

# GPU (faster)
python ml_models/train_image_model.py --epochs 30 --batch_size 32 --device cuda
```

### 3. Set your Gemini API key (for AI Checker & Lifestyle Guide)

**Option A — `.env` file (recommended for local dev):**

```bash
cp .env.example .env
# Edit .env and replace the placeholder with your real key
```

**Option B — Sidebar:** Paste the key directly into the sidebar input at runtime.

> The **ML Diagnosis page works fully offline** — no API key needed.

Get a free Gemini key at [Google AI Studio](https://aistudio.google.com/app/apikey).

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## ML Model Details

| Model | Architecture | Dataset | Accuracy |
|-------|-------------|---------|----------|
| Text Disease Classifier | VotingClassifier (RandomForest + GradientBoosting) | Kaggle Disease Prediction | ~98% |
| Image Classifier | MobileNetV2 (fine-tuned, PyTorch) | SCIN / TCIA / MedImg | varies by dataset |

### Dataset Citations

- **Kaggle Disease Prediction:** kaushil268 — [Dataset Link](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)
- **UCI Heart Disease:** Janosi, A., Steinbrunn, W., Pfisterer, M., Detrano, R. — [Dataset Link](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **UCI Pima Diabetes:** Smith, J.W., et al. — [Dataset Link](https://archive.ics.uci.edu/dataset/34/diabetes)
- **Google SCIN:** Rotemberg, V., et al. (Google Research) — [GitHub](https://github.com/google-research-datasets/scin)
- **TCIA:** Clark, K., et al. — [https://www.cancerimagingarchive.net/](https://www.cancerimagingarchive.net/)
- **MedImg:** CUI Lab — [https://www.cuilab.cn/medimg/](https://www.cuilab.cn/medimg/)

---

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set `GEMINI_API_KEY` as a **Secret** in the Streamlit Cloud dashboard (Settings → Secrets):

```toml
GEMINI_API_KEY = "your_key_here"
```

4. Set **Main file path** to `app.py` and deploy.

> **Note:** Trained `.pkl` and `.pth` model files must be committed to the repo (or downloaded at startup via `st.cache_resource`) for the ML Diagnosis page to work on Cloud.

---

## Disclaimer

This application is for **informational purposes only**. The ML predictions are generated by locally trained models and do not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for any health concerns.
