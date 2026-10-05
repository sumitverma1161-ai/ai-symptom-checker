"""
ml_engine.py
-------------
Offline ML inference engine for the Symptom Checker app.

Provides two predictors:
  1. TextPredictor  — maps free-text symptom descriptions to diseases.
                      Loads pre-trained .pkl files if they exist, OR
                      auto-trains from the Kaggle dataset via kagglehub
                      (works on Streamlit Cloud — no manual CSV needed).
  2. ImagePredictor — classifies uploaded medical/skin images
                      using the fine-tuned MobileNetV2 (.pth)

Both predictors gracefully degrade when model files are missing
(they return informative placeholder results instead of crashing).

Streamlit Cloud notes:
  - Model artifacts are cached in /tmp/ml_models/ (writable on Cloud).
  - Set KAGGLE_USERNAME and KAGGLE_KEY in Streamlit Secrets to enable
    auto-training of the text model on first load.
"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# ── Lazy imports (only loaded when actually used) ─────────────────
_np = None
_joblib = None
_torch = None
_transforms = None
_models = None
_Image = None

MODEL_DIR = Path(__file__).parent

# On Streamlit Cloud the repo is read-only; use /tmp for trained artifacts
_TMP_MODEL_DIR = Path(tempfile.gettempdir()) / "ml_models"
_TMP_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _model_path(filename: str) -> Path:
    """
    Return the path for a model artifact.
    Prefer the repo-local path (exists when committed or trained locally),
    fall back to the /tmp cache path (used on Streamlit Cloud).
    """
    local = MODEL_DIR / filename
    if local.exists():
        return local
    return _TMP_MODEL_DIR / filename


def _load_numpy():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def _load_joblib():
    global _joblib
    if _joblib is None:
        import joblib
        _joblib = joblib
    return _joblib


def _load_torch():
    global _torch, _transforms, _models
    if _torch is None:
        import torch
        from torchvision import transforms, models
        _torch = torch
        _transforms = transforms
        _models = models
    return _torch, _transforms, _models


def _load_pil():
    global _Image
    if _Image is None:
        from PIL import Image
        _Image = Image
    return _Image


# ─────────────────────────────────────────────────────────────────
# Helper: load JSON meta file
# ─────────────────────────────────────────────────────────────────

def _load_meta(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────
# Kaggle credential injection (Streamlit Cloud secrets → env vars)
# ─────────────────────────────────────────────────────────────────

def _inject_kaggle_credentials():
    """
    Streamlit Cloud stores secrets as st.secrets.
    kagglehub reads KAGGLE_USERNAME / KAGGLE_KEY from env vars.
    This function bridges the two — safe to call even outside Streamlit.
    """
    # Already set — nothing to do
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    try:
        import streamlit as st
        username = st.secrets.get("KAGGLE_USERNAME", "")
        key      = st.secrets.get("KAGGLE_KEY", "")
        if username and key:
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"]      = key
    except Exception:
        pass  # Not running inside Streamlit, or secrets not configured


# ─────────────────────────────────────────────────────────────────
# In-process training (used when .pkl files are absent)
# ─────────────────────────────────────────────────────────────────

def _train_text_model_inprocess(out_dir: Path, jl) -> dict:
    """
    Download the Kaggle dataset via kagglehub, train the RF+GB ensemble
    entirely in-process, persist artifacts to out_dir, and return them.
    """
    import kagglehub
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    np = _load_numpy()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Download dataset ──────────────────────────────────────
    dataset_path = Path(
        kagglehub.dataset_download("kaushil268/disease-prediction-using-machine-learning")
    )

    def _find_csv(base: Path, prefix: str) -> Path:
        matches = sorted(base.rglob(f"{prefix}*.csv"))
        if not matches:
            raise FileNotFoundError(f"'{prefix}*.csv' not found inside {base}")
        return matches[0]

    train_df = pd.read_csv(_find_csv(dataset_path, "Training"))
    test_df  = pd.read_csv(_find_csv(dataset_path, "Testing"))

    for df in (train_df, test_df):
        df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")],
                inplace=True, errors="ignore")

    df = pd.concat([train_df, test_df], ignore_index=True)

    # ── 2. Encode labels ─────────────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(df["prognosis"])
    X  = df.drop(columns=["prognosis"]).astype(int)
    symptom_columns = list(X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # ── 3. Train ensemble (lighter settings for cloud speed) ─────
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=42,
        n_jobs=-1, class_weight="balanced"
    )
    gb = GradientBoostingClassifier(
        n_estimators=80, learning_rate=0.1, max_depth=5, random_state=42
    )
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)], voting="soft", n_jobs=-1
    )
    ensemble.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, ensemble.predict(X_te))

    # ── 4. Build keyword map ──────────────────────────────────────
    kw_map: dict[str, str] = {}
    for col in symptom_columns:
        readable = col.replace("_", " ")
        kw_map[readable] = col
        kw_map[col]      = col
        kw_map[col.replace("_", "")] = col

    # ── 5. Persist to out_dir ─────────────────────────────────────
    jl.dump(ensemble,        out_dir / "text_model.pkl")
    jl.dump(symptom_columns, out_dir / "symptom_columns.pkl")
    jl.dump(le,              out_dir / "label_encoder.pkl")
    jl.dump(kw_map,          out_dir / "symptom_keyword_map.pkl")

    meta = {
        "model_type": "VotingClassifier(RF+GB)",
        "num_classes": int(len(le.classes_)),
        "classes": list(le.classes_),
        "num_symptoms": len(symptom_columns),
        "test_accuracy": round(float(acc), 4),
        "dataset": "Kaggle Disease Prediction (kaushil268)",
        "dataset_url": "https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning",
        "trained_at_runtime": True,
    }
    with open(out_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "model":   ensemble,
        "columns": symptom_columns,
        "le":      le,
        "kw_map":  kw_map,
        "meta":    meta,
    }


# ─────────────────────────────────────────────────────────────────
# TEXT PREDICTOR
# ─────────────────────────────────────────────────────────────────

class TextPredictor:
    """
    Predicts likely diseases from a free-text symptom description.

    Workflow:
      1. Tokenise the input text into individual words / phrases.
      2. Match tokens against the known symptom keyword map.
      3. Build a binary feature vector over all 132 symptom columns.
      4. Run the trained VotingClassifier to get class probabilities.
      5. Return top-N predictions with confidence scores.
    """

    _instance: Optional["TextPredictor"] = None

    def __init__(self):
        self._model = None
        self._columns: list[str] = []
        self._le = None
        self._kw_map: dict[str, str] = {}
        self._loaded = False
        self._load_error: Optional[str] = None
        self._meta: dict = {}

    @classmethod
    def get(cls) -> "TextPredictor":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._try_load()
        return cls._instance

    def _try_load(self):
        jl = _load_joblib()

        model_pkl   = _model_path("text_model.pkl")
        cols_pkl    = _model_path("symptom_columns.pkl")
        le_pkl      = _model_path("label_encoder.pkl")
        kw_pkl      = _model_path("symptom_keyword_map.pkl")
        meta_json   = _model_path("model_meta.json")

        # ── If all artifacts exist, load them directly ─────────
        if model_pkl.exists() and cols_pkl.exists() and le_pkl.exists():
            try:
                self._model   = jl.load(model_pkl)
                self._columns = jl.load(cols_pkl)
                self._le      = jl.load(le_pkl)
                if kw_pkl.exists():
                    self._kw_map = jl.load(kw_pkl)
                self._meta = _load_meta(meta_json)
                self._loaded = True
            except Exception as exc:
                self._load_error = f"Failed to load text model: {exc}"
            return

        # ── Artifacts missing — attempt auto-training via kagglehub ──
        # Inject Kaggle credentials from environment / Streamlit secrets
        _inject_kaggle_credentials()

        try:
            import kagglehub
        except ImportError:
            self._load_error = (
                "Text model not found and kagglehub is not installed.\n"
                "Run:  pip install kagglehub\n"
                "Then set KAGGLE_USERNAME and KAGGLE_KEY in your environment\n"
                "or in Streamlit Secrets, and restart the app."
            )
            return

        try:
            print("[ml_engine] Auto-training text model via kagglehub …")
            _artifacts = _train_text_model_inprocess(
                out_dir=_TMP_MODEL_DIR, jl=jl
            )
            self._model   = _artifacts["model"]
            self._columns = _artifacts["columns"]
            self._le      = _artifacts["le"]
            self._kw_map  = _artifacts["kw_map"]
            self._meta    = _artifacts["meta"]
            self._loaded  = True
            print("[ml_engine] Text model ready.")
        except Exception as exc:
            self._load_error = (
                f"Auto-training failed: {exc}\n\n"
                "To fix on Streamlit Cloud:\n"
                "  1. Go to App Settings → Secrets\n"
                "  2. Add:  KAGGLE_USERNAME = \"your_username\"\n"
                "           KAGGLE_KEY = \"your_api_key\"\n"
                "  3. Reboot the app.\n\n"
                "To fix locally:\n"
                "  Place Training.csv in ml_models/data/ then run:\n"
                "  python ml_models/train_text_model.py"
            )

    # ── Public API ───────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def meta(self) -> dict:
        return self._meta

    def extract_symptoms(self, text: str) -> list[str]:
        """
        Return the list of known symptom column names detected in text.
        Handles multi-word phrases and underscore-separated tokens.
        """
        text_lower = text.lower()
        # Normalise punctuation
        text_lower = re.sub(r"[,;./\\()\[\]]", " ", text_lower)

        matched: set[str] = set()

        # 1. Direct phrase matching (longest first to avoid substring conflicts)
        sorted_phrases = sorted(self._kw_map.keys(), key=len, reverse=True)
        for phrase in sorted_phrases:
            if phrase in text_lower:
                matched.add(self._kw_map[phrase])

        # 2. Individual word fallback
        words = re.split(r"\s+", text_lower)
        for word in words:
            if word in self._kw_map:
                matched.add(self._kw_map[word])

        return list(matched)

    def build_feature_vector(self, symptom_cols: list[str]):
        """Build a binary 1×N feature array from matched symptom column names."""
        np = _load_numpy()
        vec = np.zeros((1, len(self._columns)), dtype=int)
        for col in symptom_cols:
            if col in self._columns:
                vec[0, self._columns.index(col)] = 1
        return vec

    def predict(self, text: str, top_n: int = 5) -> dict:
        """
        Run inference on symptom text.

        Returns:
            {
                "matched_symptoms": [...],
                "predictions": [
                    {"disease": "Malaria", "confidence": 0.87, "rank": 1},
                    ...
                ],
                "num_symptoms_matched": int,
                "model_info": {...}
            }
        """
        if not self._loaded:
            return {
                "error": self._load_error,
                "matched_symptoms": [],
                "predictions": [],
                "num_symptoms_matched": 0,
                "model_info": {},
            }

        matched = self.extract_symptoms(text)
        vec = self.build_feature_vector(matched)

        # Get probability estimates
        np = _load_numpy()
        proba = self._model.predict_proba(vec)[0]  # shape (n_classes,)
        top_indices = np.argsort(proba)[::-1][:top_n]

        predictions = [
            {
                "disease": self._le.inverse_transform([idx])[0],
                "confidence": round(float(proba[idx]), 4),
                "rank": i + 1,
            }
            for i, idx in enumerate(top_indices)
            if proba[idx] > 0.01  # suppress near-zero noise
        ]

        return {
            "matched_symptoms": [s.replace("_", " ").title() for s in matched],
            "predictions": predictions,
            "num_symptoms_matched": len(matched),
            "model_info": {
                "type": self._meta.get("model_type", "Ensemble"),
                "accuracy": self._meta.get("test_accuracy", "N/A"),
                "num_classes": self._meta.get("num_classes", len(self._le.classes_)),
                "dataset": self._meta.get("dataset", "Kaggle Disease Prediction"),
                "dataset_url": self._meta.get("dataset_url", ""),
            },
        }


# ─────────────────────────────────────────────────────────────────
# IMAGE PREDICTOR
# ─────────────────────────────────────────────────────────────────

_EVAL_TRANSFORM = None

def _get_eval_transform():
    global _EVAL_TRANSFORM
    if _EVAL_TRANSFORM is None:
        _, _tf, _ = _load_torch()
        _EVAL_TRANSFORM = _tf.Compose([
            _tf.Resize((224, 224)),
            _tf.ToTensor(),
            _tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _EVAL_TRANSFORM


class ImagePredictor:
    """
    Classifies a medical/skin image using the fine-tuned MobileNetV2.

    Accepts: PIL.Image, bytes, or a file-like object.
    Returns top-N predicted conditions with confidence scores.
    """

    _instance: Optional["ImagePredictor"] = None

    def __init__(self):
        self._model = None
        self._classes: list[str] = []
        self._device = None
        self._loaded = False
        self._load_error: Optional[str] = None
        self._meta: dict = {}

    @classmethod
    def get(cls) -> "ImagePredictor":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._try_load()
        return cls._instance

    def _try_load(self):
        model_path = MODEL_DIR / "image_model.pth"
        classes_path = MODEL_DIR / "image_classes.pkl"

        missing = [str(p) for p in [model_path, classes_path] if not p.exists()]
        if missing:
            self._load_error = (
                "Image model files not found. Run:\n"
                "  python ml_models/train_image_model.py\n"
                "after placing image datasets in ml_models/data/images/\n"
                f"Missing: {', '.join(missing)}"
            )
            return

        try:
            torch, _, _mdl = _load_torch()
            jl = _load_joblib()
            import pickle

            with open(classes_path, "rb") as f:
                self._classes = pickle.load(f)

            num_classes = len(self._classes)

            # Rebuild architecture
            from torchvision.models import MobileNet_V2_Weights
            model = _mdl.mobilenet_v2(weights=None)
            in_features = model.classifier[1].in_features
            import torch.nn as nn
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes),
            )

            self._device = torch.device("cpu")
            state = torch.load(model_path, map_location=self._device)
            model.load_state_dict(state)
            model.eval()
            self._model = model
            self._meta = _load_meta(MODEL_DIR / "image_model_meta.json")
            self._loaded = True

        except Exception as exc:
            self._load_error = f"Failed to load image model: {exc}"

    # ── Public API ───────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def meta(self) -> dict:
        return self._meta

    def predict(self, image_data, top_n: int = 5) -> dict:
        """
        Classify an image.

        Args:
            image_data: bytes | file-like | PIL.Image
            top_n: number of top predictions to return

        Returns:
            {
                "predictions": [
                    {"condition": "Eczema", "confidence": 0.72, "rank": 1},
                    ...
                ],
                "model_info": {...}
            }
        """
        if not self._loaded:
            return {
                "error": self._load_error,
                "predictions": [],
                "model_info": {},
            }

        try:
            torch, _, _ = _load_torch()
            Image = _load_pil()
            transform = _get_eval_transform()

            # Normalise input to PIL.Image
            if isinstance(image_data, bytes):
                img = Image.open(io.BytesIO(image_data)).convert("RGB")
            elif hasattr(image_data, "read"):
                img = Image.open(image_data).convert("RGB")
            else:
                img = image_data.convert("RGB")

            tensor = transform(img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            np = _load_numpy()
            top_indices = np.argsort(probs)[::-1][:top_n]
            predictions = [
                {
                    "condition": self._classes[idx],
                    "confidence": round(float(probs[idx]), 4),
                    "rank": i + 1,
                }
                for i, idx in enumerate(top_indices)
                if probs[idx] > 0.01
            ]

            return {
                "predictions": predictions,
                "model_info": {
                    "type": self._meta.get("model_type", "MobileNetV2"),
                    "accuracy": self._meta.get("best_val_accuracy", "N/A"),
                    "num_classes": len(self._classes),
                    "datasets": self._meta.get("datasets", []),
                },
            }

        except Exception as exc:
            return {
                "error": f"Image inference failed: {exc}",
                "predictions": [],
                "model_info": {},
            }


# ─────────────────────────────────────────────────────────────────
# Convenience functions (used by app.py)
# ─────────────────────────────────────────────────────────────────

def predict_from_text(symptom_text: str, top_n: int = 5) -> dict:
    """Run text-based disease prediction. Returns result dict."""
    return TextPredictor.get().predict(symptom_text, top_n=top_n)


def predict_from_image(image_data, top_n: int = 5) -> dict:
    """Run image-based condition classification. Returns result dict."""
    return ImagePredictor.get().predict(image_data, top_n=top_n)


def get_model_status() -> dict:
    """Return load status for both models (used by sidebar info panel)."""
    tp = TextPredictor.get()
    ip = ImagePredictor.get()
    return {
        "text_model": {
            "ready": tp.is_ready,
            "error": tp.load_error,
            "meta": tp.meta,
        },
        "image_model": {
            "ready": ip.is_ready,
            "error": ip.load_error,
            "meta": ip.meta,
        },
    }
