"""
ml_engine.py
-------------
Offline ML inference engine for the Symptom Checker app.

Provides two predictors:
  1. TextPredictor  — maps free-text symptom descriptions to diseases
                      using the trained RandomForest+GB ensemble (.pkl)
  2. ImagePredictor — classifies uploaded medical/skin images
                      using the fine-tuned MobileNetV2 (.pth)

Both predictors gracefully degrade when model files are missing
(they return informative placeholder results instead of crashing).
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Optional

# ── Lazy imports (only loaded when actually used) ─────────────────
_np = None
_joblib = None
_torch = None
_transforms = None
_models = None
_Image = None

MODEL_DIR = Path(__file__).parent


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
        required = [
            MODEL_DIR / "text_model.pkl",
            MODEL_DIR / "symptom_columns.pkl",
            MODEL_DIR / "label_encoder.pkl",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            self._load_error = (
                "Text model files not found. Run:\n"
                "  python ml_models/train_text_model.py\n"
                "after placing Training.csv in ml_models/data/\n"
                f"Missing: {', '.join(missing)}"
            )
            return

        try:
            self._model = jl.load(MODEL_DIR / "text_model.pkl")
            self._columns = jl.load(MODEL_DIR / "symptom_columns.pkl")
            self._le = jl.load(MODEL_DIR / "label_encoder.pkl")
            kw_path = MODEL_DIR / "symptom_keyword_map.pkl"
            if kw_path.exists():
                self._kw_map = jl.load(kw_path)
            self._meta = _load_meta(MODEL_DIR / "model_meta.json")
            self._loaded = True
        except Exception as exc:
            self._load_error = f"Failed to load text model: {exc}"

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
