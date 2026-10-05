"""
train_text_model.py
--------------------
Trains a Random Forest + Gradient Boosting ensemble classifier on the
Kaggle "Disease Prediction Using Machine Learning" dataset.

The dataset is downloaded automatically via kagglehub — no manual
CSV placement required. kagglehub caches downloads locally so
subsequent runs are instant.

  Dataset: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning

Usage:
    python ml_models/train_text_model.py

Requirements:
    pip install kagglehub
    # Authenticate once: kaggle.json in ~/.kaggle/ OR set env vars:
    #   KAGGLE_USERNAME=<your_username>
    #   KAGGLE_KEY=<your_api_key>

Outputs:
    ml_models/text_model.pkl          — trained VotingClassifier (RF + GB)
    ml_models/symptom_columns.pkl     — ordered feature column names
    ml_models/label_encoder.pkl       — LabelEncoder for disease names
    ml_models/symptom_keyword_map.pkl — natural-language → column mapping
    ml_models/model_meta.json         — accuracy & dataset info
"""

import os
import json
import warnings
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────
# 1.  Download & load Kaggle disease-prediction dataset via kagglehub
# ─────────────────────────────────────────────────────────────────

def load_kaggle_dataset():
    """
    Downloads the dataset with kagglehub (cached after first run) and
    returns a combined DataFrame plus the label column name.
    """
    print("[kagglehub] Downloading dataset kaushil268/disease-prediction-using-machine-learning …")
    dataset_path = kagglehub.dataset_download("kaushil268/disease-prediction-using-machine-learning")
    print(f"[kagglehub] Path to dataset files: {dataset_path}")

    dataset_path = Path(dataset_path)

    # Locate Training.csv / Testing.csv inside the downloaded folder
    train_path = _find_csv(dataset_path, "Training")
    test_path  = _find_csv(dataset_path, "Testing")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Drop any unnamed trailing columns
    train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
    test_df  = test_df.loc[:,  ~test_df.columns.str.contains("^Unnamed")]

    df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"[dataset] Loaded {len(df)} samples, {df.shape[1]-1} symptom features")
    return df, "prognosis"


def _find_csv(base: Path, stem_prefix: str) -> Path:
    """Recursively find the first CSV whose stem starts with stem_prefix."""
    matches = sorted(base.rglob(f"{stem_prefix}*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{stem_prefix}*.csv' inside {base}.\n"
            "Make sure the kagglehub download completed successfully."
        )
    return matches[0]


# ─────────────────────────────────────────────────────────────────
# 2.  Build a free-text → symptom-vector helper
#     (used at inference time when user types natural language)
# ─────────────────────────────────────────────────────────────────

def build_symptom_keyword_map(symptom_columns: list[str]) -> dict[str, str]:
    """Map human-readable phrases to canonical column names."""
    mapping = {}
    for col in symptom_columns:
        # e.g. "high_fever" → ["high fever", "high_fever", "highfever"]
        readable = col.replace("_", " ")
        mapping[readable] = col
        mapping[col] = col
        mapping[col.replace("_", "")] = col
    return mapping


# ─────────────────────────────────────────────────────────────────
# 3.  Main training routine
# ─────────────────────────────────────────────────────────────────

def train():
    os.makedirs(DATA_DIR, exist_ok=True)

    df, label_col = load_kaggle_dataset()

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    X = df.drop(columns=[label_col]).astype(int)
    symptom_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # ── Ensemble: RF + GB voting ───────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    gb = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
    )
    ensemble = VotingClassifier(estimators=[("rf", rf), ("gb", gb)], voting="soft", n_jobs=-1)

    print("Training ensemble (RandomForest + GradientBoosting)…")
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    # Cross-validation
    cv_scores = cross_val_score(ensemble, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Persist artifacts ─────────────────────────────────────
    joblib.dump(ensemble, OUT_DIR / "text_model.pkl")
    joblib.dump(symptom_columns, OUT_DIR / "symptom_columns.pkl")
    joblib.dump(le, OUT_DIR / "label_encoder.pkl")

    kw_map = build_symptom_keyword_map(symptom_columns)
    joblib.dump(kw_map, OUT_DIR / "symptom_keyword_map.pkl")

    meta = {
        "model_type": "VotingClassifier(RF+GB)",
        "num_classes": int(len(le.classes_)),
        "classes": list(le.classes_),
        "num_symptoms": len(symptom_columns),
        "test_accuracy": round(float(acc), 4),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "dataset": "Kaggle Disease Prediction (kaushil268)",
        "dataset_url": "https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning",
    }
    with open(OUT_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved artifacts to {OUT_DIR}/")
    print(f"  text_model.pkl       ({len(le.classes_)} disease classes)")
    print(f"  symptom_columns.pkl  ({len(symptom_columns)} symptom features)")
    print(f"  label_encoder.pkl")
    print(f"  symptom_keyword_map.pkl")
    print(f"  model_meta.json")

    # Print classification summary
    print("\n" + classification_report(y_test, y_pred, target_names=le.classes_))


if __name__ == "__main__":
    train()
