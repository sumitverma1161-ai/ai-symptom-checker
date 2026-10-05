"""
train_text_model.py
--------------------
Trains a Random Forest + Gradient Boosting ensemble classifier on the
Kaggle "Disease Prediction Using Machine Learning" dataset and the
UCI ML Repository Heart Disease / Diabetes datasets.

Dataset sources (download manually and place in ml_models/data/):
  - Kaggle: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
      → Training.csv   (133 symptom columns + prognosis label)
      → Testing.csv
  - UCI Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease
      → heart.csv  (processed.cleveland.data renamed to heart.csv)
  - UCI Diabetes (Pima Indians): https://archive.ics.uci.edu/dataset/34/diabetes
      → diabetes.csv

Usage:
    python ml_models/train_text_model.py

Outputs:
    ml_models/text_model.pkl       — trained scikit-learn Pipeline
    ml_models/symptom_columns.pkl  — ordered feature column names
    ml_models/label_encoder.pkl    — LabelEncoder for disease names
    ml_models/model_meta.json      — accuracy & dataset info
"""

import os
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────
# 1.  Load Kaggle disease-prediction dataset (primary — 133 symptoms)
# ─────────────────────────────────────────────────────────────────

def load_kaggle_dataset():
    train_path = DATA_DIR / "Training.csv"
    test_path = DATA_DIR / "Testing.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training.csv not found at {train_path}.\n"
            "Download from: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning\n"
            "and place Training.csv and Testing.csv in ml_models/data/"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Drop unnamed last column if present
    train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
    test_df = test_df.loc[:, ~test_df.columns.str.contains("^Unnamed")]

    df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"[Kaggle dataset] Loaded {len(df)} samples, {df.shape[1]-1} symptom features")
    return df, "prognosis"


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
