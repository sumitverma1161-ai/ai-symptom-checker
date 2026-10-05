# ml_models/data — Dataset Placement Guide

This directory holds the raw datasets used to train the offline ML models.
**Do not commit large dataset files to git** — add them locally before training.

---

## Text Model Datasets

### 1. Kaggle Disease Prediction (Primary — Required)
- **URL:** https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning  
- **Files to download:** `Training.csv`, `Testing.csv`  
- **Place at:** `ml_models/data/Training.csv` and `ml_models/data/Testing.csv`  
- **Description:** 4920 samples, 132 symptom binary features, 41 disease classes.

### 2. UCI Heart Disease (Optional)
- **URL:** https://archive.ics.uci.edu/dataset/45/heart+disease  
- **File:** `processed.cleveland.data` → rename to `heart.csv`  
- **Place at:** `ml_models/data/heart.csv`

### 3. UCI Pima Indians Diabetes (Optional)
- **URL:** https://archive.ics.uci.edu/dataset/34/diabetes  
- **File:** `diabetes.csv`  
- **Place at:** `ml_models/data/diabetes.csv`

---

## Image Model Datasets

Each image dataset must follow PyTorch **ImageFolder** layout:

```
ml_models/data/images/
  <dataset_name>/
    <class_label_1>/
      image001.jpg
      image002.png
      ...
    <class_label_2>/
      ...
```

### 1. Google SCIN (Skin Condition Image Network)
- **URL:** https://github.com/google-research-datasets/scin  
- **Place at:** `ml_models/data/images/scin/<condition_name>/<image>.jpg`  
- **Classes:** ~30 skin conditions (eczema, psoriasis, acne, melanoma, etc.)

### 2. The Cancer Imaging Archive (TCIA)
- **URL:** https://www.cancerimagingarchive.net/  
- **Note:** Images are in DICOM format — convert to JPEG/PNG first.  
  Install pydicom: `pip install pydicom`  
  Then use the provided conversion snippet in `train_image_model.py`.  
- **Place at:** `ml_models/data/images/tcia/<diagnosis_label>/<image>.jpg`

### 3. MedImg / CUI Lab
- **URL:** https://www.cuilab.cn/medimg/  
- **Place at:** `ml_models/data/images/medimg/<condition_name>/<image>.jpg`

---

## After Placing Data, Train Models

```bash
# Train the text disease classifier
python ml_models/train_text_model.py

# Train the image classifier (CPU — slow)
python ml_models/train_image_model.py --epochs 20 --batch_size 16

# Train with GPU (much faster)
python ml_models/train_image_model.py --epochs 30 --batch_size 32 --device cuda
```

Trained model artifacts will appear in `ml_models/`:
- `text_model.pkl` — VotingClassifier (RandomForest + GradientBoosting)
- `symptom_columns.pkl` — Feature column order
- `label_encoder.pkl` — Disease name encoder
- `symptom_keyword_map.pkl` — Natural language → symptom column mapping
- `model_meta.json` — Model accuracy & metadata
- `image_model.pth` — Fine-tuned MobileNetV2 weights
- `image_classes.pkl` — Class names
- `image_model_meta.json` — Training metadata
