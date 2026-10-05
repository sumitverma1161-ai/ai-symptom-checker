# ml_models/data — Dataset Guide

This directory is used as a local cache for datasets.
**Do not commit large dataset files to git.**

---

## Text Model Dataset — Auto-downloaded via kagglehub ✅

The training script uses **kagglehub** to download the dataset automatically.
No manual CSV placement is needed.

```python
import kagglehub

# Downloads and caches the dataset locally
path = kagglehub.dataset_download("kaushil268/disease-prediction-using-machine-learning")
print("Path to dataset files:", path)
```

**Dataset:** https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
**Description:** 4920 samples · 132 symptom binary features · 41 disease classes

### One-time Kaggle authentication

kagglehub needs your Kaggle credentials the first time:

**Option A — `kaggle.json` file (recommended):**
1. Go to https://www.kaggle.com/settings → API → **Create New Token**
2. Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`

**Option B — Environment variables:**
```bash
set KAGGLE_USERNAME=your_username
set KAGGLE_KEY=your_api_key
```

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
