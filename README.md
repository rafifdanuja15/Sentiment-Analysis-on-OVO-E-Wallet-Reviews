# 🔍 Sentiment Analysis — OVO App Reviews (Google Play Store)

> An end-to-end NLP pipeline for classifying user sentiment on the **OVO** mobile app, scraped from Google Play Store — covering data collection, auto-labeling, preprocessing, modeling, and inference.

---

## 📌 Overview

This project builds a three-class sentiment classification system (**Positive**, **Negative**, **Neutral**) on Indonesian-language reviews of the OVO digital payment app from Google Play Store. The pipeline is fully automated from raw data collection to a deployable inference module.

---

## 🗂️ Project Structure

```
sentiment_ovo/
│
├── 01_data_scraping.ipynb        # Scrape reviews from Google Play Store
├── 02_data_labeling.ipynb        # Automatic labeling via IndoRoBERTa
├── 03_data_preprocessing.ipynb   # Text cleaning & normalization
├── 04-data-modeling.ipynb        # Feature extraction & model training
├── 07_inference.ipynb            # Inference on new reviews
│
├── ulasan_raw_ovo.csv            # Raw scraped data
└── requirements.txt              # Dependency list
```

---

## ⚙️ Pipeline

```
Google Play Store
       │
       ▼
 [01] Data Scraping          → ulasan_raw_ovo.csv (~20,000 reviews)
       │
       ▼
 [02] Auto Labeling          → IndoRoBERTa (Positive / Negative / Neutral)
       │
       ▼
 [03] Preprocessing          → Case folding, slang normalization,
       │                        emoji handling, stopword removal,
       │                        tokenization, negation marking
       ▼
 [04] Modeling               → BoW / TF-IDF + SVM / MLP / LightGBM
       │
       ▼
 [07] Inference              → Predict sentiment on new review text
```

---

## 🛠️ Tech Stack

### Data Collection
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Google Play Scraper](https://img.shields.io/badge/google--play--scraper-1.2.7-green?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

### Auto Labeling
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Transformers-FFD21F?style=flat)

> Pretrained model: [`w11wo/indonesian-roberta-base-sentiment-classifier`](https://huggingface.co/w11wo/indonesian-roberta-base-sentiment-classifier)

### NLP Preprocessing
![NLTK](https://img.shields.io/badge/NLTK-3.9.4-blue?style=flat)
![Sastrawi](https://img.shields.io/badge/Sastrawi-1.0.1-orange?style=flat)
![emoji](https://img.shields.io/badge/emoji-2.15.0-yellow?style=flat)
![swifter](https://img.shields.io/badge/swifter-1.4.0-lightgrey?style=flat)

Preprocessing steps applied:
- Case folding
- Number normalization
- Emoji handling (convert/remove)
- Special character removal
- Indonesian slang normalization (informal lexicon)
- Tokenization
- Stopword removal
- Negation marking

### Feature Extraction & Modeling
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-9cf?style=flat)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

| Feature Extractor | Classifier |
|---|---|
| Bag-of-Words (BoW) | MLP Classifier |
| Bag-of-Words (BoW) | LightGBM |
| TF-IDF | SVM |

### Visualization
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.8-blue?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13.2-teal?style=flat)

### Model Serialization
![Joblib](https://img.shields.io/badge/Joblib-1.5.2-grey?style=flat)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/username/sentiment-ovo.git
cd sentiment-ovo
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline (in order)

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `01_data_scraping.ipynb` | Scrape ~20,000 OVO reviews from Google Play |
| 2 | `02_data_labeling.ipynb` | Auto-label reviews using IndoRoBERTa |
| 3 | `03_data_preprocessing.ipynb` | Clean and normalize text |
| 4 | `04-data-modeling.ipynb` | Train and evaluate classifiers |
| 5 | `07_inference.ipynb` | Run sentiment prediction on new input |

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Source | Google Play Store — OVO app (`ovo.id`) |
| Size | ~20,000 reviews |
| Language | Indonesian |
| Columns | `reviewId`, `content` |
| Labels | Positive, Negative, Neutral |

---

## 📋 Requirements

```
emoji==2.15.0
gensim==4.4.0
google_play_scraper==1.2.7
joblib==1.5.2
lightgbm==4.6.0
matplotlib==3.10.8
nltk==3.9.4
numpy==2.4.4
pandas==3.0.2
Sastrawi==1.0.1
scikit_learn==1.8.0
seaborn==0.13.2
swifter==1.4.0
torch==2.11.0
tqdm==4.67.3
transformers==5.6.0
```

---

## 📝 Notes

- **Stemming was intentionally skipped** — to preserve semantic context in BoW and TF-IDF representations, which are sensitive to word form variation.
- **No manual annotation** — labeling is fully automated using the IndoRoBERTa pretrained model.
- **Negation marking** is applied during preprocessing to retain negative sentiment context (e.g., "tidak bagus" ≠ "bagus").

---

## 👤 Author

**Muhammad Rafif Danuja**  
Statistics and Data Science — IPB University

---

## 📄 License

This project is for academic and research purposes.
