# MSIS 522 HW1 — Flaredown Food & Flare Prediction

**Author:** Hannah
**Course:** MSIS 522 — Data Science
**Date:** March 2026

## Overview

Can we predict whether someone with a chronic illness will have a **symptom flare** based on the **foods they eat**? This project uses the [Flaredown Autoimmune Symptom Tracker](https://www.kaggle.com/datasets/flaredown/flaredown-autoimmune-symptom-tracker) dataset (8M+ rows) to build a binary classification model.

As someone with **hypermobile Ehlers-Danlos Syndrome (hEDS)** and **POTS**, I'm personally motivated to understand how food choices relate to symptom flares.

## Key Results

- **Dataset:** 104,447 user-days with both food and symptom tracking
- **Target:** Flare (1) if max daily symptom severity ≥ 3 on 0–4 scale (68% flare rate)
- **Best Model:** XGBoost with GridSearchCV tuning
- **Features:** 104 total — 50 food items, 6 food categories, 20 treatments, weather, tags, demographics

## Project Structure

```
├── notebooks/hw1_analysis.ipynb   # Main notebook (Parts 1-3)
├── src/data_preprocessing.py      # Data loading & feature engineering
├── app/streamlit_app.py           # Streamlit app (Part 4)
├── models/                        # Saved trained models (joblib/keras)
├── data/                          # Processed parquet files
├── figures/                       # Saved visualizations
├── requirements.txt
└── README.md
```

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Preprocess Data
```bash
python src/data_preprocessing.py
```

### 2. Run Notebook
Open `notebooks/hw1_analysis.ipynb` in Jupyter and run all cells.

### 3. Run Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## Streamlit App Tabs

1. **Executive Summary** — Project overview, key findings, food recommendations
2. **Descriptive Analytics** — Interactive visualizations of food/flare patterns
3. **Model Performance** — Comparison table, ROC curves, confusion matrix
4. **Explainability & Prediction** — SHAP analysis + interactive flare predictor

## Deployed App

🔗 [Streamlit Cloud Link](https://msis522-hw1.streamlit.app)

## Models Trained

| Model | Method |
|-------|--------|
| Logistic Regression | Baseline, class_weight='balanced' |
| Decision Tree | GridSearchCV (max_depth, min_samples_leaf, criterion) |
| Random Forest | GridSearchCV (n_estimators, max_depth, min_samples_split) |
| XGBoost | GridSearchCV (n_estimators, max_depth, learning_rate, subsample) |
| MLP Neural Network | Keras/TensorFlow (128→64→1, early stopping) |

## Note on Large Files

`models/random_forest.joblib` exceeds GitHub's 100MB file size limit and is excluded from the repository. To regenerate it, run all cells in the notebook — the Random Forest model will be retrained and saved automatically. All other models are included in the repo.
