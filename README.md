# Loan Default Risk Engine

An end-to-end AI-powered credit risk assessment system that predicts
loan default probability and explains every decision using SHAP —
built with regulatory explainability (RBI compliance) in mind.

---

## Live Demo

Run locally using the setup steps below.
Interactive Streamlit app with 5 pages.

---

## Project Overview

Traditional credit models are black boxes. Banks and regulators need
to know **why** a loan was approved or rejected — not just the outcome.

This engine solves that by combining a high-performance XGBoost model
with SHAP explainability, making every prediction transparent and auditable.

---

## Results

| Metric | Score |
|---|---|
| Model | XGBoost |
| AUC-ROC | 0.7559 |
| Cross-Validation AUC | 0.7479 (+/- 0.0023) |
| Dataset Size | 307,511 applicants |
| Default Rate | 8.07% |
| Class Imbalance Handling | scale_pos_weight |

---

## Tech Stack

| Category | Tools |
|---|---|
| ML Models | XGBoost, LightGBM |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Plotly, Matplotlib, Seaborn |
| App Framework | Streamlit |
| Cloud Storage | AWS S3 (boto3) |
| Environment | Python 3.x, virtualenv |

---

## Project Structure
```
loan-default-risk-engine/
├── data/                            # Charts and visualizations
├── notebooks/
│   ├── 01_eda.ipynb                 # Initial data exploration
│   ├── 02_eda_deep.ipynb            # Deep EDA and business insights
│   ├── 03_preprocessing.ipynb      # Cleaning and feature engineering
│   ├── 04_model_building.ipynb     # XGBoost and LightGBM training
│   └── 05_shap_explainability.ipynb # SHAP analysis
├── src/
│   ├── s3_handler.py                # AWS S3 storage utility
│   └── upload_artifacts.py         # Artifact upload pipeline
├── app/
│   └── streamlit_app.py            # Interactive web application
├── models/
│   ├── xgb_model.pkl               # Trained XGBoost model
│   └── lgb_model.pkl               # Trained LightGBM model
├── .env.example                    # Environment variable template
├── requirements.txt                # Project dependencies
└── README.md
```

---

## Key Features

### 1. End-to-End ML Pipeline
- Data cleaning with professional imputation strategies
- Feature engineering — 7 domain-driven features created
- Winsorization for outlier handling at 1st and 99th percentile
- Stratified train-test split preserving class distribution

### 2. Dual Model Comparison
- XGBoost and LightGBM trained and evaluated
- 5-fold stratified cross validation
- Evaluation using AUC-ROC and Precision-Recall curves
- Class imbalance handled via scale_pos_weight

### 3. SHAP Explainability
- Global feature importance via summary plots
- Individual applicant explanations via waterfall plots
- Defaulter vs repayer side-by-side comparison
- Business-language interpretation of every prediction

### 4. Interactive Streamlit App
Five pages covering:
- **Home** — project overview and key metrics
- **Risk Predictor** — input applicant details, get instant risk score + SHAP explanation
- **Model Insights** — ROC curve, confusion matrix, score distribution
- **SHAP Dashboard** — global and individual applicant explanations
- **Portfolio Dashboard** — full portfolio risk breakdown with action recommendations

### 5. AWS S3 Integration
- Model artifacts and charts stored in S3
- Reusable boto3 utility with upload, download and list operations
- Production-ready architecture — switching from local simulation
  to real AWS requires changing only the client initialization

---

## Feature Engineering

| Feature | Formula | Business Meaning |
|---|---|---|
| CREDIT_INCOME_RATIO | AMT_CREDIT / AMT_INCOME | Loan burden relative to income |
| ANNUITY_INCOME_RATIO | AMT_ANNUITY / AMT_INCOME | Monthly repayment burden |
| CREDIT_GOODS_RATIO | AMT_CREDIT / AMT_GOODS_PRICE | Financing proportion |
| AGE_YEARS | DAYS_BIRTH / 365 | Applicant age in years |
| YEARS_EMPLOYED | DAYS_EMPLOYED / 365 | Employment stability |
| EMPLOYMENT_AGE_RATIO | YEARS_EMPLOYED / AGE_YEARS | Career maturity |
| INCOME_PER_PERSON | AMT_INCOME / CNT_FAM_MEMBERS | Per-capita household income |

---

## SHAP Explainability Example

Every prediction comes with a breakdown like this:
```
Applicant flagged as HIGH RISK (74.3% default probability)

Top contributing factors:
  EXT_SOURCE_3          -0.42   Low external credit score
  CREDIT_INCOME_RATIO   +0.31   Loan amount is 8x annual income
  ANNUITY_INCOME_RATIO  +0.24   EMI is 42% of monthly income
  YEARS_EMPLOYED        -0.18   Only 6 months employment history
  EXT_SOURCE_2          -0.15   Secondary credit score is low
```

This level of transparency is what RBI-compliant lending requires.

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/AvantiSavji/loan-default-risk-engine.git
cd loan-default-risk-engine
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
Download `application_train.csv` from
[Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk)
and place it in the `data/` folder.

### 5. Run notebooks in order
```
notebooks/01_eda.ipynb
notebooks/02_eda_deep.ipynb
notebooks/03_preprocessing.ipynb
notebooks/04_model_building.ipynb
notebooks/05_shap_explainability.ipynb
```

### 6. Launch the Streamlit app
```bash
cd app
streamlit run streamlit_app.py
```

---

## Business Impact

This project directly addresses a real problem in the Indian lending
industry — the need for explainable credit decisions under RBI guidelines.

The SHAP integration means every loan rejection can be justified with
specific, auditable reasons — protecting both the bank and the applicant.

---

## Dataset

| Detail | Info |
|---|---|
| Source | [Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk) |
| Size | 307,511 applications |
| Features | 122 original + 7 engineered |
| Target | Binary — 0 (Repaid), 1 (Defaulted) |
| Class Split | 91.93% Repaid / 8.07% Defaulted |

> Note: Raw data files are not included in this repository
> due to Kaggle terms of service. Download from the link above.

---

## Author

**Avanti Savji**
Pursuing Electronics & Telecommunication Engineering — VIIT Pune

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](your-linkedin-url)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/AvantiSavji)
