# 🏥 Hospital Patient Analytics & Outcome Prediction

A data analytics and machine learning web application built with Python and Streamlit to analyze inpatient healthcare records, examine clinical biomarkers, audit hospital treatment costs, and predict patient discharge outcomes (**Recovered**, **Complicated**, or **Deceased**).

---

## 📌 Project Overview

In healthcare management, understanding inpatient recovery patterns and identifying high-risk cases early can significantly improve hospital resource planning and clinical decision-making. 

This project covers the full end-to-end data science lifecycle:
1. **Data Pipeline & Preprocessing**: Merging raw patient admissions, diagnostic categories, outcome statuses, and laboratory test results into a structured dataset.
2. **Exploratory Data Analytics (EDA)**: Interactive visualizations for patient demographics, clinical biomarker correlations, length of stay, and treatment expenses.
3. **Machine Learning Modeling**: Training and comparing multiple classifiers (*Random Forest, Gradient Boosting, Logistic Regression, Decision Tree*) to predict patient outcomes.
4. **Interactive Web App**: A multi-page Streamlit application featuring dark/light mode, real-time lab range checks, preset profiles, and batch CSV predictions.

---

## ✨ Features

### 📊 1. Analytics Dashboard
- **Patient Demographics**: Age distributions with box plots, gender breakdowns, and outcome ratios.
- **Biomarker Correlations**: Interactive correlation matrix across all numeric clinical parameters.
- **Clinical Variations**: Boxplots showing how blood pressure, blood sugar, creatinine, and hemoglobin levels differ by outcome.
- **Operations & Financials**: Treatment costs and average length of stay across diagnoses.
- **Searchable Records**: Inpatient data explorer with instant search and filtered CSV export.

### 🤖 2. Outcome Prediction & Clinical Decision Support
- **Single Patient Prediction**: Input patient vitals and diagnosis to predict the likely discharge outcome with confidence scores.
- **Clinical Reference Ranges**: Real-time evaluation badges (*Normal, Prehypertension, High, Deficient*) across all 6 lab measurements:
  - Blood Pressure (`mmHg`)
  - Blood Sugar (`mg/dL`)
  - Total Cholesterol (`mg/dL`)
  - Serum Creatinine (`mg/dL`)
  - Hemoglobin (`g/dL`)
  - Vitamin D (`ng/mL`)
- **Quick Preset Profiles**: Load realistic patient scenarios (*Healthy Routine*, *Cardiovascular High-Risk*, *Diabetic Renal Risk*) with one click.
- **Batch CSV Upload**: Upload bulk patient records to get instantaneous predictions, triage charts, and downloadable enriched CSVs.
- **Downloadable Clinical Summary**: Export clean `.txt` summary reports for individual predictions.

### 📈 3. Model Performance & Benchmarking
- **Champion Model Evaluation**: Accuracy, per-class precision/recall/F1-scores, and support metrics.
- **Interactive Confusion Matrix**: Heatmap with actual vs. predicted counts and hover details.
- **Multi-Algorithm Benchmark**: Comparative metrics across 4 machine learning models.
- **Feature Importance**: Ranked clinical features driving outcome predictions (Age, Blood Pressure, Hemoglobin, and Blood Sugar are top drivers).

### 🌓 4. UI & Theme System
- One-click toggle between **Dark Mode** and **Light Mode**.
- Clean glassmorphism styling, responsive layout, and customized charts matching the active theme.

---

## 🔬 Machine Learning Results

We trained models using an **80:20 stratified split** (4,000 training samples, 1,000 test samples) on 9 features (*Age, Gender, Diagnosis, Blood Pressure, Blood Sugar, Cholesterol, Creatinine, Hemoglobin, Vitamin D*).

### Model Comparison Table

| Algorithm | Test Accuracy | Macro F1-Score | Weighted F1-Score | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **83.30%** | **0.7431** | **0.8317** | **0.83s** |
| **Logistic Regression** | 84.60% | 0.7679 | 0.8465 | 0.11s |
| **Gradient Boosting** | 82.90% | 0.7412 | 0.8285 | 5.10s |
| **Decision Tree** | 80.00% | 0.7091 | 0.8004 | 0.04s |

*Random Forest was chosen as the champion pipeline for its balanced multi-class performance and ability to model complex nonlinear interactions between age and lab vitals.*

### Class Breakdown (Random Forest)
- **Recovered**: Precision: `91.8%` | Recall: `92.9%` | F1: `92.4%` (650 test patients)
- **Complicated**: Precision: `66.0%` | Recall: `68.4%` | F1: `67.2%` (250 test patients)
- **Deceased**: Precision: `69.9%` | Recall: `58.0%` | F1: `63.4%` (100 test patients)

---

## 🗂 Project Structure

```
├── data/
│   ├── raw/                       # Raw CSVs (patients, diagnoses, outcomes, labs)
│   └── processed/
│       └── healthcare_cleaned.csv # Cleaned & merged 5,000-record dataset
├── models/
│   ├── best_model.pkl             # Trained scikit-learn pipeline
│   ├── model_metadata.json        # Dynamic evaluation metrics & hyperparameters
│   ├── model_comparison.csv       # Multi-algorithm benchmark comparison
│   ├── feature_importance.csv     # Ranked feature importances
│   ├── confusion_matrix.png       # Confusion matrix plot
│   └── classification_report.txt  # Classification report text
├── notebooks/
│   ├── EDA.ipynb                  # Exploratory analysis notebook
│   └── ML_process.ipynb           # Model development & testing notebook
├── src/
│   ├── data_preprocessing.py      # Automated data cleaning & merging ETL
│   └── train_model3.py            # Model training & benchmarking pipeline
├── streamlit_app/
│   ├── app.py                     # Main dashboard home page
│   ├── assets/
│   │   └── style.css              # Custom CSS styling (Dark/Light mode support)
│   ├── pages/
│   │   ├── About.py               # About & clinical reference guide
│   │   ├── Dashboard.py           # Inpatient analytics dashboard
│   │   ├── Model_Performance.py   # Model evaluation & benchmark suite
│   │   └── Predict.py             # Single & batch outcome prediction engine
│   └── utils/
│       └── theme.py               # Theme engine, dynamic loaders & clinical helpers
├── tests/
│   ├── test_data_pipeline.py      # Dataset validation tests
│   ├── test_model_pipeline.py     # Model inference tests
│   └── test_app_helpers.py        # Clinical status helper tests
├── requirements.txt               # Project dependencies
├── run_app.py                     # Application startup script
└── README.md                      # Project documentation
```

---

## ⚡ Quick Start

### 1. Clone the repository & set up environment
```bash
git clone https://github.com/the-ayush-ch0udhary/Healthcare-Data-Analytics-and-ML-Project.git
cd Healthcare-Data-Analytics-and-ML-Project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Run the data pipeline and training
If you want to re-process data or re-train models from scratch:
```bash
# Ingest and clean raw data
python src/data_preprocessing.py

# Train models and generate benchmarks
python src/train_model3.py
```

### 4. Run tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 5. Start the web application
```bash
python run_app.py
```
Or directly with Streamlit:
```bash
streamlit run streamlit_app/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🛠 Tech Stack

- **Language**: Python 3.10+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, Joblib
- **Web App & Visualization**: Streamlit, Plotly Express, Matplotlib, Seaborn
- **Testing**: Unittest

---

## 👥 Authors

- **Ayush** — System Architecture, ML Pipelines, Streamlit Dashboard & UI Design
- **Moon** — Data Preprocessing, Exploratory Analysis, QA & Documentation

---

## ⚠️ Note
*This project uses synthetic patient data and is built for educational and analytical purposes. Predictions should not be used as clinical medical advice.*
