# 🏥 Hospital Patient Analytics

A healthcare analytics and machine learning project that analyzes patient records, identifies clinical patterns, explores treatment outcomes, and predicts patient outcomes using **Random Forest Classification**.

The project combines **Data Analytics, Exploratory Data Analysis, Machine Learning, and an interactive Streamlit dashboard** into a single healthcare intelligence platform.

---

## 📌 Project Overview

Healthcare organizations generate large amounts of patient data containing information about demographics, diagnoses, laboratory results, treatment costs, and patient outcomes.

This project uses that data to answer practical questions such as:

* How are patients distributed across different outcomes?
* Which diagnoses are most common?
* What patterns can be observed in patient health indicators?
* How do laboratory values vary across patient groups?
* Which factors contribute most to predicting patient outcomes?
* Can Machine Learning help predict whether a patient will recover, develop complications, or become deceased?

The project processes raw healthcare datasets, performs data cleaning and exploratory analysis, trains a classification model, evaluates its performance, and presents the results through an interactive Streamlit application.

---

## 🎯 Objectives

The main objectives of this project are:

1. Clean and prepare healthcare datasets for analysis.
2. Combine relevant patient, diagnosis, laboratory, and outcome information.
3. Perform Exploratory Data Analysis (EDA).
4. Identify important healthcare and patient-level patterns.
5. Build a Machine Learning model for patient outcome prediction.
6. Compare and evaluate model performance.
7. Identify important features influencing predictions.
8. Develop an interactive healthcare analytics dashboard.
9. Provide a simple interface for making individual patient outcome predictions.

---

## 🧠 Machine Learning

The project treats patient outcome prediction as a **multi-class classification problem**.

### Target Variable

The target variable is:

```text
OutcomeName
```

The model predicts one of three possible outcomes:

* **Recovered**
* **Complicated**
* **Deceased**

### Features Used

The Machine Learning model uses the following patient and clinical features:

| Feature        | Description                   |
| -------------- | ----------------------------- |
| Age            | Patient age                   |
| Gender         | Patient gender                |
| DiagnosisID    | Diagnosis category identifier |
| Blood Pressure | Patient blood pressure        |
| Blood Sugar    | Blood sugar level             |
| Cholesterol    | Cholesterol level             |
| Creatinine     | Creatinine level              |
| Hemoglobin     | Hemoglobin level              |
| Vitamin D      | Vitamin D level               |

### Features Removed

Certain fields were excluded from model training because they are identifiers, dates, direct outcome-related information, or variables that could introduce unnecessary information into the prediction process.

These include:

```text
PatientID
Name
AdmissionDate
DischargeDate
TreatmentCost
LengthOfStay
OutcomeID
```

---

## 🤖 Best Performing Model

The selected model is:

### Random Forest Classifier

The Random Forest model was selected as the best-performing model for the project.

Configuration:

```python
RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
```

The complete preprocessing and model are stored together as a Scikit-learn pipeline.

The trained pipeline is saved as:

```text
models/best_model.pkl
```

---

## 📊 Model Performance

The saved evaluation report shows an overall accuracy of approximately:

### 🎯 83.3%

| Class            | Precision |   Recall | F1-Score |  Support |
| ---------------- | --------: | -------: | -------: | -------: |
| Recovered        |      0.92 |     0.93 |     0.92 |      650 |
| Complicated      |      0.66 |     0.68 |     0.67 |      250 |
| Deceased         |      0.70 |     0.58 |     0.63 |      100 |
| **Accuracy**     |           |          | **0.83** | **1000** |
| **Macro Avg**    |  **0.76** | **0.73** | **0.74** | **1000** |
| **Weighted Avg** |  **0.83** | **0.83** | **0.83** | **1000** |

### What the results indicate

The model performs particularly well for the **Recovered** class, achieving:

* **92% precision**
* **93% recall**
* **92% F1-score**

Performance is lower for the **Complicated** and **Deceased** classes.

This is important because the dataset is imbalanced, with substantially more recovered patients than deceased patients. Therefore, accuracy alone should not be used to judge the model.

The recall of **58% for the Deceased class** indicates that the model misses a meaningful number of deceased cases. For a real healthcare application, this would require further investigation and improvement before the model could be considered suitable for clinical decision-making.

> **Note:** The model is intended for educational and analytical purposes and should not be used as a medical diagnostic or treatment system.

---

## 📈 Dataset Distribution

The evaluation dataset contains:

```text
1,000 test samples
```

The test-set class distribution is:

| Outcome     | Samples |
| ----------- | ------: |
| Recovered   |     650 |
| Complicated |     250 |
| Deceased    |     100 |

This imbalance is one reason why precision, recall, and F1-score are included alongside accuracy.

---

## 🔍 Feature Importance

Random Forest provides feature importance values that help identify which input variables contribute most to the model's predictions.

The project generates:

```text
models/feature_importance.csv
models/feature_importance.png
```

The feature importance visualization provides an easier way to understand which clinical variables have the greatest influence on the trained model.

---

## 📊 Exploratory Data Analysis

The project includes an EDA notebook containing analysis of the healthcare dataset.

The analysis focuses on:

* Patient demographics
* Diagnosis distribution
* Patient outcomes
* Laboratory measurements
* Treatment costs
* Length of stay
* Relationships between clinical variables
* Outcome-wise comparisons
* Distribution of numerical variables
* Correlation analysis

EDA helps identify patterns in the dataset before applying Machine Learning.

---

## 🖥️ Interactive Dashboard

The project includes a Streamlit-based web application.

### Dashboard Features

The application provides several modules:

### 📊 Healthcare Dashboard

Provides a high-level overview of the healthcare dataset, including:

* Total patients
* Diagnosis categories
* Average treatment cost
* Patient outcome distribution
* Recovery rate
* Complicated case rate
* Deceased case rate
* Healthcare trends

### 🤖 Patient Prediction

The prediction module allows users to enter patient information and generate a predicted outcome using the trained Random Forest model.

Input fields include:

* Age
* Gender
* Diagnosis
* Blood Pressure
* Blood Sugar
* Cholesterol
* Creatinine
* Hemoglobin
* Vitamin D

The trained pipeline processes the input and returns the predicted patient outcome.

### 📈 Model Performance

The application provides model evaluation information such as:

* Accuracy
* Classification report
* Confusion matrix
* Feature importance

### ℹ️ About

The About section provides information about the project, its purpose, technology stack, and Machine Learning approach.

---

## 🗂️ Project Structure

```text
Hospital-Patient-Analytics/
│
├── data/
│   ├── raw/
│   │   ├── diagnoses.csv
│   │   ├── labs.csv
│   │   ├── outcomes.csv
│   │   └── patients.csv
│   │
│   └── processed/
│       └── healthcare_cleaned.csv
│
├── models/
│   ├── best_model.pkl
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── feature_importance.csv
│   └── feature_importance.png
│
├── notebooks/
│   ├── EDA.ipynb
│   └── ML_process.ipynb
│
├── src/
│   └── train_model3.py
│
├── streamlit_app/
│   ├── app.py
│   │
│   ├── assets/
│   │   └── style.css
│   │
│   ├── pages/
│   │   ├── About.py
│   │   ├── Dashboard.py
│   │   ├── Model_Performance.py
│   │   └── Predict.py
│   │
│   └── utils/
│       └── theme.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔄 Machine Learning Workflow

The overall workflow of the project is:

```text
Raw Healthcare Data
        ↓
Data Cleaning
        ↓
Data Integration
        ↓
Exploratory Data Analysis
        ↓
Feature Selection
        ↓
Train / Test Split
        ↓
Data Preprocessing
        ↓
Random Forest Classifier
        ↓
Model Evaluation
        ↓
Feature Importance
        ↓
Save Trained Model
        ↓
Streamlit Prediction Dashboard
```

---

## ⚙️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Random Forest Classifier

### Model Management

* Joblib

### Dashboard

* Streamlit

### Development Tools

* Jupyter Notebook
* Git
* GitHub

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Hospital-Patient-Analytics.git
```

Move into the project directory:

```bash
cd Hospital-Patient-Analytics
```

---

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

If you want to run only the Streamlit application, you can also install the application requirements:

```bash
pip install -r streamlit_app/requirements.txt
```

---

## ▶️ Running the Project

### Run the Streamlit Dashboard

From the project root directory:

```bash
streamlit run streamlit_app/app.py
```

After starting the application, Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser.

---

## 🧪 Running the Machine Learning Pipeline

The model training script is located at:

```text
src/train_model3.py
```

Run:

```bash
python src/train_model3.py
```

The script:

1. Loads the processed healthcare dataset.
2. Selects the required features.
3. Separates numerical and categorical variables.
4. Performs preprocessing.
5. Splits the data into training and testing sets.
6. Trains the Random Forest classifier.
7. Generates predictions.
8. Calculates accuracy.
9. Generates the classification report.
10. Creates the confusion matrix.
11. Calculates feature importance.
12. Saves the trained model.

The generated model and evaluation files are stored inside:

```text
models/
```

---

## 💾 Saved Model

The complete preprocessing and Random Forest pipeline is stored using Joblib:

```text
models/best_model.pkl
```

Saving the complete pipeline means that the same preprocessing steps used during training can be applied when making predictions from the Streamlit application.

---

## 📁 Important Output Files

| File                        | Purpose                                  |
| --------------------------- | ---------------------------------------- |
| `best_model.pkl`            | Trained Random Forest pipeline           |
| `classification_report.txt` | Precision, recall, F1-score and accuracy |
| `confusion_matrix.png`      | Visual model error analysis              |
| `feature_importance.csv`    | Feature importance values                |
| `feature_importance.png`    | Feature importance visualization         |

---

## 🔬 Model Evaluation

Several evaluation metrics are used instead of relying only on accuracy.

### Accuracy

Measures the overall percentage of correctly classified samples.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many predicted instances of a class were actually members of that class.

```text
Precision = True Positives / (True Positives + False Positives)
```

### Recall

Measures how many actual instances of a class were correctly identified.

```text
Recall = True Positives / (True Positives + False Negatives)
```

### F1-Score

The harmonic mean of precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

For healthcare-related classification, recall is particularly important because missing certain outcome cases can have serious consequences.

---

## 📚 Notebooks

The project contains two Jupyter notebooks.

### `EDA.ipynb`

Used for:

* Data exploration
* Data understanding
* Statistical analysis
* Visualization
* Identifying patterns and relationships

### `ML_process.ipynb`

Used for:

* Feature preparation
* Model development
* Training
* Evaluation
* Machine Learning experimentation

---

## 🎓 Project Type

```text
Academic / Portfolio Project
```

This project demonstrates practical skills in:

* Python
* Pandas
* NumPy
* Data Cleaning
* Exploratory Data Analysis
* Data Visualization
* Machine Learning
* Classification
* Model Evaluation
* Feature Importance
* Streamlit
* Git & GitHub

---

## 👨‍💻 Author

**Ayush** & **Moon**

This project was developed as part of a Data Analytics and Machine Learning portfolio.

---
## ⚕️ Disclaimer

This project is created for **educational and demonstration purposes only**.
