# 🏥 Hospital Patient Analytics & Outcome Prediction

A healthcare analytics and machine learning project that analyzes hospital patient records, explores treatment and outcome trends, and predicts patient outcomes using Machine Learning.

The project combines **Python, Pandas, Scikit-Learn, Plotly, and Streamlit** to create an interactive healthcare analytics application.

---

## 📌 About the Project

Healthcare organizations generate a large amount of patient data, but raw data by itself is difficult to interpret.

This project was built to turn hospital records into useful information through:

- Patient data analysis
- Disease and demographic analysis
- Treatment cost analysis
- Patient outcome analysis
- Interactive visualizations
- Machine Learning-based outcome prediction
- Model performance evaluation

The final result is a Streamlit web application where users can explore the data and enter patient information to receive a predicted outcome.

---

## 🎯 Project Objectives

The main objectives of this project are to:

- Understand and clean hospital patient data
- Perform exploratory data analysis
- Identify important healthcare trends
- Analyze treatment costs and patient outcomes
- Engineer useful features for Machine Learning
- Compare multiple classification algorithms
- Tune the best-performing model
- Build an interactive dashboard
- Deploy the trained model inside a user-friendly application

---

## 📊 Dataset

The project uses a structured hospital patient dataset containing demographic, diagnosis, treatment, laboratory, and outcome information.

Some of the important fields include:

| Feature | Description |
|---|---|
| `PatientID` | Unique patient identifier |
| `Age` | Patient age |
| `Gender` | Patient gender |
| `DiagnosisName` | Patient diagnosis |
| `TreatmentCost` | Total treatment cost |
| `LengthOfStay` | Number of days spent in hospital |
| `AvgLabResult` | Average laboratory result |
| `NumberOfTests` | Number of laboratory tests |
| `OutcomeName` | Final patient outcome |

The dataset used for this project is synthetic and is included for demonstration and development purposes.

---

## 🧹 Data Preparation

Before training the models, the data was cleaned and transformed.

The preprocessing included:

- Handling missing laboratory results
- Handling missing test counts
- Creating age groups
- Calculating treatment cost per day
- Creating a laboratory-record indicator
- Creating laboratory result per test
- Encoding categorical variables
- Scaling numerical variables
- Handling missing values through a preprocessing pipeline

### Feature Engineering

Additional features were created to improve the information available to the models:

```text
AgeGroup
CostPerDay
HasLabRecord
LabPerTest
