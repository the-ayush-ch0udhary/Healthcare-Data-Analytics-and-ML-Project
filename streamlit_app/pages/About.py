"""
About Page & Clinical Reference Guide
Hospital Patient Analytics Platform
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIG & THEME
# ============================================================

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(APP_DIR))

from utils.theme import (
    apply_theme,
    get_model_metadata,
    load_dataset,
    render_sidebar,
)

st.set_page_config(
    page_title="About & Clinical Reference",
    page_icon="ℹ️",
    layout="wide",
)

apply_theme()

BASE_DIR = APP_DIR.parent
df = load_dataset()
meta = get_model_metadata()

render_sidebar(
    total_patients=len(df) if not df.empty else meta.get("dataset_total", 5000),
    model_accuracy=meta.get("accuracy_pct", "83.30%"),
    total_diagnoses=df["DiagnosisName"].nunique() if not df.empty else 10,
)

# ============================================================
# HERO HEADER
# ============================================================

left, right = st.columns([3, 1])

with left:
    st.title("🏥 About Hospital Patient Analytics")
    st.markdown(
        """
        ### AI-Powered Healthcare Intelligence & Clinical Decision Support System
        
        The **Hospital Patient Analytics** platform is an end-to-end data analytics and machine learning solution designed to unlock actionable insights from inpatient hospital records, evaluate clinical biomarker trends, and predict patient discharge outcomes (*Recovered, Complicated, Deceased*).
        """
    )

with right:
    st.info(
        f"""
        ### 📌 Platform Stats
        👥 **Patients:** {len(df) if not df.empty else 5000:,}
        
        🩺 **Diagnoses:** {df['DiagnosisName'].nunique() if not df.empty else 10}
        
        🎯 **Accuracy:** {meta.get('accuracy_pct', '83.30%')}
        
        🚀 **Version:** 2.0
        """
    )

st.divider()

# ============================================================
# CLINICAL REFERENCE RANGES TABLE
# ============================================================

st.markdown("## 🧪 Clinical Biomarker Reference Guide")
st.write(
    "Standard clinical laboratory thresholds utilized across the platform for patient assessment and risk stratification:"
)

ref_data = [
    {
        "Biomarker": "Blood Pressure (Systolic)",
        "Unit": "mmHg",
        "Normal Range": "90 – 120",
        "Elevated / Warning": "121 – 139",
        "High / Critical": "≥ 140 (Hypertension)",
        "Clinical Relevance": "Cardiovascular risk indicator and hemodynamic stability marker.",
    },
    {
        "Biomarker": "Blood Sugar (Fasting)",
        "Unit": "mg/dL",
        "Normal Range": "70 – 99",
        "Elevated / Warning": "100 – 125 (Prediabetes)",
        "High / Critical": "≥ 126 (Diabetes) / < 70 (Hypoglycemia)",
        "Clinical Relevance": "Glycemic regulation, diabetic monitoring, and metabolic stress indicator.",
    },
    {
        "Biomarker": "Total Cholesterol",
        "Unit": "mg/dL",
        "Normal Range": "< 200 (Desirable)",
        "Elevated / Warning": "200 – 239 (Borderline)",
        "High / Critical": "≥ 240 (High)",
        "Clinical Relevance": "Lipid profiling, atherosclerosis, and long-term coronary disease risk.",
    },
    {
        "Biomarker": "Serum Creatinine",
        "Unit": "mg/dL",
        "Normal Range": "0.6 – 1.2",
        "Elevated / Warning": "1.3 – 1.8",
        "High / Critical": "≥ 1.9 (Renal Stress / Failure)",
        "Clinical Relevance": "Renal filtration efficiency, kidney disease staging, and toxicity marker.",
    },
    {
        "Biomarker": "Hemoglobin",
        "Unit": "g/dL",
        "Normal Range": "12.0 – 17.5",
        "Elevated / Warning": "10.0 – 11.9 (Mild Anemia)",
        "High / Critical": "< 10.0 (Severe Anemia) / > 17.5",
        "Clinical Relevance": "Oxygen carrying capacity, acute blood loss, and systemic perfusion.",
    },
    {
        "Biomarker": "Vitamin D",
        "Unit": "ng/mL",
        "Normal Range": "30 – 100 (Optimal)",
        "Elevated / Warning": "20 – 29 (Insufficient)",
        "High / Critical": "< 20 (Deficient)",
        "Clinical Relevance": "Immune function, bone mineral density, and general metabolic resilience.",
    },
]

st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

st.divider()

# ============================================================
# SYSTEM ARCHITECTURE & WORKFLOW
# ============================================================

st.markdown("## 🔄 End-to-End System Workflow")

w1, w2, w3, w4, w5 = st.columns(5)

with w1:
    with st.container(border=True):
        st.markdown("### 1. Ingestion")
        st.write("Patients, Labs, Diagnoses & Outcomes raw CSV datasets.")

with w2:
    with st.container(border=True):
        st.markdown("### 2. ETL")
        st.write("Merging, pivot aggregation, and schema standardization.")

with w3:
    with st.container(border=True):
        st.markdown("### 3. Analytics")
        st.write("Interactive dashboard, correlations, and demographic insights.")

with w4:
    with st.container(border=True):
        st.markdown("### 4. ML Pipeline")
        st.write("Multi-model benchmark, Random Forest champion model training.")

with w5:
    with st.container(border=True):
        st.markdown("### 5. Prediction")
        st.write("Single patient clinical triage and batch CSV inference engine.")

st.divider()

# ============================================================
# TECHNOLOGY STACK
# ============================================================

st.markdown("## 🛠 Technology Stack")

t1, t2, t3 = st.columns(3)

with t1:
    with st.container(border=True):
        st.markdown("### 🐍 Core & Data Processing")
        st.write("• **Python 3.10+** - Core programming language")
        st.write("• **Pandas** - High-performance tabular manipulation")
        st.write("• **NumPy** - Numerical computing & vectorization")
        st.write("• **Joblib** - Pipeline serialization & persistence")

with t2:
    with st.container(border=True):
        st.markdown("### 🤖 Machine Learning")
        st.write("• **Scikit-Learn** - Modeling, Pipelines & Evaluation")
        st.write("• **Random Forest** - Champion nonlinear classifier")
        st.write("• **Gradient Boosting & Logistic Regression** - Benchmarks")
        st.write("• **ColumnTransformer & StandardScaler** - Preprocessing")

with t3:
    with st.container(border=True):
        st.markdown("### 🖥 Visualization & UI")
        st.write("• **Streamlit** - Interactive multi-page web platform")
        st.write("• **Plotly Express** - Dynamic interactive charts")
        st.write("• **Seaborn & Matplotlib** - Model evaluation charts")
        st.write("• **Custom CSS** - Dark/Light responsive design system")

st.divider()

# ============================================================
# DEVELOPERS & CREDITS
# ============================================================

st.markdown("## 👨‍💻 Project Development Team")

dev1, dev2 = st.columns(2)

with dev1:
    with st.container(border=True):
        st.markdown("### 👨‍💻 Ayush")
        st.write("• Architecture Design & End-to-End Development")
        st.write("• Machine Learning Pipeline & Model Benchmarking")
        st.write("• Interactive Streamlit UI & Design System")
        st.write("• Predictive Decision Support Integration")

with dev2:
    with st.container(border=True):
        st.markdown("### 👩‍💻 Moon")
        st.write("• Data Preprocessing Pipeline & Verification")
        st.write("• Exploratory Data Analysis & Clinical Metrics")
        st.write("• Documentation & Quality Assurance Testing")
        st.write("• Analytics Dashboard Design")

st.divider()

# ============================================================
# DISCLAIMER & FOOTER
# ============================================================

st.caption(
    """
    ⚠️ **Disclaimer**: This application is developed for educational, analytical, and clinical decision support demonstration purposes using synthetic patient records.
    It does not replace professional medical diagnosis or clinical advice.
    """
)

st.html(
    """
    <div style="text-align:center; padding:15px 0 5px 0; color:var(--text-muted); font-size:12px;">
        Hospital Patient Analytics • Version 2.0 • Ayush &amp; Moon
    </div>
    """
)