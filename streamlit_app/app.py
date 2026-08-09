import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st


# ============================================================
# PATH SETUP
# ============================================================

APP_DIR = Path(__file__).resolve().parent

sys.path.append(
    str(APP_DIR)
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hospital Patient Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# THEME
# ============================================================

from utils.theme import (
    apply_theme,
    render_sidebar
)

apply_theme()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = APP_DIR.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "healthcare_cleaned.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_PATH
    )


df = load_data()


# ============================================================
# DATA INFORMATION
# ============================================================

TOTAL_PATIENTS = len(df)

TOTAL_DIAGNOSES = (
    df["DiagnosisName"]
    .nunique()
)

AVG_COST = int(
    df["TreatmentCost"].mean()
)

MODEL_ACCURACY = "83.3%"

TODAY = datetime.now().strftime(
    "%d %b %Y"
)


# ============================================================
# SIDEBAR
# ============================================================

render_sidebar(
    total_patients=TOTAL_PATIENTS,
    model_accuracy=MODEL_ACCURACY,
    total_diagnoses=TOTAL_DIAGNOSES
)


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div style="
        padding:38px 34px 32px 34px;
        border:1px solid var(--card-border);
        border-radius:20px;
        background:var(--card-bg);
        box-shadow:0 8px 32px var(--card-shadow);
        margin-bottom:25px;
    ">

        <div style="
            font-size:48px;
            font-weight:800;
            line-height:1.15;
            margin-bottom:18px;
            background:linear-gradient(
                90deg,
                var(--heading-start),
                var(--heading-end)
            );
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
        ">
            🏥 Hospital Patient Analytics
        </div>


        <div style="
            font-size:27px;
            font-weight:700;
            color:var(--text-primary);
            margin-bottom:14px;
        ">
            AI-Powered Healthcare Intelligence Platform
        </div>


        <div style="
            font-size:16px;
            line-height:1.7;
            color:var(--text-secondary);
            max-width:950px;
            margin-bottom:28px;
        ">
            Analyze patient records, discover healthcare insights,
            understand clinical patterns and predict patient outcomes
            using Machine Learning.
        </div>


        <div style="
            display:grid;
            grid-template-columns:repeat(4,1fr);
            gap:14px;
        ">

            <div style="
                padding:18px;
                border:1px solid var(--card-border);
                border-radius:14px;
                background:var(--input-bg);
            ">

                <div style="font-size:25px;">
                    📊
                </div>

                <div style="
                    font-size:16px;
                    font-weight:700;
                    margin-top:8px;
                    color:var(--text-primary);
                ">
                    Healthcare Analytics
                </div>

                <div style="
                    font-size:13px;
                    margin-top:5px;
                    color:var(--text-muted);
                ">
                    Explore patient and hospital trends
                </div>

            </div>


            <div style="
                padding:18px;
                border:1px solid var(--card-border);
                border-radius:14px;
                background:var(--input-bg);
            ">

                <div style="font-size:25px;">
                    🩺
                </div>

                <div style="
                    font-size:16px;
                    font-weight:700;
                    margin-top:8px;
                    color:var(--text-primary);
                ">
                    Clinical Insights
                </div>

                <div style="
                    font-size:13px;
                    margin-top:5px;
                    color:var(--text-muted);
                ">
                    Analyze patient health indicators
                </div>

            </div>


            <div style="
                padding:18px;
                border:1px solid var(--card-border);
                border-radius:14px;
                background:var(--input-bg);
            ">

                <div style="font-size:25px;">
                    🤖
                </div>

                <div style="
                    font-size:16px;
                    font-weight:700;
                    margin-top:8px;
                    color:var(--text-primary);
                ">
                    Outcome Prediction
                </div>

                <div style="
                    font-size:13px;
                    margin-top:5px;
                    color:var(--text-muted);
                ">
                    Predict patient outcomes with ML
                </div>

            </div>


            <div style="
                padding:18px;
                border:1px solid var(--card-border);
                border-radius:14px;
                background:var(--input-bg);
            ">

                <div style="font-size:25px;">
                    📈
                </div>

                <div style="
                    font-size:16px;
                    font-weight:700;
                    margin-top:8px;
                    color:var(--text-primary);
                ">
                    Model Performance
                </div>

                <div style="
                    font-size:13px;
                    margin-top:5px;
                    color:var(--text-muted);
                ">
                    Evaluate accuracy and metrics
                </div>

            </div>

        </div>


        <div style="
            margin-top:20px;
            padding:14px 18px;
            border-radius:12px;
            background:var(--input-bg);
            border:1px solid var(--card-border);
            display:grid;
            grid-template-columns:repeat(4,1fr);
            gap:15px;
        ">

            <div style="
                font-size:14px;
                color:var(--text-secondary);
            ">
                🟢 <strong>Platform</strong><br>
                Operational
            </div>

            <div style="
                font-size:14px;
                color:var(--text-secondary);
            ">
                🤖 <strong>Model</strong><br>
                Random Forest
            </div>

            <div style="
                font-size:14px;
                color:var(--text-secondary);
            ">
                🎯 <strong>Accuracy</strong><br>
                83.3%
            </div>

            <div style="
                font-size:14px;
                color:var(--text-secondary);
            ">
                👥 <strong>Records</strong><br>
                5,000
            </div>

        </div>

    </div>
    """
)


# ============================================================
# PLATFORM
# ============================================================

st.markdown(
    "## 🚀 Explore the Platform"
)

st.write(
    "Use the navigation panel to explore healthcare "
    "analytics, predictions and model performance."
)

st.divider()


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

st.markdown(
    "## 📊 Executive Dashboard"
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)


with kpi1:

    st.metric(
        label="👨‍⚕️ Total Patients",
        value=f"{TOTAL_PATIENTS:,}",
        delta="Records"
    )


with kpi2:

    st.metric(
        label="🎯 Model Accuracy",
        value=MODEL_ACCURACY,
        delta="Random Forest"
    )


with kpi3:

    st.metric(
        label="🩺 Diagnosis Categories",
        value=TOTAL_DIAGNOSES,
        delta="Categories"
    )


with kpi4:

    st.metric(
        label="💰 Avg Treatment Cost",
        value=f"₹ {AVG_COST:,}",
        delta="Per Patient"
    )


st.divider()


# ============================================================
# OUTCOME ANALYSIS
# ============================================================

st.markdown(
    "## 📌 Patient Outcome Overview"
)

overview1, overview2, overview3 = st.columns(3)


# ------------------------------------------------------------
# RECOVERED
# ------------------------------------------------------------

with overview1:

    recovered = (
        df["OutcomeName"]
        == "Recovered"
    ).sum()

    recovery_rate = round(
        recovered
        / TOTAL_PATIENTS
        * 100,
        1
    )

    st.success(
        f"""
        ### ✅ Recovery

        **Recovered Patients**

        {recovered:,}

        **Recovery Rate**

        {recovery_rate}%
        """
    )


# ------------------------------------------------------------
# COMPLICATED
# ------------------------------------------------------------

with overview2:

    complicated = (
        df["OutcomeName"]
        == "Complicated"
    ).sum()

    complicated_rate = round(
        complicated
        / TOTAL_PATIENTS
        * 100,
        1
    )

    st.info(
        f"""
        ### ⚠ Complicated

        **Complicated Cases**

        {complicated:,}

        **Case Rate**

        {complicated_rate}%
        """
    )


# ------------------------------------------------------------
# DECEASED
# ------------------------------------------------------------

with overview3:

    deceased = (
        df["OutcomeName"]
        == "Deceased"
    ).sum()

    deceased_rate = round(
        deceased
        / TOTAL_PATIENTS
        * 100,
        1
    )

    st.warning(
        f"""
        ### 🔴 Deceased

        **Deceased Patients**

        {deceased:,}

        **Case Rate**

        {deceased_rate}%
        """
    )


st.divider()


# ============================================================
# APPLICATION MODULES
# ============================================================

st.markdown(
    "## 🚀 Application Modules"
)

module1, module2, module3 = st.columns(3)


# ------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------

with module1:

    with st.container(border=True):

        st.markdown(
            "### 📊 Dashboard"
        )

        st.write(
            """
            Explore patient statistics,
            diagnosis distribution, clinical
            measurements and hospital analytics.
            """
        )

        st.page_link(
            "pages/Dashboard.py",
            label="Open Dashboard",
            icon="📊"
        )


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

with module2:

    with st.container(border=True):

        st.markdown(
            "### 🤖 Prediction"
        )

        st.write(
            """
            Predict patient outcomes using
            the trained Machine Learning model.
            """
        )

        st.page_link(
            "pages/Predict.py",
            label="Open Prediction",
            icon="🤖"
        )


# ------------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------------

with module3:

    with st.container(border=True):

        st.markdown(
            "### 📈 Model Performance"
        )

        st.write(
            """
            Review accuracy, classification metrics,
            confusion matrix and feature importance.
            """
        )

        st.page_link(
            "pages/Model_Performance.py",
            label="View Performance",
            icon="📈"
        )


st.divider()


# ============================================================
# DATASET SUMMARY
# ============================================================

st.markdown(
    "## 📈 Dataset Summary"
)

summary1, summary2, summary3 = st.columns(3)


with summary1:

    st.metric(
        "Recovered",
        f"{recovered:,}"
    )

    st.metric(
        "Complicated",
        f"{complicated:,}"
    )


with summary2:

    st.metric(
        "Deceased",
        f"{deceased:,}"
    )

    st.metric(
        "Average Age",
        f"{df['Age'].mean():.1f}"
    )


with summary3:

    st.metric(
        "Maximum Treatment Cost",
        f"₹ {df['TreatmentCost'].max():,}"
    )

    st.metric(
        "Average Length of Stay",
        f"{df['LengthOfStay'].mean():.1f} Days"
    )


st.divider()


# ============================================================
# MACHINE LEARNING
# ============================================================

st.markdown(
    "## 🤖 Machine Learning Summary"
)

ml1, ml2, ml3 = st.columns(3)


with ml1:

    st.metric(
        "Algorithm",
        "Random Forest"
    )


with ml2:

    st.metric(
        "Test Accuracy",
        "83.3%"
    )


with ml3:

    st.metric(
        "Test Samples",
        "1,000"
    )


st.caption(
    """
    The model was evaluated using a stratified 80:20
    train-test split across three patient outcome classes.
    """
)


st.divider()


# ============================================================
# PROJECT HIGHLIGHTS
# ============================================================

st.markdown(
    "## ⭐ Project Highlights"
)

highlight1, highlight2 = st.columns(2)


with highlight1:

    st.success(
        """
        ### 📊 Data Analytics

        ✔ Interactive Healthcare Dashboard

        ✔ Patient Statistics

        ✔ Diagnosis Analysis

        ✔ Treatment Cost Analysis

        ✔ Clinical Measurements

        ✔ Outcome Analysis
        """
    )


with highlight2:

    st.info(
        """
        ### 🤖 Machine Learning

        ✔ Outcome Prediction

        ✔ Random Forest Classification

        ✔ Model Evaluation

        ✔ Confusion Matrix

        ✔ Feature Importance

        ✔ Classification Report
        """
    )


st.divider()


# ============================================================
# FUTURE ENHANCEMENTS
# ============================================================

st.markdown(
    "## 🚀 Future Enhancements"
)

future1, future2, future3 = st.columns(3)


with future1:

    with st.container(border=True):

        st.markdown(
            "### 🏥 Healthcare"
        )

        st.write(
            "• Electronic Health Records"
        )

        st.write(
            "• Doctor Dashboard"
        )

        st.write(
            "• Patient Portal"
        )

        st.write(
            "• Real-Time Monitoring"
        )


with future2:

    with st.container(border=True):

        st.markdown(
            "### 🤖 AI"
        )

        st.write(
            "• Explainable AI"
        )

        st.write(
            "• Advanced Ensemble Models"
        )

        st.write(
            "• Risk Prediction"
        )

        st.write(
            "• Model Optimization"
        )


with future3:

    with st.container(border=True):

        st.markdown(
            "### ☁ Deployment"
        )

        st.write(
            "• REST API"
        )

        st.write(
            "• Cloud Hosting"
        )

        st.write(
            "• Authentication"
        )

        st.write(
            "• Mobile Dashboard"
        )


st.divider()


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.markdown(
    "## 📌 Project Information"
)

info1, info2 = st.columns([2, 1])


with info1:

    st.write(
        """
        Hospital Patient Analytics is an end-to-end
        Data Analytics and Machine Learning project.

        The platform combines healthcare data analysis,
        interactive visualization and machine learning
        based patient outcome prediction in a Streamlit
        application.
        """
    )


with info2:

    st.markdown(
        "### 🛠 Technology Stack"
    )

    st.write("🐍 Python")
    st.write("📊 Pandas")
    st.write("📈 Plotly")
    st.write("🤖 Scikit-Learn")
    st.write("🖥 Streamlit")


st.divider()


# ============================================================
# DISCLAIMER
# ============================================================

st.caption(
    """
    ⚠️ This project uses synthetic healthcare data and is
    intended for educational and analytical purposes only.
    Model predictions must not be used as a substitute for
    professional medical diagnosis or clinical decision-making.
    """
)


st.divider()


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div style="
        margin-top:40px;
        padding:30px 20px 25px 20px;
        text-align:center;
        border-top:1px solid var(--divider);
    ">

        <div style="
            font-size:22px;
            font-weight:800;
            color:var(--text-primary);
            margin-bottom:10px;
        ">
            🏥 Hospital Patient Analytics
        </div>


        <div style="
            font-size:14px;
            color:var(--text-secondary);
            margin-bottom:8px;
        ">
            Developed using
            <strong style="
                color:var(--text-primary);
            ">
                Python • Pandas • Plotly • Streamlit • Scikit-Learn
            </strong>
        </div>


        <div style="
            font-size:13px;
            color:var(--text-muted);
            margin-bottom:10px;
        ">
            Healthcare Data Analytics & Machine Learning Project
        </div>


        <div style="
            font-size:14px;
            font-weight:600;
            color:var(--text-secondary);
        ">
            Developed by
            <strong style="
                color:var(--text-primary);
            ">
                Ayush &amp; Moon
            </strong>
        </div>


        <div style="
            font-size:12px;
            margin-top:10px;
            color:var(--text-muted);
        ">
            Version 1.0
        </div>

    </div>
    """
)