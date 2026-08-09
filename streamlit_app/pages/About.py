import streamlit as st
import sys
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="About",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# THEME
# ============================================================

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)


from utils.theme import apply_theme, render_sidebar

apply_theme()

render_sidebar(
    total_patients=5000,
    model_accuracy="83.3%",
    total_diagnoses=10
)

# ============================================================
# HERO HEADER
# ============================================================

left, right = st.columns(
    [3, 1]
)

with left:

    st.title(
        "🏥 Hospital Patient Analytics"
    )

    st.markdown(
        """
        ## Healthcare Data Analytics & Machine Learning Platform

        Transforming hospital data into meaningful insights through
        interactive dashboards, exploratory data analysis and
        predictive machine learning.

        This project demonstrates an end-to-end Healthcare Analytics
        solution using Python, Pandas, Scikit-learn, Plotly and Streamlit.
        """
    )


with right:

    st.info(
        """
        ### 📌 Project

        🏥 Healthcare Analytics

        📊 Data Analytics

        🤖 Machine Learning

        🖥 Interactive Dashboard

        ✅ Version 1.0
        """
    )


st.divider()


# ============================================================
# PROJECT STATISTICS
# ============================================================

st.markdown(
    "## 📊 Project Statistics"
)

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "👨‍⚕️ Patients",
        "5,000"
    )


with c2:

    st.metric(
        "🩺 Diagnoses",
        "10"
    )


with c3:

    st.metric(
        "🎯 Outcomes",
        "3"
    )


with c4:

    st.metric(
        "🤖 Model Accuracy",
        "83.3%"
    )


st.divider()


# ============================================================
# PROJECT HIGHLIGHTS
# ============================================================

st.markdown(
    "## ⭐ Project Highlights"
)

h1, h2, h3 = st.columns(3)


with h1:

    st.success(
        """
        ### 📊 Analytics

        ✔ Interactive Dashboard

        ✔ Patient Analytics

        ✔ KPI Monitoring

        ✔ Hospital Statistics
        """
    )


with h2:

    st.info(
        """
        ### 🤖 Machine Learning

        ✔ Patient Outcome Prediction

        ✔ Random Forest Classification

        ✔ Model Evaluation

        ✔ Feature Importance Analysis
        """
    )


with h3:

    st.warning(
        """
        ### 🚀 Deployment

        ✔ Streamlit

        ✔ Interactive Charts

        ✔ Responsive UI

        ✔ User-Friendly Interface
        """
    )


st.divider()


# ============================================================
# TECHNOLOGY STACK
# ============================================================

st.markdown(
    "## 🛠 Technology Stack"
)

tech1, tech2, tech3 = st.columns(3)


# ------------------------------------------------------------
# PROGRAMMING & DATA
# ------------------------------------------------------------

with tech1:

    with st.container(border=True):

        st.markdown("## 🐍")

        st.markdown(
            "### Programming & Data"
        )

        st.write("• Python")

        st.write("• Pandas")

        st.write("• NumPy")

        st.write("• Joblib")

        st.write("• Jupyter Notebook")


# ------------------------------------------------------------
# MACHINE LEARNING
# ------------------------------------------------------------

with tech2:

    with st.container(border=True):

        st.markdown("## 🤖")

        st.markdown(
            "### Machine Learning"
        )

        st.write("• Scikit-learn")

        st.write("• Random Forest")

        st.write("• Classification")

        st.write("• Feature Importance")

        st.write("• Model Evaluation")


# ------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------

with tech3:

    with st.container(border=True):

        st.markdown("## 📊")

        st.markdown(
            "### Visualization & UI"
        )

        st.write("• Streamlit")

        st.write("• Plotly")

        st.write("• Matplotlib")

        st.write("• Seaborn")

        st.write("• CSS")


st.divider()


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

st.markdown(
    "## 🤖 Machine Learning Model"
)

model_col1, model_col2 = st.columns(
    [2, 1]
)


with model_col1:

    st.markdown(
        """
        ### Random Forest Classifier

        The platform uses a Random Forest classification model to
        predict the patient's hospital outcome.

        The model is trained using clinical and demographic features:

        • Age

        • Gender

        • Diagnosis

        • Blood Pressure

        • Blood Sugar

        • Cholesterol

        • Creatinine

        • Hemoglobin

        • Vitamin D
        """
    )


with model_col2:

    st.metric(
        "Test Accuracy",
        "83.3%"
    )

    st.metric(
        "Macro F1 Score",
        "0.74"
    )

    st.metric(
        "Test Samples",
        "1,000"
    )


st.divider()


# ============================================================
# PROJECT WORKFLOW
# ============================================================

st.markdown(
    "## 🔄 Project Workflow"
)

step1, step2, step3, step4, step5 = st.columns(5)


with step1:

    with st.container(border=True):

        st.markdown("### 📂")

        st.markdown("Dataset")

        st.caption(
            "Raw Patient Records"
        )


with step2:

    with st.container(border=True):

        st.markdown("### 🧹")

        st.markdown("Cleaning")

        st.caption(
            "Data Preprocessing"
        )


with step3:

    with st.container(border=True):

        st.markdown("### 📈")

        st.markdown("EDA")

        st.caption(
            "Exploratory Analysis"
        )


with step4:

    with st.container(border=True):

        st.markdown("### 🤖")

        st.markdown("ML")

        st.caption(
            "Model Training"
        )


with step5:

    with st.container(border=True):

        st.markdown("### 🚀")

        st.markdown("Deployment")

        st.caption(
            "Streamlit Application"
        )


st.divider()


# ============================================================
# PROJECT TIMELINE
# ============================================================

st.markdown(
    "## 📌 Development Journey"
)

st.write(
    "📥 Data Collection"
)

st.progress(100)


st.write(
    "🧹 Data Cleaning & Preprocessing"
)

st.progress(100)


st.write(
    "📊 Exploratory Data Analysis"
)

st.progress(100)


st.write(
    "🤖 Machine Learning"
)

st.progress(100)


st.write(
    "🚀 Streamlit Deployment"
)

st.progress(100)


st.divider()


# ============================================================
# KEY FEATURES
# ============================================================

st.markdown(
    "## ⭐ Key Features"
)

feature1, feature2, feature3 = st.columns(3)


with feature1:

    with st.container(border=True):

        st.markdown(
            "### 📊 Data Analytics"
        )

        st.write(
            "✔ Interactive Dashboard"
        )

        st.write(
            "✔ Patient Analytics"
        )

        st.write(
            "✔ Hospital KPIs"
        )

        st.write(
            "✔ Business Insights"
        )


with feature2:

    with st.container(border=True):

        st.markdown(
            "### 🤖 Machine Learning"
        )

        st.write(
            "✔ Outcome Prediction"
        )

        st.write(
            "✔ Random Forest Model"
        )

        st.write(
            "✔ Model Evaluation"
        )

        st.write(
            "✔ Feature Importance"
        )


with feature3:

    with st.container(border=True):

        st.markdown(
            "### 📈 Visualization"
        )

        st.write(
            "✔ Interactive Charts"
        )

        st.write(
            "✔ Plotly Visualizations"
        )

        st.write(
            "✔ Responsive Dashboard"
        )

        st.write(
            "✔ Light / Dark Theme"
        )


st.divider()


# ============================================================
# FUTURE SCOPE
# ============================================================

st.markdown(
    "## 🚀 Future Scope"
)

future1, future2 = st.columns(2)


with future1:

    st.success(
        """
        ### Future Enhancements

        • Larger Healthcare Dataset

        • Real Hospital Records

        • Explainable AI (XAI)

        • Cloud Deployment

        • Advanced Model Optimization
        """
    )


with future2:

    st.info(
        """
        ### Possible Extensions

        • Doctor Dashboard

        • Patient Portal

        • REST API Integration

        • Mobile Application

        • Real-Time Prediction
        """
    )


st.divider()


# ============================================================
# DEVELOPERS
# ============================================================

st.markdown(
    "## 👨‍💻 Developers"
)

dev1, dev2 = st.columns(2)


with dev1:

    with st.container(border=True):

        st.markdown(
            "### 👨‍💻 Ayush"
        )

        st.write(
            "• Data Analytics"
        )

        st.write(
            "• Machine Learning"
        )

        st.write(
            "• Streamlit Dashboard"
        )

        st.write(
            "• Model Development"
        )


with dev2:

    with st.container(border=True):

        st.markdown(
            "### 👩‍💻 Moon"
        )

        st.write(
            "• Data Preparation"
        )

        st.write(
            "• Testing"
        )

        st.write(
            "• Documentation"
        )

        st.write(
            "• Project Support"
        )

        st.write(
            "• Model Development"
        )


st.divider()


# ============================================================
# DISCLAIMER
# ============================================================

st.caption(
    """
    ⚠️ This project is developed for educational and analytical
    purposes using synthetic healthcare data. Model predictions
    should not be used as a substitute for professional medical
    diagnosis or clinical decision-making.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="text-align:center; padding:20px 0 5px 0;">
        <p style="margin:0;">
            Developed by <strong>Ayush & Moon</strong>
        </p>
        <p style="margin:5px 0 0 0; opacity:0.7;">
            Hospital Patient Analytics • Version 1.0
        </p>
    </div>
    """,
    unsafe_allow_html=True
)