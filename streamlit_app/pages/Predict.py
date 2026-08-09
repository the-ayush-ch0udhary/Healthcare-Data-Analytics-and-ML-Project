import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Patient Outcome Prediction",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# THEME
# ============================================================

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from utils.theme import apply_theme, render_sidebar, plotly_template

apply_theme()

render_sidebar(
    total_patients=5000,
    model_accuracy="83.3%",
    total_diagnoses=10
)

PLOTLY_TEMPLATE = plotly_template()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_model.pkl"
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "healthcare_cleaned.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_PATH
    )


model = load_model()


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
# PAGE HEADER
# ============================================================

left, right = st.columns(
    [3, 1]
)

with left:

    st.title(
        "🤖 Patient Outcome Prediction"
    )

    st.subheader(
        "Machine Learning Based Hospital Outcome Prediction"
    )

    st.write(
        """
        Enter the patient's clinical information below.
        The trained Random Forest model will predict the
        most likely hospital outcome.
        """
    )


with right:

    st.success(
        """
        ### 🟢 Model Status

        **Ready**

        **Model**

        Random Forest

        **Test Accuracy**

        83.30%
        """
    )


st.divider()


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.markdown(
    "## 👨‍⚕️ Patient Clinical Information"
)


left, right = st.columns(
    2
)


# ============================================================
# LEFT COLUMN
# ============================================================

with left:

    age = st.slider(
        "Age",
        min_value=25,
        max_value=90,
        value=50
    )

    gender = st.selectbox(
        "Gender",
        sorted(
            df["Gender"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    diagnosis_options = (
        df[
            [
                "DiagnosisID",
                "DiagnosisName"
            ]
        ]
        .drop_duplicates()
        .sort_values("DiagnosisID")
    )

    diagnosis_name = st.selectbox(
        "Diagnosis",
        diagnosis_options[
            "DiagnosisName"
        ].tolist()
    )

    diagnosis_id = int(
        diagnosis_options.loc[
            diagnosis_options["DiagnosisName"]
            == diagnosis_name,
            "DiagnosisID"
        ].iloc[0]
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with right:

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=80.0,
        max_value=220.0,
        value=120.0,
        step=0.1
    )

    blood_sugar = st.number_input(
        "Blood Sugar",
        min_value=40.0,
        max_value=350.0,
        value=100.0,
        step=0.1
    )

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=80.0,
        max_value=350.0,
        value=190.0,
        step=0.1
    )

    creatinine = st.number_input(
        "Creatinine",
        min_value=0.3,
        max_value=5.0,
        value=1.0,
        step=0.01
    )

    hemoglobin = st.number_input(
        "Hemoglobin",
        min_value=7.0,
        max_value=20.0,
        value=14.0,
        step=0.1
    )

    vitamin_d = st.number_input(
        "Vitamin D",
        min_value=5.0,
        max_value=80.0,
        value=35.0,
        step=0.1
    )


st.divider()


# ============================================================
# INPUT SUMMARY
# ============================================================

st.markdown(
    "## 📋 Input Summary"
)

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:

    st.metric(
        "Age",
        age
    )

    st.metric(
        "Gender",
        gender
    )

with summary_col2:

    st.metric(
        "Diagnosis",
        diagnosis_name
    )

    st.metric(
        "Diagnosis ID",
        diagnosis_id
    )

with summary_col3:

    st.metric(
        "Blood Pressure",
        f"{blood_pressure:.1f}"
    )

    st.metric(
        "Blood Sugar",
        f"{blood_sugar:.1f}"
    )


st.divider()


# ============================================================
# PREDICT BUTTON
# ============================================================

predict = st.button(
    "🔍 Predict Patient Outcome",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

    input_df = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [gender],
            "DiagnosisID": [str(diagnosis_id)],
            "Blood Pressure": [blood_pressure],
            "Blood Sugar": [blood_sugar],
            "Cholesterol": [cholesterol],
            "Creatinine": [creatinine],
            "Hemoglobin": [hemoglobin],
            "Vitamin D": [vitamin_d]
        }
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        input_df
    )[0]

    probabilities = model.predict_proba(
        input_df
    )[0]

    classes = model.classes_

    confidence = (
        max(probabilities) * 100
    )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "## 🎯 Prediction Result"
    )


    if prediction == "Recovered":

        st.success(
            f"""
            ### ✅ Predicted Outcome

            # {prediction}

            **Prediction Confidence:
            {confidence:.2f}%**
            """
        )

        recommendation = """
        The model predicts a Recovered outcome.

        • Continue appropriate clinical monitoring.

        • Maintain regular follow-up.

        • Review patient condition according to
          standard clinical practice.
        """


    elif prediction == "Complicated":

        st.warning(
            f"""
            ### ⚠️ Predicted Outcome

            # {prediction}

            **Prediction Confidence:
            {confidence:.2f}%**
            """
        )

        recommendation = """
        The model predicts a Complicated outcome.

        • Monitor the patient closely.

        • Review clinical measurements.

        • Consider additional evaluation
          according to clinical judgment.
        """


    else:

        st.error(
            f"""
            ### 🚨 Predicted Outcome

            # {prediction}

            **Prediction Confidence:
            {confidence:.2f}%**
            """
        )

        recommendation = """
        The model predicts a Deceased outcome.

        This is a machine-learning prediction only
        and must not be interpreted as a clinical diagnosis.

        • Immediate professional medical evaluation
          is required for any real patient.
        """


    st.divider()


    # ========================================================
    # PROBABILITY CHART
    # ========================================================

    st.subheader(
        "📊 Prediction Probabilities"
    )

    probability_df = pd.DataFrame(
        {
            "Outcome": classes,
            "Probability": probabilities * 100
        }
    )

    fig = px.bar(
        probability_df,
        x="Outcome",
        y="Probability",
        color="Probability",
        text="Probability",
        color_continuous_scale="Viridis"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=450,
        showlegend=False,
        xaxis_title="Outcome",
        yaxis_title="Probability (%)",
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    st.divider()

    st.subheader(
        "📈 Prediction Confidence"
    )

    confidence_level = (
        "🟢 High"
        if confidence >= 80
        else "🟡 Medium"
        if confidence >= 60
        else "🔴 Low"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

    with col2:

        st.metric(
            "Confidence Level",
            confidence_level
        )


    # ========================================================
    # PATIENT SUMMARY
    # ========================================================

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🧾 Patient Summary"
        )

        st.write(
            f"**Age:** {age}"
        )

        st.write(
            f"**Gender:** {gender}"
        )

        st.write(
            f"**Diagnosis:** {diagnosis_name}"
        )

        st.write(
            f"**Diagnosis ID:** {diagnosis_id}"
        )

        st.write(
            f"**Blood Pressure:** "
            f"{blood_pressure:.2f}"
        )

        st.write(
            f"**Blood Sugar:** "
            f"{blood_sugar:.2f}"
        )

        st.write(
            f"**Cholesterol:** "
            f"{cholesterol:.2f}"
        )

        st.write(
            f"**Creatinine:** "
            f"{creatinine:.2f}"
        )

        st.write(
            f"**Hemoglobin:** "
            f"{hemoglobin:.2f}"
        )

        st.write(
            f"**Vitamin D:** "
            f"{vitamin_d:.2f}"
        )


    with col2:

        st.subheader(
            "🩺 Model Interpretation"
        )

        st.info(
            recommendation
        )


    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.divider()

    st.subheader(
        "📄 Prediction Report"
    )

    report = f"""
Hospital Patient Outcome Prediction Report
==========================================

Predicted Outcome : {prediction}

Prediction Confidence : {confidence:.2f}%

------------------------------------------

Patient Information

Age : {age}

Gender : {gender}

Diagnosis : {diagnosis_name}

Diagnosis ID : {diagnosis_id}

Blood Pressure : {blood_pressure:.2f}

Blood Sugar : {blood_sugar:.2f}

Cholesterol : {cholesterol:.2f}

Creatinine : {creatinine:.2f}

Hemoglobin : {hemoglobin:.2f}

Vitamin D : {vitamin_d:.2f}

------------------------------------------

Model

Algorithm : Random Forest

Test Accuracy : 83.30%

------------------------------------------

This prediction is generated using a machine
learning model trained on synthetic healthcare
data.

It is intended for educational and analytical
purposes only and must not be used as a substitute
for professional medical advice or clinical
decision-making.
"""


    st.download_button(
        label="📥 Download Prediction Report",
        data=report,
        file_name="prediction_report.txt",
        mime="text/plain",
        use_container_width=True
    )