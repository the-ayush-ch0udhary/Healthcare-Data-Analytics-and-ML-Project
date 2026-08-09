import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hospital Dashboard",
    page_icon="📊",
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

    return pd.read_csv(DATA_PATH)


df = load_data()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📊 Hospital Analytics Dashboard")

st.write(
    """
    Explore hospital records through interactive charts,
    healthcare KPIs and patient analytics.
    """
)

st.divider()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Dashboard Filters")


gender_options = (
    ["All"]
    + sorted(
        df["Gender"]
        .dropna()
        .unique()
        .tolist()
    )
)

diagnosis_options = (
    ["All"]
    + sorted(
        df["DiagnosisName"]
        .dropna()
        .unique()
        .tolist()
    )
)

outcome_options = (
    ["All"]
    + sorted(
        df["OutcomeName"]
        .dropna()
        .unique()
        .tolist()
    )
)


gender = st.sidebar.selectbox(
    "Gender",
    gender_options
)

diagnosis = st.sidebar.selectbox(
    "Diagnosis",
    diagnosis_options
)

outcome = st.sidebar.selectbox(
    "Outcome",
    outcome_options
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df.copy()


if gender != "All":

    filtered_df = filtered_df[
        filtered_df["Gender"] == gender
    ]


if diagnosis != "All":

    filtered_df = filtered_df[
        filtered_df["DiagnosisName"] == diagnosis
    ]


if outcome != "All":

    filtered_df = filtered_df[
        filtered_df["OutcomeName"] == outcome
    ]


# ============================================================
# EMPTY DATA CHECK
# ============================================================

if filtered_df.empty:

    st.warning(
        "No patients match the selected filters."
    )

    st.stop()


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

st.markdown(
    "## 📈 Executive Summary"
)


total_patients = len(
    filtered_df
)


average_age = round(
    filtered_df["Age"].mean(),
    1
)


average_cost = int(
    filtered_df["TreatmentCost"].mean()
)


recovery_rate = round(
    (
        filtered_df["OutcomeName"]
        == "Recovered"
    ).mean() * 100,
    1
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "👨‍⚕️ Patients",
        f"{total_patients:,}"
    )


with col2:

    st.metric(
        "🎂 Average Age",
        average_age
    )


with col3:

    st.metric(
        "💰 Average Treatment Cost",
        f"₹ {average_cost:,}"
    )


with col4:

    st.metric(
        "✅ Recovery Rate",
        f"{recovery_rate}%"
    )


st.divider()


# ============================================================
# PATIENT DEMOGRAPHICS
# ============================================================

st.markdown(
    "## 👨‍⚕️ Patient Demographics"
)


left_chart, right_chart = st.columns(2)


# ============================================================
# AGE DISTRIBUTION
# ============================================================

with left_chart:

    st.subheader(
        "📈 Age Distribution"
    )

    fig = px.histogram(
        filtered_df,
        x="Age",
        nbins=15,
        color_discrete_sequence=[
            "#3B82F6"
        ]
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=420,
        xaxis_title="Age",
        yaxis_title="Patients"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# GENDER DISTRIBUTION
# ============================================================

with right_chart:

    st.subheader(
        "👥 Gender Distribution"
    )

    gender_df = (
        filtered_df["Gender"]
        .value_counts()
        .reset_index()
    )

    gender_df.columns = [
        "Gender",
        "Patients"
    ]

    fig = px.pie(
        gender_df,
        names="Gender",
        values="Patients",
        hole=0.55,
        color_discrete_sequence=(
            px.colors.qualitative.Set2
        )
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=420
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


st.divider()


# ============================================================
# CLINICAL ANALYTICS
# ============================================================

st.markdown(
    "## 🏥 Clinical Analytics"
)


left_chart, right_chart = st.columns(2)


# ============================================================
# OUTCOME DISTRIBUTION
# ============================================================

with left_chart:

    st.subheader(
        "📊 Patient Outcome Distribution"
    )

    outcome_df = (
        filtered_df["OutcomeName"]
        .value_counts()
        .reset_index()
    )

    outcome_df.columns = [
        "Outcome",
        "Patients"
    ]

    fig = px.bar(
        outcome_df,
        x="Outcome",
        y="Patients",
        color="Outcome",
        text="Patients",
        color_discrete_sequence=(
            px.colors.qualitative.Bold
        )
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=420,
        showlegend=False,
        xaxis_title="Outcome",
        yaxis_title="Patients"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# DIAGNOSIS DISTRIBUTION
# ============================================================

with right_chart:

    st.subheader(
        "🩺 Diagnosis Distribution"
    )

    diagnosis_df = (
        filtered_df["DiagnosisName"]
        .value_counts()
        .reset_index()
    )

    diagnosis_df.columns = [
        "Diagnosis",
        "Patients"
    ]

    fig = px.bar(
        diagnosis_df,
        x="Patients",
        y="Diagnosis",
        orientation="h",
        color="Patients",
        text="Patients",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=420,
        yaxis_title="",
        xaxis_title="Patients"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


st.divider()


# ============================================================
# CLINICAL MEASUREMENTS
# ============================================================

st.markdown(
    "## 🧪 Clinical Measurements"
)


measurement_cols = st.columns(4)


measurements = [
    (
        "Blood Pressure",
        "Blood Pressure",
        "mmHg"
    ),
    (
        "Blood Sugar",
        "Blood Sugar",
        "mg/dL"
    ),
    (
        "Cholesterol",
        "Cholesterol",
        "mg/dL"
    ),
    (
        "Creatinine",
        "Creatinine",
        "mg/dL"
    )
]


for column, (
    label,
    field,
    unit
) in zip(
    measurement_cols,
    measurements
):

    with column:

        value = filtered_df[field].mean()

        st.metric(
            label,
            f"{value:.2f}",
            unit
        )


measurement_cols_2 = st.columns(2)


with measurement_cols_2[0]:

    st.metric(
        "Hemoglobin",
        f"{filtered_df['Hemoglobin'].mean():.2f}",
        "g/dL"
    )


with measurement_cols_2[1]:

    st.metric(
        "Vitamin D",
        f"{filtered_df['Vitamin D'].mean():.2f}",
        "ng/mL"
    )


st.divider()


# ============================================================
# FINANCIAL ANALYTICS
# ============================================================

st.markdown(
    "## 💰 Financial Analytics"
)


left_chart, right_chart = st.columns(2)


# ============================================================
# TREATMENT COST
# ============================================================

with left_chart:

    st.subheader(
        "💰 Average Treatment Cost by Diagnosis"
    )

    cost_df = (
        filtered_df
        .groupby("DiagnosisName")["TreatmentCost"]
        .mean()
        .round(0)
        .reset_index()
        .sort_values(
            by="TreatmentCost",
            ascending=False
        )
    )

    fig = px.bar(
        cost_df,
        x="DiagnosisName",
        y="TreatmentCost",
        color="TreatmentCost",
        text="TreatmentCost",
        color_continuous_scale="Blues"
    )

    fig.update_traces(
        texttemplate="₹%{text:.0f}",
        textposition="outside"
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=450,
        xaxis_title="Diagnosis",
        yaxis_title="Average Cost (₹)",
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# LENGTH OF STAY
# ============================================================

with right_chart:

    st.subheader(
        "🏥 Average Length of Stay"
    )

    stay_df = (
        filtered_df
        .groupby("DiagnosisName")["LengthOfStay"]
        .mean()
        .round(1)
        .reset_index()
        .sort_values(
            by="LengthOfStay",
            ascending=False
        )
    )

    fig = px.bar(
        stay_df,
        x="DiagnosisName",
        y="LengthOfStay",
        color="LengthOfStay",
        text="LengthOfStay",
        color_continuous_scale="Teal"
    )

    fig.update_traces(
        texttemplate="%{text} Days",
        textposition="outside"
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=450,
        xaxis_title="Diagnosis",
        yaxis_title="Days",
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


st.divider()


# ============================================================
# OUTCOME BY DIAGNOSIS
# ============================================================

st.markdown(
    "## 🩺 Outcome by Diagnosis"
)


outcome_diagnosis_df = (
    filtered_df
    .groupby(
        [
            "DiagnosisName",
            "OutcomeName"
        ]
    )
    .size()
    .reset_index(
        name="Patients"
    )
)


fig = px.bar(
    outcome_diagnosis_df,
    x="DiagnosisName",
    y="Patients",
    color="OutcomeName",
    barmode="group",
    text="Patients"
)

fig.update_layout(
    template=PLOTLY_TEMPLATE,
    height=500,
    xaxis_title="Diagnosis",
    yaxis_title="Patients",
    legend_title="Outcome"
)

fig.update_traces(
    textposition="outside"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


st.divider()


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.markdown(
    "## 💡 Business Insights"
)


left, right = st.columns(2)


# ============================================================
# FINANCIAL INSIGHTS
# ============================================================

with left:

    cost_by_diagnosis = (
        filtered_df
        .groupby("DiagnosisName")["TreatmentCost"]
        .mean()
    )

    stay_by_diagnosis = (
        filtered_df
        .groupby("DiagnosisName")["LengthOfStay"]
        .mean()
    )

    highest_cost = (
        cost_by_diagnosis.idxmax()
    )

    highest_cost_value = (
        cost_by_diagnosis.max()
    )

    longest_stay = (
        stay_by_diagnosis.idxmax()
    )

    longest_days = (
        stay_by_diagnosis.max()
    )

    st.info(
        f"""
        ### 💰 Financial Insights

        **Highest Average Treatment Cost**

        🏥 {highest_cost}

        ₹ {highest_cost_value:,.0f}

        ---

        **Longest Average Stay**

        🏥 {longest_stay}

        🛏 {longest_days:.1f} Days
        """
    )


# ============================================================
# CLINICAL INSIGHTS
# ============================================================

with right:

    diagnosis_counts = (
        filtered_df["DiagnosisName"]
        .value_counts()
    )

    most_common = (
        diagnosis_counts.idxmax()
    )

    patient_count = (
        diagnosis_counts.max()
    )

    recovery = (
        (
            filtered_df["OutcomeName"]
            == "Recovered"
        ).mean() * 100
    )

    st.success(
        f"""
        ### 📈 Clinical Insights

        **Most Common Diagnosis**

        🩺 {most_common}

        👨‍⚕️ {patient_count} Patients

        ---

        **Recovery Rate**

        ✅ {recovery:.1f}%
        """
    )


st.divider()


# ============================================================
# DATASET INFORMATION
# ============================================================

st.markdown(
    "## 📋 Dataset Information"
)


info1, info2, info3 = st.columns(3)


with info1:

    st.metric(
        "Total Dataset Records",
        f"{len(df):,}"
    )


with info2:

    st.metric(
        "Filtered Records",
        f"{len(filtered_df):,}"
    )


with info3:

    st.metric(
        "Dataset Features",
        len(df.columns)
    )


st.divider()


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div style="
    margin-top: 50px;
    padding: 30px 20px 25px 20px;
    text-align: center;
    border-top: 1px solid rgba(128, 128, 128, 0.25);
">

    <div style="
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 10px;
    ">
        🏥 Hospital Patient Analytics
    </div>

    <div style="
        font-size: 14px;
        margin-bottom: 8px;
        opacity: 0.8;
    ">
        Developed using
        <strong>Python • Pandas • Plotly • Streamlit • Scikit-Learn</strong>
    </div>

    <div style="
        font-size: 13px;
        margin-bottom: 10px;
        opacity: 0.65;
    ">
        Healthcare Data Analytics & Machine Learning Project
    </div>

    <div style="
        font-size: 14px;
        font-weight: 600;
        opacity: 0.9;
    ">
        Developed by Ayush & Moon
    </div>

    <div style="
        font-size: 12px;
        margin-top: 10px;
        opacity: 0.5;
    ">
        Version 1.0
    </div>

</div>
""")