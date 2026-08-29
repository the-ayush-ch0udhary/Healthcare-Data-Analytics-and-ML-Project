"""
Hospital Analytics Dashboard
Comprehensive clinical, demographic, operational, and financial analytics.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
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
    plotly_template,
    render_sidebar,
)

st.set_page_config(
    page_title="Hospital Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

apply_theme()

# ============================================================
# DATA INGESTION
# ============================================================

df = load_dataset()
meta = get_model_metadata()

if df.empty:
    st.error("Healthcare dataset could not be loaded. Please ensure data/processed/healthcare_cleaned.csv exists.")
    st.stop()

render_sidebar(
    total_patients=len(df),
    model_accuracy=meta.get("accuracy_pct", "83.30%"),
    total_diagnoses=df["DiagnosisName"].nunique(),
)

PLOTLY_TEMPLATE = plotly_template()

# ============================================================
# HEADER
# ============================================================

st.title("📊 Hospital Analytics & Clinical Intelligence")
st.write(
    "Interactive analytics exploring patient demographics, clinical biomarkers, hospital operations, and treatment outcomes."
)

st.divider()

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("### 🔎 Filter Patients")

if st.sidebar.button("🔄 Reset Filters", use_container_width=True):
    st.session_state.filter_gender = "All"
    st.session_state.filter_diagnosis = "All"
    st.session_state.filter_outcome = "All"
    st.session_state.filter_age = (int(df["Age"].min()), int(df["Age"].max()))
    st.rerun()

gender_options = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
diagnosis_options = ["All"] + sorted(df["DiagnosisName"].dropna().unique().tolist())
outcome_options = ["All"] + sorted(df["OutcomeName"].dropna().unique().tolist())

min_age_val = int(df["Age"].min())
max_age_val = int(df["Age"].max())

selected_gender = st.sidebar.selectbox("Gender", gender_options, key="filter_gender")
selected_diagnosis = st.sidebar.selectbox("Diagnosis", diagnosis_options, key="filter_diagnosis")
selected_outcome = st.sidebar.selectbox("Outcome", outcome_options, key="filter_outcome")

selected_age = st.sidebar.slider(
    "Age Range",
    min_value=min_age_val,
    max_value=max_age_val,
    value=(min_age_val, max_age_val),
    key="filter_age",
)

# Apply filters
filtered_df = df.copy()

if selected_gender != "All":
    filtered_df = filtered_df[filtered_df["Gender"] == selected_gender]

if selected_diagnosis != "All":
    filtered_df = filtered_df[filtered_df["DiagnosisName"] == selected_diagnosis]

if selected_outcome != "All":
    filtered_df = filtered_df[filtered_df["OutcomeName"] == selected_outcome]

filtered_df = filtered_df[
    (filtered_df["Age"] >= selected_age[0]) & (filtered_df["Age"] <= selected_age[1])
]

if filtered_df.empty:
    st.warning("⚠️ No patient records match the selected filter combination. Try resetting the filters.")
    st.stop()

# ============================================================
# EXECUTIVE SUMMARY KPIS
# ============================================================

total_filtered = len(filtered_df)
avg_age = round(filtered_df["Age"].mean(), 1)
avg_cost = int(filtered_df["TreatmentCost"].mean())
avg_stay = round(filtered_df["LengthOfStay"].mean(), 1)
recovery_pct = round((filtered_df["OutcomeName"] == "Recovered").mean() * 100, 1)

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric("Total Patients", f"{total_filtered:,}", delta=f"{round(total_filtered/len(df)*100, 1)}% of total")

with k2:
    st.metric("Average Age", f"{avg_age} Yrs")

with k3:
    st.metric("Avg Treatment Cost", f"₹ {avg_cost:,}")

with k4:
    st.metric("Avg Length of Stay", f"{avg_stay} Days")

with k5:
    st.metric("Recovery Rate", f"{recovery_pct}%")

st.divider()

# ============================================================
# TABS NAVIGATION
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Demographics & Outcomes",
        "🧪 Clinical Biomarkers & Correlations",
        "💰 Financial & Operational Analytics",
        "🔍 Patient Records Explorer",
    ]
)

# ============================================================
# TAB 1: DEMOGRAPHICS & OUTCOMES
# ============================================================

with tab1:
    st.markdown("### 👨‍⚕️ Patient Demographics & Outcome Distributions")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Age Distribution")
        fig_age = px.histogram(
            filtered_df,
            x="Age",
            nbins=20,
            color="OutcomeName",
            color_discrete_map={
                "Recovered": "#10b981",
                "Complicated": "#f59e0b",
                "Deceased": "#ef4444",
            },
            marginal="box",
        )
        fig_age.update_layout(
            template=PLOTLY_TEMPLATE,
            height=430,
            xaxis_title="Age (Years)",
            yaxis_title="Patient Count",
            legend_title="Outcome",
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with c2:
        st.subheader("Gender Breakdown")
        gender_agg = filtered_df["Gender"].value_counts().reset_index()
        gender_agg.columns = ["Gender", "Count"]
        fig_gender = px.pie(
            gender_agg,
            names="Gender",
            values="Count",
            hole=0.55,
            color_discrete_sequence=["#38bdf8", "#ec4899"],
        )
        fig_gender.update_layout(
            template=PLOTLY_TEMPLATE,
            height=430,
        )
        st.plotly_chart(fig_gender, use_container_width=True)

    st.markdown("---")

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Clinical Outcome Distribution")
        outcome_agg = filtered_df["OutcomeName"].value_counts().reset_index()
        outcome_agg.columns = ["Outcome", "Patients"]
        fig_outcome = px.bar(
            outcome_agg,
            x="Outcome",
            y="Patients",
            color="Outcome",
            text="Patients",
            color_discrete_map={
                "Recovered": "#10b981",
                "Complicated": "#f59e0b",
                "Deceased": "#ef4444",
            },
        )
        fig_outcome.update_traces(textposition="outside")
        fig_outcome.update_layout(
            template=PLOTLY_TEMPLATE,
            height=420,
            showlegend=False,
            xaxis_title="Clinical Outcome",
            yaxis_title="Patients",
        )
        st.plotly_chart(fig_outcome, use_container_width=True)

    with c4:
        st.subheader("Diagnosis Prevalence")
        diag_agg = filtered_df["DiagnosisName"].value_counts().reset_index()
        diag_agg.columns = ["Diagnosis", "Patients"]
        fig_diag = px.bar(
            diag_agg.sort_values(by="Patients"),
            x="Patients",
            y="Diagnosis",
            orientation="h",
            color="Patients",
            text="Patients",
            color_continuous_scale="Viridis",
        )
        fig_diag.update_traces(textposition="outside")
        fig_diag.update_layout(
            template=PLOTLY_TEMPLATE,
            height=420,
            xaxis_title="Patient Volume",
            yaxis_title="",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_diag, use_container_width=True)


# ============================================================
# TAB 2: CLINICAL BIOMARKERS & CORRELATIONS
# ============================================================

with tab2:
    st.markdown("### 🧪 Clinical Biomarker Profiling & Correlation Heatmap")

    # Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    with m_col1:
        st.metric("Blood Pressure", f"{filtered_df['Blood Pressure'].mean():.1f}", "mmHg")
    with m_col2:
        st.metric("Blood Sugar", f"{filtered_df['Blood Sugar'].mean():.1f}", "mg/dL")
    with m_col3:
        st.metric("Cholesterol", f"{filtered_df['Cholesterol'].mean():.1f}", "mg/dL")
    with m_col4:
        st.metric("Creatinine", f"{filtered_df['Creatinine'].mean():.2f}", "mg/dL")
    with m_col5:
        st.metric("Hemoglobin", f"{filtered_df['Hemoglobin'].mean():.2f}", "g/dL")
    with m_col6:
        st.metric("Vitamin D", f"{filtered_df['Vitamin D'].mean():.1f}", "ng/mL")

    st.markdown("---")

    bio_c1, bio_c2 = st.columns(2)

    with bio_c1:
        st.subheader("Biomarker Correlation Matrix")
        numeric_cols = [
            "Age",
            "Blood Pressure",
            "Blood Sugar",
            "Cholesterol",
            "Creatinine",
            "Hemoglobin",
            "Vitamin D",
            "LengthOfStay",
            "TreatmentCost",
        ]
        corr_matrix = filtered_df[numeric_cols].corr().round(2)

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        fig_corr.update_layout(
            template=PLOTLY_TEMPLATE,
            height=460,
            xaxis_title="",
            yaxis_title="",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with bio_c2:
        st.subheader("Biomarker Levels by Patient Outcome")
        selected_biomarker = st.selectbox(
            "Select Biomarker to Inspect:",
            [
                "Blood Pressure",
                "Blood Sugar",
                "Cholesterol",
                "Creatinine",
                "Hemoglobin",
                "Vitamin D",
                "Age",
            ],
        )

        fig_box = px.box(
            filtered_df,
            x="OutcomeName",
            y=selected_biomarker,
            color="OutcomeName",
            points="outliers",
            color_discrete_map={
                "Recovered": "#10b981",
                "Complicated": "#f59e0b",
                "Deceased": "#ef4444",
            },
        )
        fig_box.update_layout(
            template=PLOTLY_TEMPLATE,
            height=460,
            showlegend=False,
            xaxis_title="Patient Outcome",
            yaxis_title=selected_biomarker,
        )
        st.plotly_chart(fig_box, use_container_width=True)


# ============================================================
# TAB 3: FINANCIAL & OPERATIONAL ANALYTICS
# ============================================================

with tab3:
    st.markdown("### 💰 Financial Economics & Hospital Operations")

    fin_c1, fin_c2 = st.columns(2)

    with fin_c1:
        st.subheader("Treatment Cost by Diagnosis")
        cost_df = (
            filtered_df.groupby("DiagnosisName")["TreatmentCost"]
            .mean()
            .round(0)
            .reset_index()
            .sort_values(by="TreatmentCost", ascending=False)
        )

        fig_cost = px.bar(
            cost_df,
            x="DiagnosisName",
            y="TreatmentCost",
            color="TreatmentCost",
            text="TreatmentCost",
            color_continuous_scale="Purples",
        )
        fig_cost.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside")
        fig_cost.update_layout(
            template=PLOTLY_TEMPLATE,
            height=440,
            xaxis_title="Diagnosis",
            yaxis_title="Average Treatment Cost (₹)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    with fin_c2:
        st.subheader("Inpatient Length of Stay (Days) by Diagnosis")
        stay_df = (
            filtered_df.groupby("DiagnosisName")["LengthOfStay"]
            .mean()
            .round(1)
            .reset_index()
            .sort_values(by="LengthOfStay", ascending=False)
        )

        fig_stay = px.bar(
            stay_df,
            x="DiagnosisName",
            y="LengthOfStay",
            color="LengthOfStay",
            text="LengthOfStay",
            color_continuous_scale="Teal",
        )
        fig_stay.update_traces(texttemplate="%{text} Days", textposition="outside")
        fig_stay.update_layout(
            template=PLOTLY_TEMPLATE,
            height=440,
            xaxis_title="Diagnosis",
            yaxis_title="Length of Stay (Days)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_stay, use_container_width=True)

    st.markdown("---")

    st.subheader("Treatment Cost vs. Length of Stay by Outcome")
    fig_scatter = px.scatter(
        filtered_df,
        x="LengthOfStay",
        y="TreatmentCost",
        color="OutcomeName",
        size="Age",
        hover_data=["Name", "DiagnosisName", "Blood Pressure", "Blood Sugar"],
        color_discrete_map={
            "Recovered": "#10b981",
            "Complicated": "#f59e0b",
            "Deceased": "#ef4444",
        },
        trendline="ols",
    )
    fig_scatter.update_layout(
        template=PLOTLY_TEMPLATE,
        height=480,
        xaxis_title="Length of Stay (Days)",
        yaxis_title="Treatment Cost (₹)",
        legend_title="Outcome",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ============================================================
# TAB 4: PATIENT RECORDS EXPLORER
# ============================================================

with tab4:
    st.markdown("### 🔍 Inpatient Records & Data Table Explorer")

    search_query = st.text_input("🔍 Search by Patient Name or ID:", placeholder="e.g., Aarav or 105")

    display_table_df = filtered_df.copy()
    if search_query:
        display_table_df = display_table_df[
            display_table_df["Name"].astype(str).str.contains(search_query, case=False, na=False)
            | display_table_df["PatientID"].astype(str).str.contains(search_query, case=False, na=False)
        ]

    st.write(f"Showing **{len(display_table_df):,}** matching patient records:")

    st.dataframe(
        display_table_df,
        use_container_width=True,
        hide_index=True,
    )

    # CSV Download Button
    csv_data = display_table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Patient Records (CSV)",
        data=csv_data,
        file_name="hospital_filtered_patients.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()

# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div style="text-align:center; padding:15px 0 5px 0; color:var(--text-muted); font-size:12px;">
        Hospital Patient Analytics • Comprehensive Dashboard • Ayush &amp; Moon
    </div>
    """
)