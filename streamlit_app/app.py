"""
Hospital Patient Analytics & Outcome Prediction Platform
Main Entry Point
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# PATH SETUP & THEME
# ============================================================

APP_DIR = Path(__file__).resolve().parent
sys.path.append(str(APP_DIR))

from utils.theme import (
    apply_theme,
    get_model_metadata,
    load_dataset,
    plotly_template,
    render_sidebar,
)

st.set_page_config(
    page_title="Hospital Patient Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# ============================================================
# DYNAMIC DATA & METADATA
# ============================================================

df = load_dataset()
meta = get_model_metadata()

TOTAL_PATIENTS = len(df) if not df.empty else meta.get("dataset_total", 5000)
TOTAL_DIAGNOSES = df["DiagnosisName"].nunique() if not df.empty else 10
AVG_COST = int(df["TreatmentCost"].mean()) if not df.empty else 5240
AVG_STAY = round(df["LengthOfStay"].mean(), 1) if not df.empty else 5.5
MODEL_ACCURACY = meta.get("accuracy_pct", "83.30%")
MODEL_NAME = meta.get("model_name", "Random Forest")

# Outcome counts
if not df.empty:
    recovered_count = (df["OutcomeName"] == "Recovered").sum()
    complicated_count = (df["OutcomeName"] == "Complicated").sum()
    deceased_count = (df["OutcomeName"] == "Deceased").sum()
    recovery_rate = round((recovered_count / TOTAL_PATIENTS) * 100, 1)
    complicated_rate = round((complicated_count / TOTAL_PATIENTS) * 100, 1)
    deceased_rate = round((deceased_count / TOTAL_PATIENTS) * 100, 1)
else:
    recovered_count, complicated_count, deceased_count = 3250, 1250, 500
    recovery_rate, complicated_rate, deceased_rate = 65.0, 25.0, 10.0

render_sidebar(
    total_patients=TOTAL_PATIENTS,
    model_accuracy=MODEL_ACCURACY,
    total_diagnoses=TOTAL_DIAGNOSES,
)

# ============================================================
# HERO BANNER
# ============================================================

st.html(
    f"""
    <div style="
        padding:36px 32px 30px 32px;
        border:1px solid var(--card-border);
        border-radius:20px;
        background:var(--card-bg);
        box-shadow:0 12px 36px var(--card-shadow);
        margin-bottom:25px;
    ">
        <div style="
            font-size:44px;
            font-weight:800;
            line-height:1.15;
            margin-bottom:12px;
            background:linear-gradient(90deg, var(--heading-start), var(--heading-end));
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
        ">
            🏥 Hospital Patient Analytics
        </div>

        <div style="
            font-size:24px;
            font-weight:700;
            color:var(--text-primary);
            margin-bottom:12px;
        ">
            AI-Powered Healthcare Intelligence &amp; Clinical Decision Platform
        </div>

        <div style="
            font-size:16px;
            line-height:1.7;
            color:var(--text-secondary);
            max-width:920px;
            margin-bottom:24px;
        ">
            Analyze multi-dimensional inpatient records, uncover clinical biomarkers,
            audit hospital treatment costs, and forecast patient outcomes with high precision
            using calibrated Machine Learning models.
        </div>

        <div style="
            display:grid;
            grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));
            gap:14px;
        ">
            <div style="padding:16px; border:1px solid var(--card-border); border-radius:12px; background:var(--input-bg);">
                <div style="font-size:24px;">📊</div>
                <div style="font-size:15px; font-weight:700; margin-top:6px; color:var(--text-primary);">Healthcare Analytics</div>
                <div style="font-size:12px; margin-top:4px; color:var(--text-muted);">Demographics, KPIs &amp; financial trends</div>
            </div>

            <div style="padding:16px; border:1px solid var(--card-border); border-radius:12px; background:var(--input-bg);">
                <div style="font-size:24px;">🧪</div>
                <div style="font-size:15px; font-weight:700; margin-top:6px; color:var(--text-primary);">Biomarker Tracking</div>
                <div style="font-size:12px; margin-top:4px; color:var(--text-muted);">Clinical reference ranges &amp; risk flags</div>
            </div>

            <div style="padding:16px; border:1px solid var(--card-border); border-radius:12px; background:var(--input-bg);">
                <div style="font-size:24px;">🤖</div>
                <div style="font-size:15px; font-weight:700; margin-top:6px; color:var(--text-primary);">Outcome Prediction</div>
                <div style="font-size:12px; margin-top:4px; color:var(--text-muted);">Single &amp; batch patient risk classification</div>
            </div>

            <div style="padding:16px; border:1px solid var(--card-border); border-radius:12px; background:var(--input-bg);">
                <div style="font-size:24px;">📈</div>
                <div style="font-size:15px; font-weight:700; margin-top:6px; color:var(--text-primary);">Model Benchmarks</div>
                <div style="font-size:12px; margin-top:4px; color:var(--text-muted);">Multi-algorithm evaluation &amp; feature importance</div>
            </div>
        </div>
    </div>
    """
)

# ============================================================
# EXECUTIVE METRICS
# ============================================================

st.markdown("## 📊 Executive Summary")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="Total Patients",
        value=f"{TOTAL_PATIENTS:,}",
        delta="Clinical Records",
    )

with kpi2:
    st.metric(
        label="Model Accuracy",
        value=MODEL_ACCURACY,
        delta=MODEL_NAME,
    )

with kpi3:
    st.metric(
        label="Diagnoses Monitored",
        value=TOTAL_DIAGNOSES,
        delta="Disease Categories",
    )

with kpi4:
    st.metric(
        label="Avg Treatment Cost",
        value=f"₹ {AVG_COST:,}",
        delta=f"Avg Stay: {AVG_STAY} Days",
    )

st.divider()

# ============================================================
# OUTCOME DISTRIBUTION CARDS
# ============================================================

st.markdown("## 📌 Patient Clinical Outcomes Overview")

o1, o2, o3 = st.columns(3)

with o1:
    st.success(
        f"""
        ### ✅ Recovered
        **Total Patients:** {recovered_count:,}
        
        **Recovery Rate:** **{recovery_rate}%**
        
        Patients successfully discharged with resolved clinical status.
        """
    )

with o2:
    st.warning(
        f"""
        ### ⚠️ Complicated
        **Total Patients:** {complicated_count:,}
        
        **Complication Rate:** **{complicated_rate}%**
        
        Patients experiencing post-treatment complications or extended monitoring.
        """
    )

with o3:
    st.error(
        f"""
        ### 🚨 Deceased
        **Total Patients:** {deceased_count:,}
        
        **Mortality Rate:** **{deceased_rate}%**
        
        Severe cases requiring intensive clinical review and mortality audits.
        """
    )

st.divider()

# ============================================================
# APPLICATION MODULES NAVIGATION
# ============================================================

st.markdown("## 🚀 Application Navigation Hub")

mod1, mod2, mod3, mod4 = st.columns(4)

with mod1:
    with st.container(border=True):
        st.markdown("### 📊 Dashboard")
        st.write(
            "Interactive charts, diagnosis distribution, clinical biomarker correlations, and searchable records."
        )
        st.page_link("pages/Dashboard.py", label="Open Dashboard", icon="📊")

with mod2:
    with st.container(border=True):
        st.markdown("### 🤖 Predict")
        st.write(
            "Predict patient hospital outcomes with clinical biomarker ranges, patient presets, and batch CSV inference."
        )
        st.page_link("pages/Predict.py", label="Launch Prediction", icon="🤖")

with mod3:
    with st.container(border=True):
        st.markdown("### 📈 Model Metrics")
        st.write(
            "Inspect accuracy, precision-recall, interactive confusion matrix, feature importance, and multi-model benchmark."
        )
        st.page_link("pages/Model_Performance.py", label="View Performance", icon="📈")

with mod4:
    with st.container(border=True):
        st.markdown("### ℹ️ About")
        st.write(
            "System architecture, clinical reference ranges guide, dataset schemas, and development methodology."
        )
        st.page_link("pages/About.py", label="Explore Platform", icon="ℹ️")

st.divider()

# ============================================================
# DATASET AT A GLANCE
# ============================================================

if not df.empty:
    st.markdown("## 📈 Quick Analytics Snapshot")

    snap1, snap2 = st.columns(2)

    with snap1:
        st.subheader("Outcome Breakdown by Diagnosis")
        diag_out = (
            df.groupby(["DiagnosisName", "OutcomeName"])
            .size()
            .reset_index(name="Count")
        )
        fig_diag = px.bar(
            diag_out,
            x="DiagnosisName",
            y="Count",
            color="OutcomeName",
            barmode="stack",
            color_discrete_map={
                "Recovered": "#10b981",
                "Complicated": "#f59e0b",
                "Deceased": "#ef4444",
            },
        )
        fig_diag.update_layout(
            template=plotly_template(),
            height=380,
            xaxis_title="",
            yaxis_title="Patients",
            legend_title="Outcome",
        )
        st.plotly_chart(fig_diag, use_container_width=True)

    with snap2:
        st.subheader("Key Clinical Biomarkers Distribution")
        lab_summary = pd.DataFrame(
            {
                "Biomarker": [
                    "Blood Pressure (mmHg)",
                    "Blood Sugar (mg/dL)",
                    "Cholesterol (mg/dL)",
                    "Creatinine (mg/dL)",
                    "Hemoglobin (g/dL)",
                    "Vitamin D (ng/mL)",
                ],
                "Mean Value": [
                    f"{df['Blood Pressure'].mean():.1f}",
                    f"{df['Blood Sugar'].mean():.1f}",
                    f"{df['Cholesterol'].mean():.1f}",
                    f"{df['Creatinine'].mean():.2f}",
                    f"{df['Hemoglobin'].mean():.2f}",
                    f"{df['Vitamin D'].mean():.1f}",
                ],
                "Normal Range": [
                    "90 - 120 mmHg",
                    "70 - 99 mg/dL",
                    "< 200 mg/dL",
                    "0.6 - 1.2 mg/dL",
                    "12.0 - 17.5 g/dL",
                    "30 - 100 ng/mL",
                ],
            }
        )
        st.dataframe(lab_summary, use_container_width=True, hide_index=True)

st.divider()

# ============================================================
# DISCLAIMER & FOOTER
# ============================================================

st.caption(
    """
    ⚠️ **Medical Disclaimer**: This application is developed for educational, analytical, and clinical decision support demonstration purposes.
    Predictions and analytics generated are based on synthetic patient records and should not be used as the sole basis for actual patient clinical management.
    """
)

st.html(
    """
    <div style="
        margin-top:30px;
        padding:25px 20px 20px 20px;
        text-align:center;
        border-top:1px solid var(--divider);
    ">
        <div style="font-size:20px; font-weight:800; color:var(--text-primary); margin-bottom:8px;">
            🏥 Hospital Patient Analytics Platform
        </div>
        <div style="font-size:13px; color:var(--text-secondary); margin-bottom:6px;">
            Engineered with <strong>Python • Pandas • Scikit-Learn • Plotly • Streamlit</strong>
        </div>
        <div style="font-size:12px; color:var(--text-muted);">
            Developed by <strong>Ayush &amp; Moon</strong> • Version 2.0
        </div>
    </div>
    """
)