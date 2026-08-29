"""
Patient Outcome Prediction Engine
Single Patient Prediction with Clinical Biomarker Analysis & Batch CSV Inference
"""

import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# PAGE CONFIG & THEME
# ============================================================

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(APP_DIR))

from utils.theme import (
    apply_theme,
    get_clinical_status,
    get_model_metadata,
    load_dataset,
    plotly_template,
    render_sidebar,
)

st.set_page_config(
    page_title="Patient Outcome Prediction",
    page_icon="🤖",
    layout="wide",
)

apply_theme()

BASE_DIR = APP_DIR.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

df = load_dataset()
meta = get_model_metadata()

render_sidebar(
    total_patients=len(df) if not df.empty else meta.get("dataset_total", 5000),
    model_accuracy=meta.get("accuracy_pct", "83.30%"),
    total_diagnoses=df["DiagnosisName"].nunique() if not df.empty else 10,
)

PLOTLY_TEMPLATE = plotly_template()

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_prediction_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


model = load_prediction_model()

# ============================================================
# HEADER
# ============================================================

head_col1, head_col2 = st.columns([3, 1])

with head_col1:
    st.title("🤖 Patient Outcome Prediction Engine")
    st.write(
        "Forecast patient hospital outcome (Recovered, Complicated, Deceased) with clinical biomarker risk assessment and batch prediction capabilities."
    )

with head_col2:
    if model is not None:
        st.success(
            f"""
            ### 🟢 Model Active
            **Model:** {meta.get('model_name', 'Random Forest')}
            
            **Accuracy:** **{meta.get('accuracy_pct', '83.30%')}**
            """
        )
    else:
        st.error("Model file not found. Please train the model first.")

st.divider()

if model is None:
    st.stop()

# ============================================================
# PRESET PROFILES DATA
# ============================================================

PRESETS = {
    "Custom / Manual Input": {
        "age": 45,
        "gender": "M",
        "diagnosis_name": "Hypertension",
        "bp": 120.0,
        "sugar": 95.0,
        "chol": 185.0,
        "creat": 1.0,
        "hemo": 14.5,
        "vitd": 35.0,
    },
    "🟢 Profile A: Healthy Routine Patient": {
        "age": 34,
        "gender": "F",
        "diagnosis_name": "Asthma",
        "bp": 115.0,
        "sugar": 88.0,
        "chol": 165.0,
        "creat": 0.85,
        "hemo": 14.2,
        "vitd": 42.0,
    },
    "🟡 Profile B: Chronic Diabetic & Renal Risk": {
        "age": 68,
        "gender": "M",
        "diagnosis_name": "Kidney Disease",
        "bp": 142.0,
        "sugar": 195.0,
        "chol": 215.0,
        "creat": 2.30,
        "hemo": 11.2,
        "vitd": 22.0,
    },
    "🔴 Profile C: High-Risk Cardiovascular Inpatient": {
        "age": 78,
        "gender": "M",
        "diagnosis_name": "Heart Disease",
        "bp": 165.0,
        "sugar": 145.0,
        "chol": 260.0,
        "creat": 1.85,
        "hemo": 10.1,
        "vitd": 16.0,
    },
}

# ============================================================
# PREDICTION MODE TABS
# ============================================================

pred_tab1, pred_tab2 = st.tabs(
    [
        "🧑‍⚕️ Single Patient Prediction & Clinical Risk",
        "📁 Batch Patient Prediction (CSV Upload)",
    ]
)

# ============================================================
# TAB 1: SINGLE PATIENT PREDICTION
# ============================================================

with pred_tab1:
    st.markdown("### 🧑‍⚕️ Clinical Patient Parameters")

    # Diagnosis Lookup
    diagnosis_map = {
        1: "Hypertension",
        2: "Diabetes",
        3: "Heart Disease",
        4: "Asthma",
        5: "Stroke",
        6: "COPD",
        7: "Cancer",
        8: "Arthritis",
        9: "Kidney Disease",
        10: "Liver Disease",
    }
    diagnosis_name_to_id = {v: k for k, v in diagnosis_map.items()}

    # Preset Selector
    preset_choice = st.selectbox(
        "⚡ Quick Load Clinical Preset Profile:",
        list(PRESETS.keys()),
        index=0,
        help="Select a predefined patient profile for rapid clinical validation.",
    )
    p_data = PRESETS[preset_choice]

    col_demo, col_labs = st.columns(2)

    with col_demo:
        st.markdown("#### 👤 Demographics & Primary Diagnosis")

        age = st.slider("Patient Age (Years)", min_value=18, max_value=100, value=int(p_data["age"]))
        gender = st.selectbox("Biological Gender", ["M", "F"], index=0 if p_data["gender"] == "M" else 1)

        diag_names = list(diagnosis_map.values())
        diag_idx = diag_names.index(p_data["diagnosis_name"]) if p_data["diagnosis_name"] in diag_names else 0
        selected_diag_name = st.selectbox("Primary Diagnosis", diag_names, index=diag_idx)
        selected_diag_id = diagnosis_name_to_id[selected_diag_name]

    with col_labs:
        st.markdown("#### 🧪 Laboratory Biomarkers & Clinical Status")

        # Blood Pressure
        bp_val = st.number_input("Blood Pressure (Systolic mmHg)", min_value=60.0, max_value=240.0, value=float(p_data["bp"]), step=1.0)
        bp_stat, bp_badge, bp_desc = get_clinical_status("Blood Pressure", bp_val)
        st.markdown(f"Status: <span class='{bp_badge}'>{bp_stat}</span> - <small>{bp_desc}</small>", unsafe_allow_html=True)

        # Blood Sugar
        sugar_val = st.number_input("Blood Sugar (Fasting mg/dL)", min_value=40.0, max_value=400.0, value=float(p_data["sugar"]), step=1.0)
        sugar_stat, sugar_badge, sugar_desc = get_clinical_status("Blood Sugar", sugar_val)
        st.markdown(f"Status: <span class='{sugar_badge}'>{sugar_stat}</span> - <small>{sugar_desc}</small>", unsafe_allow_html=True)

        # Cholesterol
        chol_val = st.number_input("Total Cholesterol (mg/dL)", min_value=80.0, max_value=450.0, value=float(p_data["chol"]), step=1.0)
        chol_stat, chol_badge, chol_desc = get_clinical_status("Cholesterol", chol_val)
        st.markdown(f"Status: <span class='{chol_badge}'>{chol_stat}</span> - <small>{chol_desc}</small>", unsafe_allow_html=True)

        # Creatinine
        creat_val = st.number_input("Serum Creatinine (mg/dL)", min_value=0.2, max_value=8.0, value=float(p_data["creat"]), step=0.05)
        creat_stat, creat_badge, creat_desc = get_clinical_status("Creatinine", creat_val)
        st.markdown(f"Status: <span class='{creat_badge}'>{creat_stat}</span> - <small>{creat_desc}</small>", unsafe_allow_html=True)

        # Hemoglobin
        hemo_val = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=22.0, value=float(p_data["hemo"]), step=0.1)
        hemo_stat, hemo_badge, hemo_desc = get_clinical_status("Hemoglobin", hemo_val)
        st.markdown(f"Status: <span class='{hemo_badge}'>{hemo_stat}</span> - <small>{hemo_desc}</small>", unsafe_allow_html=True)

        # Vitamin D
        vitd_val = st.number_input("Vitamin D (ng/mL)", min_value=4.0, max_value=120.0, value=float(p_data["vitd"]), step=0.5)
        vitd_stat, vitd_badge, vitd_desc = get_clinical_status("Vitamin D", vitd_val)
        st.markdown(f"Status: <span class='{vitd_badge}'>{vitd_stat}</span> - <small>{vitd_desc}</small>", unsafe_allow_html=True)

    st.markdown("---")

    # Single Prediction Trigger
    if st.button("🔮 Predict Patient Outcome & Assess Risk", type="primary", use_container_width=True):
        input_data = pd.DataFrame(
            {
                "Age": [age],
                "Gender": [gender],
                "DiagnosisID": [str(selected_diag_id)],
                "Blood Pressure": [bp_val],
                "Blood Sugar": [sugar_val],
                "Cholesterol": [chol_val],
                "Creatinine": [creat_val],
                "Hemoglobin": [hemo_val],
                "Vitamin D": [vitd_val],
            }
        )

        pred_outcome = model.predict(input_data)[0]
        pred_probs = model.predict_proba(input_data)[0]
        classes = list(model.classes_)
        confidence_val = max(pred_probs) * 100

        st.markdown("## 🎯 Clinical Prediction Result")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if pred_outcome == "Recovered":
                st.success(
                    f"""
                    ### ✅ Predicted Outcome: **{pred_outcome}**
                    
                    **Confidence Score:** **{confidence_val:.2f}%**
                    
                    **Clinical Classification:** Favorable prognosis. Expected standard recovery trajectory under primary care protocols.
                    """
                )
            elif pred_outcome == "Complicated":
                st.warning(
                    f"""
                    ### ⚠️ Predicted Outcome: **{pred_outcome}**
                    
                    **Confidence Score:** **{confidence_val:.2f}%**
                    
                    **Clinical Classification:** High risk of secondary complications. Close hemodynamic and laboratory monitoring recommended.
                    """
                )
            else:
                st.error(
                    f"""
                    ### 🚨 Predicted Outcome: **{pred_outcome}**
                    
                    **Confidence Score:** **{confidence_val:.2f}%**
                    
                    **Clinical Classification:** Critical severity alert. Immediate multidisciplinary evaluation and intensive care auditing indicated.
                    """
                )

            # Recommendations
            st.info(
                f"""
                **Clinical Decision Support Guidelines:**
                - **Blood Pressure Status:** {bp_stat} ({bp_val:.1f} mmHg)
                - **Glycemic Control:** {sugar_stat} ({sugar_val:.1f} mg/dL)
                - **Renal Function:** {creat_stat} ({creat_val:.2f} mg/dL)
                - **Hematology:** {hemo_stat} ({hemo_val:.2f} g/dL)
                """
            )

        with res_col2:
            st.subheader("Outcome Probability Distribution")
            prob_df = pd.DataFrame(
                {"Outcome": classes, "Probability": [p * 100 for p in pred_probs]}
            )
            fig_prob = px.bar(
                prob_df,
                x="Outcome",
                y="Probability",
                color="Outcome",
                text="Probability",
                color_discrete_map={
                    "Recovered": "#10b981",
                    "Complicated": "#f59e0b",
                    "Deceased": "#ef4444",
                },
            )
            fig_prob.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_prob.update_layout(
                template=PLOTLY_TEMPLATE,
                height=360,
                showlegend=False,
                yaxis_title="Probability (%)",
                xaxis_title="",
                yaxis_range=[0, 100],
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        # Download Report
        report_text = f"""====================================================
HOSPITAL PATIENT OUTCOME PREDICTION REPORT
Generated by Hospital Patient Analytics Intelligence Platform
====================================================

PATIENT CLINICAL SUMMARY
----------------------------------------------------
Age                 : {age} Years
Gender              : {gender}
Diagnosis           : {selected_diag_name} (ID: {selected_diag_id})

LABORATORY BIOMARKERS
----------------------------------------------------
Blood Pressure      : {bp_val:.1f} mmHg [{bp_stat}]
Blood Sugar (Fast)  : {sugar_val:.1f} mg/dL [{sugar_stat}]
Total Cholesterol   : {chol_val:.1f} mg/dL [{chol_stat}]
Serum Creatinine    : {creat_val:.2f} mg/dL [{creat_stat}]
Hemoglobin          : {hemo_val:.2f} g/dL [{hemo_stat}]
Vitamin D           : {vitd_val:.1f} ng/mL [{vitd_stat}]

MODEL PREDICTION & RISK ASSESSMENT
----------------------------------------------------
Predicted Outcome   : {pred_outcome}
Confidence Score    : {confidence_val:.2f}%
Model Architecture  : {meta.get('model_name', 'Random Forest')}
Test Accuracy       : {meta.get('accuracy_pct', '83.30%')}

PROBABILITIES BREAKDOWN
----------------------------------------------------
"""
        for cls, p in zip(classes, pred_probs):
            report_text += f"{cls:18s}: {p*100:.2f}%\n"

        report_text += """
====================================================
DISCLAIMER: This report is generated by a Machine Learning model
for clinical decision support and educational analytics. It is not
a replacement for direct clinical diagnosis by a licensed physician.
====================================================
"""
        st.download_button(
            label="📄 Download Patient Outcome Clinical Report (.txt)",
            data=report_text,
            file_name=f"patient_prediction_{pred_outcome.lower()}.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ============================================================
# TAB 2: BATCH PATIENT PREDICTION (CSV UPLOAD)
# ============================================================

with pred_tab2:
    st.markdown("### 📁 Batch Inpatient Prediction & Triage")
    st.write(
        "Upload a batch CSV file of inpatient records to perform bulk outcome prediction, risk scoring, and high-risk triage."
    )

    # Sample CSV Template Generation
    sample_df = pd.DataFrame(
        [
            {
                "PatientID": 1001,
                "Name": "Sample Patient 1",
                "Age": 55,
                "Gender": "M",
                "DiagnosisID": 1,
                "Blood Pressure": 128.5,
                "Blood Sugar": 105.0,
                "Cholesterol": 190.0,
                "Creatinine": 0.95,
                "Hemoglobin": 14.0,
                "Vitamin D": 32.0,
            },
            {
                "PatientID": 1002,
                "Name": "Sample Patient 2",
                "Age": 72,
                "Gender": "F",
                "DiagnosisID": 9,
                "Blood Pressure": 150.2,
                "Blood Sugar": 140.0,
                "Cholesterol": 220.0,
                "Creatinine": 2.10,
                "Hemoglobin": 10.8,
                "Vitamin D": 20.0,
            },
            {
                "PatientID": 1003,
                "Name": "Sample Patient 3",
                "Age": 41,
                "Gender": "M",
                "DiagnosisID": 4,
                "Blood Pressure": 118.0,
                "Blood Sugar": 92.0,
                "Cholesterol": 175.0,
                "Creatinine": 0.88,
                "Hemoglobin": 15.1,
                "Vitamin D": 38.0,
            },
        ]
    )

    col_up, col_tpl = st.columns([3, 1])

    with col_tpl:
        st.download_button(
            label="📥 Download Batch CSV Template",
            data=sample_df.to_csv(index=False).encode("utf-8"),
            file_name="batch_patient_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Batch CSV File",
            type=["csv"],
            help="File must contain features: Age, Gender, DiagnosisID, Blood Pressure, Blood Sugar, Cholesterol, Creatinine, Hemoglobin, Vitamin D",
        )

    if uploaded_file is not None:
        try:
            batch_input_df = pd.read_csv(uploaded_file)
            required_cols = [
                "Age",
                "Gender",
                "DiagnosisID",
                "Blood Pressure",
                "Blood Sugar",
                "Cholesterol",
                "Creatinine",
                "Hemoglobin",
                "Vitamin D",
            ]

            missing = [c for c in required_cols if c not in batch_input_df.columns]
            if missing:
                st.error(f"Uploaded CSV is missing required columns: {missing}")
            else:
                st.success(f"Successfully ingested **{len(batch_input_df):,}** patient records for batch inference.")

                # Prepare input data
                model_input = batch_input_df[required_cols].copy()
                model_input["Gender"] = model_input["Gender"].astype(str)
                model_input["DiagnosisID"] = model_input["DiagnosisID"].astype(str)

                # Batch Inference
                batch_preds = model.predict(model_input)
                batch_probs = model.predict_proba(model_input)
                max_confidences = [round(max(p) * 100, 2) for p in batch_probs]

                # Enrich output dataframe
                enriched_df = batch_input_df.copy()
                enriched_df["Predicted_Outcome"] = batch_preds
                enriched_df["Confidence_%"] = max_confidences

                # Visual Summary
                st.markdown("### 📊 Batch Prediction Overview")
                b_c1, b_c2 = st.columns(2)

                with b_c1:
                    batch_counts = enriched_df["Predicted_Outcome"].value_counts().reset_index()
                    batch_counts.columns = ["Predicted_Outcome", "Patients"]
                    fig_batch = px.pie(
                        batch_counts,
                        names="Predicted_Outcome",
                        values="Patients",
                        hole=0.5,
                        color="Predicted_Outcome",
                        color_discrete_map={
                            "Recovered": "#10b981",
                            "Complicated": "#f59e0b",
                            "Deceased": "#ef4444",
                        },
                    )
                    fig_batch.update_layout(template=PLOTLY_TEMPLATE, height=350)
                    st.plotly_chart(fig_batch, use_container_width=True)

                with b_c2:
                    st.subheader("Triage Summary")
                    st.metric("Total Inpatients Processed", f"{len(enriched_df):,}")
                    rec_cnt = (enriched_df["Predicted_Outcome"] == "Recovered").sum()
                    comp_cnt = (enriched_df["Predicted_Outcome"] == "Complicated").sum()
                    dec_cnt = (enriched_df["Predicted_Outcome"] == "Deceased").sum()
                    st.write(f"• **Recovered (Routine):** {rec_cnt:,} ({rec_cnt/len(enriched_df)*100:.1f}%)")
                    st.write(f"• **Complicated (Monitoring):** {comp_cnt:,} ({comp_cnt/len(enriched_df)*100:.1f}%)")
                    st.write(f"• **Deceased (Critical Alert):** {dec_cnt:,} ({dec_cnt/len(enriched_df)*100:.1f}%)")

                st.markdown("### 📋 Enriched Batch Results Table")
                st.dataframe(enriched_df, use_container_width=True, hide_index=True)

                # Download Enriched CSV
                enriched_csv = enriched_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Full Batch Predictions (CSV)",
                    data=enriched_csv,
                    file_name="batch_predictions_enriched.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error processing uploaded CSV: {str(e)}")

st.divider()

# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div style="text-align:center; padding:15px 0 5px 0; color:var(--text-muted); font-size:12px;">
        Hospital Patient Analytics • Predictive Intelligence Engine • Ayush &amp; Moon
    </div>
    """
)