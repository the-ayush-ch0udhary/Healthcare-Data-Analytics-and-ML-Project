"""
Machine Learning Model Performance & Benchmark Suite
Detailed evaluation of the Random Forest champion model and multi-algorithm benchmarking.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

# ============================================================
# PAGE CONFIG & THEME
# ============================================================

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(APP_DIR))

from utils.theme import (
    apply_theme,
    get_model_comparison,
    get_model_metadata,
    load_dataset,
    plotly_template,
    render_sidebar,
)

st.set_page_config(
    page_title="Model Performance & Benchmarks",
    page_icon="📈",
    layout="wide",
)

apply_theme()

BASE_DIR = APP_DIR.parent
df = load_dataset()
meta = get_model_metadata()
benchmark_df = get_model_comparison()

render_sidebar(
    total_patients=len(df) if not df.empty else meta.get("dataset_total", 5000),
    model_accuracy=meta.get("accuracy_pct", "83.30%"),
    total_diagnoses=df["DiagnosisName"].nunique() if not df.empty else 10,
)

PLOTLY_TEMPLATE = plotly_template()

# ============================================================
# HEADER
# ============================================================

head1, head2 = st.columns([3, 1])

with head1:
    st.title("📈 Machine Learning Performance & Benchmarks")
    st.write(
        "Comprehensive validation analytics, per-class classification metrics, interactive confusion matrix, feature importances, and multi-model benchmark evaluation."
    )

with head2:
    st.success(
        f"""
        ### 🟢 Champion Model
        **{meta.get('model_name', 'Random Forest')}**
        
        **Test Accuracy:** **{meta.get('accuracy_pct', '83.30%')}**
        
        **F1 Macro:** **{meta.get('macro_avg', {}).get('f1_score', 0.7431):.4f}**
        """
    )

st.divider()

# ============================================================
# TABS
# ============================================================

perf_tab1, perf_tab2, perf_tab3 = st.tabs(
    [
        "🏆 Champion Model Evaluation",
        "📊 Multi-Algorithm Benchmark",
        "⚙️ Pipeline Architecture & Parameters",
    ]
)

# ============================================================
# TAB 1: CHAMPION MODEL EVALUATION
# ============================================================

with perf_tab1:
    st.markdown("### 🏆 Random Forest Classifier Performance")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric("Test Accuracy", meta.get("accuracy_pct", "83.30%"), delta="Overall Accuracy")

    with kpi2:
        macro_f1 = meta.get("macro_avg", {}).get("f1_score", 0.7431)
        st.metric("Macro F1-Score", f"{macro_f1:.4f}", delta="Unweighted Mean")

    with kpi3:
        weighted_f1 = meta.get("weighted_avg", {}).get("f1_score", 0.8317)
        st.metric("Weighted F1-Score", f"{weighted_f1:.4f}", delta="Support Weighted")

    with kpi4:
        st.metric("Test Inpatients", f"{meta.get('test_samples', 1000):,}", delta="20% Stratified Split")

    st.markdown("---")

    # Classification Metrics Table & Chart
    st.markdown("#### 🎯 Per-Class Precision, Recall & F1-Score")

    class_metrics_data = meta.get("class_metrics", {})
    if class_metrics_data:
        metrics_rows = []
        for outcome_name, scores in class_metrics_data.items():
            metrics_rows.append(
                {
                    "Outcome Class": outcome_name,
                    "Precision": scores.get("precision", 0.0),
                    "Recall": scores.get("recall", 0.0),
                    "F1 Score": scores.get("f1_score", 0.0),
                    "Test Support": scores.get("support", 0),
                }
            )
        metrics_df = pd.DataFrame(metrics_rows)

        col_tbl, col_chart = st.columns([1, 1])

        with col_tbl:
            formatted_metrics_df = metrics_df.copy()
            for c in ["Precision", "Recall", "F1 Score"]:
                formatted_metrics_df[c] = (formatted_metrics_df[c] * 100).round(2).astype(str) + "%"

            st.dataframe(formatted_metrics_df, use_container_width=True, hide_index=True)

            st.caption(
                "• **Recovered Class:** Highest precision (91.8%) and recall (92.9%) due to healthy biomarker baselines.\n"
                "• **Complicated Class:** Balanced F1 score (67.2%) identifying transitioning clinical severity.\n"
                "• **Deceased Class:** High precision (69.9%) identifying acute renal & cardiovascular risk flags."
            )

        with col_chart:
            fig_metrics = px.bar(
                metrics_df,
                x="Outcome Class",
                y=["Precision", "Recall", "F1 Score"],
                barmode="group",
                text_auto=".2f",
                color_discrete_sequence=["#38bdf8", "#818cf8", "#c084fc"],
            )
            fig_metrics.update_layout(
                template=PLOTLY_TEMPLATE,
                height=340,
                yaxis_title="Score (0 to 1.0)",
                xaxis_title="",
                yaxis_range=[0, 1.05],
                legend_title="Metric",
            )
            st.plotly_chart(fig_metrics, use_container_width=True)

    st.markdown("---")

    # Confusion Matrix (Interactive Heatmap)
    st.markdown("#### 🎯 Interactive Confusion Matrix")

    cm_data = meta.get("confusion_matrix", {})
    if cm_data and "matrix" in cm_data:
        cm_matrix = np.array(cm_data["matrix"])
        labels = cm_data["labels"]

        fig_cm = px.imshow(
            cm_matrix,
            labels=dict(x="Predicted Outcome", y="Actual Outcome", color="Patients"),
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(
            template=PLOTLY_TEMPLATE,
            height=430,
            xaxis_title="Predicted Outcome",
            yaxis_title="Actual Outcome",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    # Feature Importance Explorer
    st.markdown("#### 🔍 Feature Importance Explorer")

    feat_path = BASE_DIR / "models" / "feature_importance.csv"
    if feat_path.exists():
        feat_df = pd.read_csv(feat_path)
        # Clean feature names
        feat_df["Clean_Feature"] = (
            feat_df["Feature"]
            .str.replace("numerical__", "", regex=False)
            .str.replace("categorical__Gender_", "Gender: ", regex=False)
            .str.replace("categorical__DiagnosisID_", "Diagnosis ID: ", regex=False)
        )

        top_n = st.slider("Select Top N Features to Display:", min_value=5, max_value=len(feat_df), value=10)
        display_feat = feat_df.head(top_n).sort_values(by="Importance")

        fig_feat = px.bar(
            display_feat,
            x="Importance",
            y="Clean_Feature",
            orientation="h",
            text="Importance",
            color="Importance",
            color_continuous_scale="Viridis",
        )
        fig_feat.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_feat.update_layout(
            template=PLOTLY_TEMPLATE,
            height=440,
            xaxis_title="Gini Importance Score",
            yaxis_title="",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_feat, use_container_width=True)

        st.caption(
            "💡 **Key Finding:** Patient **Age (32.1%)**, **Blood Pressure (12.6%)**, **Hemoglobin (11.7%)**, and **Blood Sugar (10.7%)** are the most influential clinical drivers of outcome severity."
        )


# ============================================================
# TAB 2: MULTI-ALGORITHM BENCHMARK
# ============================================================

with perf_tab2:
    st.markdown("### 📊 Multi-Algorithm Benchmark Comparison")
    st.write(
        "Evaluation of four distinct machine learning classifiers trained and tested on identical stratified 80:20 inpatient splits."
    )

    if benchmark_df is not None and not benchmark_df.empty:
        st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        bench_c1, bench_c2 = st.columns(2)

        with bench_c1:
            st.subheader("Model Accuracy Comparison")
            fig_acc = px.bar(
                benchmark_df,
                x="Model",
                y="Accuracy",
                color="Model",
                text="Accuracy",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_acc.update_traces(texttemplate="%{text:.2%}", textposition="outside")
            fig_acc.update_layout(
                template=PLOTLY_TEMPLATE,
                height=380,
                showlegend=False,
                yaxis_range=[0.65, 0.95],
                yaxis_title="Accuracy",
                xaxis_title="",
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with bench_c2:
            st.subheader("Macro F1-Score vs. Training Time")
            fig_f1 = px.bar(
                benchmark_df,
                x="Model",
                y="F1_Macro",
                color="Training_Time_Sec",
                text="F1_Macro",
                color_continuous_scale="Teal",
            )
            fig_f1.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_f1.update_layout(
                template=PLOTLY_TEMPLATE,
                height=380,
                yaxis_title="Macro F1-Score",
                xaxis_title="",
                coloraxis_colorbar=dict(title="Time (s)"),
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        st.info(
            """
            **Benchmark Summary:**
            - **Random Forest:** Selected as the production champion model for superior nonlinear robustness, handling mixed lab interactions, and resilient feature scoring.
            - **Gradient Boosting:** Excellent competitive accuracy (82.90%) with tight decision boundaries.
            - **Logistic Regression:** Strong linear baseline performance (84.60%) with near-instant training time.
            - **Decision Tree:** Interpretable baseline with 80.00% accuracy.
            """
        )
    else:
        st.info("Benchmark data not yet generated. Run `src/train_model3.py` to populate benchmarks.")


# ============================================================
# TAB 3: PIPELINE ARCHITECTURE & PARAMETERS
# ============================================================

with perf_tab3:
    st.markdown("### ⚙️ Production Machine Learning Pipeline Architecture")

    p1, p2 = st.columns(2)

    with p1:
        st.info(
            """
            ### 🛠 Data Preprocessing Pipeline
            - **Numerical Transformer:** `StandardScaler()` applied to 7 clinical features (*Age, Blood Pressure, Blood Sugar, Cholesterol, Creatinine, Hemoglobin, Vitamin D*).
            - **Categorical Transformer:** `OneHotEncoder(handle_unknown='ignore')` applied to *Gender* and *DiagnosisID*.
            - **Feature Assembly:** `ColumnTransformer()` maintaining strictly isolated feature paths without data leakage.
            """
        )

    with p2:
        st.success(
            """
            ### 🌲 Champion Model Configuration
            - **Classifier:** `RandomForestClassifier()`
            - **Estimators (`n_estimators`):** 300 Trees
            - **Parallelism (`n_jobs`):** -1 (Full multi-core concurrency)
            - **Train/Test Split:** 80% Train (4,000 samples) : 20% Test (1,000 samples)
            - **Stratification:** Maintained across 3 target classes
            """
        )

    st.markdown("---")

    st.subheader("Raw Classification Report Output")
    report_file = BASE_DIR / "models" / "classification_report.txt"
    if report_file.exists():
        with open(report_file, "r") as f:
            st.code(f.read(), language="text")

st.divider()

# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div style="text-align:center; padding:15px 0 5px 0; color:var(--text-muted); font-size:12px;">
        Hospital Patient Analytics • Model Evaluation &amp; Benchmarking • Ayush &amp; Moon
    </div>
    """
)