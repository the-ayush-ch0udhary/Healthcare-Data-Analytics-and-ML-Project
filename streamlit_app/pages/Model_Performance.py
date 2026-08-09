import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
from PIL import Image

import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
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

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

REPORT_PATH = (
    BASE_DIR
    / "models"
    / "classification_report.txt"
)

CONFUSION_MATRIX_PATH = (
    BASE_DIR
    / "models"
    / "confusion_matrix.png"
)

FEATURE_IMPORTANCE_PATH = (
    BASE_DIR
    / "models"
    / "feature_importance.csv"
)

FEATURE_IMPORTANCE_IMAGE = (
    BASE_DIR
    / "models"
    / "feature_importance.png"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_NAME = "Random Forest"
MODEL_ACCURACY = 0.833
MODEL_ACCURACY_PERCENT = "83.3%"
DATASET_SIZE = 5000
TEST_SIZE = 1000
OUTCOME_CLASSES = 3


# ============================================================
# HEADER
# ============================================================

left, right = st.columns(
    [3, 1]
)


with left:

    st.title(
        "📈 Machine Learning Model Performance"
    )

    st.subheader(
        "Model Evaluation & Performance Analytics"
    )

    st.write(
        """
        Evaluate the trained Random Forest model using
        performance metrics, confusion matrix,
        classification report and feature importance.
        """
    )


with right:

    st.success(
        """
        ### 🟢 Best Model

        **Random Forest**

        **Accuracy**

        **83.3%**

        **Status**

        Ready
        """
    )


st.divider()


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

st.markdown(
    "## 📊 Executive Summary"
)


kpi1, kpi2, kpi3, kpi4 = st.columns(4)


with kpi1:

    st.metric(
        label="🏆 Final Model",
        value="Random Forest"
    )


with kpi2:

    st.metric(
        label="🎯 Test Accuracy",
        value="83.3%"
    )


with kpi3:

    st.metric(
        label="📊 Test Samples",
        value=f"{TEST_SIZE:,}"
    )


with kpi4:

    st.metric(
        label="🏥 Outcome Classes",
        value=OUTCOME_CLASSES
    )


st.divider()


# ============================================================
# MODEL SUMMARY
# ============================================================

st.markdown(
    "## 🏆 Final Model"
)


model_col1, model_col2 = st.columns(2)


with model_col1:

    st.info(
        """
        ### 🌲 Random Forest Classifier

        The final prediction system uses a Random Forest
        classification model to predict hospital patient
        outcomes.

        **Target classes:**

        • Recovered

        • Complicated

        • Deceased
        """
    )


with model_col2:

    st.success(
        """
        ### 📊 Evaluation

        **Accuracy:** 83.3%

        **Training samples:** 4,000

        **Testing samples:** 1,000

        **Train/Test Split:** 80 : 20

        **Evaluation:** Stratified test set
        """
    )


st.divider()


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

st.markdown(
    "## 🎯 Classification Performance"
)


metrics_df = pd.DataFrame(
    {
        "Outcome": [
            "Recovered",
            "Complicated",
            "Deceased"
        ],
        "Precision": [
            0.92,
            0.66,
            0.70
        ],
        "Recall": [
            0.93,
            0.68,
            0.58
        ],
        "F1 Score": [
            0.92,
            0.67,
            0.63
        ]
    }
)


display_metrics = metrics_df.copy()

display_metrics[
    [
        "Precision",
        "Recall",
        "F1 Score"
    ]
] = (
    display_metrics[
        [
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ] * 100
).round(1)


display_metrics[
    [
        "Precision",
        "Recall",
        "F1 Score"
    ]
] = display_metrics[
    [
        "Precision",
        "Recall",
        "F1 Score"
    ]
].astype(str) + "%"


st.dataframe(
    display_metrics,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# METRIC CHART
# ============================================================

fig = px.bar(
    metrics_df,
    x="Outcome",
    y=[
        "Precision",
        "Recall",
        "F1 Score"
    ],
    barmode="group",
    text_auto=".2f"
)

fig.update_layout(
    template=PLOTLY_TEMPLATE,
    height=450,
    yaxis_title="Score",
    xaxis_title="Outcome",
    yaxis_range=[0, 1]
)

st.plotly_chart(
    fig,
    use_container_width=True
)


st.divider()


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.markdown(
    "## 🎯 Confusion Matrix"
)


try:

    image = Image.open(
        CONFUSION_MATRIX_PATH
    )

    st.image(
        image,
        use_container_width=True
    )

except Exception:

    st.warning(
        "Confusion matrix image could not be loaded."
    )


st.caption(
    """
    The confusion matrix shows how the Random Forest model
    classified the patients across the three outcome classes.
    """
)


st.divider()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.markdown(
    "## 📄 Classification Report"
)


try:

    with open(
        REPORT_PATH,
        "r"
    ) as file:

        report = file.read()

    st.code(
        report,
        language="text"
    )

except Exception:

    st.warning(
        "Classification report could not be loaded."
    )


st.divider()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown(
    "## 🔍 Feature Importance"
)


try:

    feature_df = pd.read_csv(
        FEATURE_IMPORTANCE_PATH
    )

    feature_df["Feature"] = (
        feature_df["Feature"]
        .str.replace(
            "numerical__",
            "",
            regex=False
        )
        .str.replace(
            "categorical__",
            "",
            regex=False
        )
        .str.replace(
            "Gender_",
            "Gender: ",
            regex=False
        )
        .str.replace(
            "DiagnosisID_",
            "Diagnosis ID: ",
            regex=False
        )
    )

    feature_df = (
        feature_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(10)
    )


    fig = px.bar(
        feature_df.sort_values(
            "Importance"
        ),
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance"
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=500,
        xaxis_title="Importance",
        yaxis_title="Feature"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


except Exception:

    st.warning(
        "Feature importance data could not be loaded."
    )


st.divider()


# ============================================================
# MODEL INSIGHTS
# ============================================================

st.markdown(
    "## 💡 Model Insights"
)


left, right = st.columns(2)


with left:

    st.success(
        """
        ### ✅ Strengths

        • 83.3% test accuracy

        • Strong Recovered classification

        • 92% F1 score for Recovered

        • 67% F1 score for Complicated

        • 63% F1 score for Deceased

        • Handles nonlinear relationships

        • Complete preprocessing pipeline
        """
    )


with right:

    st.warning(
        """
        ### ⚠️ Limitations

        • Dataset is synthetic

        • Performance may differ on real hospital data

        • Deceased recall is lower than the other classes

        • External validation is required

        • Model predictions should not replace
          professional clinical judgment
        """
    )


st.divider()


# ============================================================
# TRAINING PIPELINE
# ============================================================

st.markdown(
    "## ⚙️ Training Pipeline"
)


pipeline_col1, pipeline_col2 = st.columns(2)


with pipeline_col1:

    st.info(
        """
        ### Data Preparation

        ✔ Data cleaning

        ✔ Missing-value checking

        ✔ Feature selection

        ✔ Categorical encoding

        ✔ Numerical scaling

        ✔ Stratified train/test split
        """
    )


with pipeline_col2:

    st.success(
        """
        ### Model Training

        ✔ Random Forest Classifier

        ✔ 300 estimators

        ✔ Complete preprocessing pipeline

        ✔ Model evaluation

        ✔ Confusion matrix

        ✔ Feature importance analysis
        """
    )


st.divider()


# ============================================================
# PREDICTION FEATURES
# ============================================================

st.markdown(
    "## 🧬 Prediction Features"
)


prediction_features = [
    "Age",
    "Gender",
    "Diagnosis ID",
    "Blood Pressure",
    "Blood Sugar",
    "Cholesterol",
    "Creatinine",
    "Hemoglobin",
    "Vitamin D"
]


feature_columns = st.columns(3)


for index, feature in enumerate(
    prediction_features
):

    with feature_columns[
        index % 3
    ]:

        st.markdown(
            f"""
            <div style="
                padding:12px;
                margin-bottom:10px;
                border:1px solid var(--card-border);
                border-radius:10px;
                text-align:center;
            ">
                <strong>{feature}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )


st.divider()


# ============================================================
# FUTURE IMPROVEMENTS
# ============================================================

st.markdown(
    "## 🚀 Future Improvements"
)


col1, col2, col3 = st.columns(3)


with col1:

    with st.container(
        border=True
    ):

        st.markdown(
            "### 📊 Data"
        )

        st.write(
            "• Larger dataset"
        )

        st.write(
            "• Real-world healthcare data"
        )

        st.write(
            "• Additional clinical features"
        )

        st.write(
            "• External validation"
        )


with col2:

    with st.container(
        border=True
    ):

        st.markdown(
            "### 🤖 Machine Learning"
        )

        st.write(
            "• XGBoost comparison"
        )

        st.write(
            "• Ensemble methods"
        )

        st.write(
            "• Hyperparameter optimization"
        )

        st.write(
            "• Explainable AI"
        )


with col3:

    with st.container(
        border=True
    ):

        st.markdown(
            "### ☁ Deployment"
        )

        st.write(
            "• REST API"
        )

        st.write(
            "• Docker"
        )

        st.write(
            "• Cloud deployment"
        )

        st.write(
            "• Real-time prediction"
        )


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

st.markdown(
    """
    <div style="
        text-align:center;
        padding:25px 10px 10px 10px;
        margin-top:20px;
    ">

        <div style="
            font-size:20px;
            font-weight:700;
            margin-bottom:8px;
        ">
            📈 Machine Learning Model Performance
        </div>

        <div style="
            font-size:14px;
            margin-bottom:6px;
            opacity:0.85;
        ">
            Hospital Patient Analytics & Outcome Prediction
        </div>

        <div style="
            font-size:13px;
            margin-bottom:6px;
            opacity:0.7;
        ">
            Developed using
            <strong>
                Python • Streamlit • Scikit-Learn • Plotly
            </strong>
        </div>

        <div style="
            font-size:14px;
            font-weight:600;
            margin-top:10px;
        ">
            Developed by Ayush & Moon
        </div>

        <div style="
            font-size:12px;
            margin-top:8px;
            opacity:0.5;
        ">
            Version 1.0
        </div>

    </div>
    """,
    unsafe_allow_html=True
)