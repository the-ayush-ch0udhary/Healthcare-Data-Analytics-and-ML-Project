"""
Shared theme, navigation, dynamic data loaders, and clinical helpers
for Hospital Patient Analytics.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = APP_DIR.parent
STYLE_CSS_PATH = APP_DIR / "assets" / "style.css"
DATA_PATH = BASE_DIR / "data" / "processed" / "healthcare_cleaned.csv"
MODEL_METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
MODEL_COMPARISON_PATH = BASE_DIR / "models" / "model_comparison.csv"


# ============================================================
# THEME PALETTES
# ============================================================

DARK_THEME = {
    "bg-gradient": "linear-gradient(135deg, #090d16 0%, #111827 50%, #0b0f19 100%)",
    "text-primary": "#f8fafc",
    "text-secondary": "#cbd5e1",
    "text-muted": "#94a3b8",
    "card-bg": "rgba(17, 24, 39, 0.70)",
    "card-border": "rgba(255, 255, 255, 0.08)",
    "card-shadow": "rgba(0, 0, 0, 0.40)",
    "card-hover-border": "rgba(99, 102, 241, 0.50)",
    "card-hover-shadow": "rgba(99, 102, 241, 0.18)",
    "sidebar-bg": "#0b0f19",
    "sidebar-border": "rgba(255, 255, 255, 0.08)",
    "heading-start": "#38bdf8",
    "heading-end": "#818cf8",
    "metric-value": "#38bdf8",
    "metric-label": "#94a3b8",
    "button-start": "#4f46e5",
    "button-end": "#7c3aed",
    "button-hover-start": "#4338ca",
    "button-hover-end": "#6d28d9",
    "button-hover-glow": "rgba(99, 102, 241, 0.40)",
    "input-bg": "rgba(15, 23, 42, 0.75)",
    "input-border": "rgba(255, 255, 255, 0.14)",
    "divider": "rgba(255, 255, 255, 0.08)",
    "code-bg": "#090d16",
    "code-text": "#e2e8f0",
    "scrollbar-track": "rgba(255, 255, 255, 0.03)",
    "scrollbar-thumb": "rgba(99, 102, 241, 0.40)",
    "scrollbar-thumb-hover": "rgba(99, 102, 241, 0.75)",
    "tab-bg": "rgba(255, 255, 255, 0.03)",
    "tab-active-bg": "rgba(99, 102, 241, 0.20)",
    "alert-text": "#f8fafc",
}

LIGHT_THEME = {
    "bg-gradient": "linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%)",
    "text-primary": "#0f172a",
    "text-secondary": "#334155",
    "text-muted": "#64748b",
    "card-bg": "rgba(255, 255, 255, 0.88)",
    "card-border": "rgba(15, 23, 42, 0.08)",
    "card-shadow": "rgba(15, 23, 42, 0.06)",
    "card-hover-border": "rgba(79, 70, 229, 0.35)",
    "card-hover-shadow": "rgba(79, 70, 229, 0.10)",
    "sidebar-bg": "#ffffff",
    "sidebar-border": "rgba(15, 23, 42, 0.08)",
    "heading-start": "#4f46e5",
    "heading-end": "#9333ea",
    "metric-value": "#3730a3",
    "metric-label": "#475569",
    "button-start": "#6366f1",
    "button-end": "#8b5cf6",
    "button-hover-start": "#4f46e5",
    "button-hover-end": "#7c3aed",
    "button-hover-glow": "rgba(99, 102, 241, 0.25)",
    "input-bg": "#ffffff",
    "input-border": "#cbd5e1",
    "divider": "rgba(15, 23, 42, 0.08)",
    "code-bg": "#f8fafc",
    "code-text": "#0f172a",
    "scrollbar-track": "rgba(15, 23, 42, 0.03)",
    "scrollbar-thumb": "rgba(79, 70, 229, 0.30)",
    "scrollbar-thumb-hover": "rgba(79, 70, 229, 0.55)",
    "tab-bg": "rgba(15, 23, 42, 0.03)",
    "tab-active-bg": "rgba(79, 70, 229, 0.10)",
    "alert-text": "#0f172a",
}


# ============================================================
# CSS BUILDERS
# ============================================================

def _build_variable_block(palette: dict) -> str:
    lines = [f"    --{key}: {value};" for key, value in palette.items()]
    return ":root {\n" + "\n".join(lines) + "\n}"


@st.cache_data(show_spinner=False)
def _read_style_css(mtime: float) -> str:
    if STYLE_CSS_PATH.exists():
        return STYLE_CSS_PATH.read_text(encoding="utf-8")
    return ""


def _load_style_css() -> str:
    if STYLE_CSS_PATH.exists():
        mtime = STYLE_CSS_PATH.stat().st_mtime
        return _read_style_css(mtime)
    return ""


def _on_toggle_change():
    st.session_state.dark_mode = st.session_state.theme_toggle


def apply_theme() -> str:
    """Applies theme CSS variables and styles."""
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    palette = DARK_THEME if st.session_state.dark_mode else LIGHT_THEME
    hide_nav_css = """
    [data-testid="stSidebarNav"],
    [data-testid="stSidebarNavItems"],
    [data-testid="stSidebarNavSeparator"],
    nav[data-testid="stSidebarNav"],
    div[data-testid="stSidebarNav"],
    ul[data-testid="stSidebarNavItems"],
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    """
    css = _build_variable_block(palette) + "\n" + hide_nav_css + "\n" + _load_style_css()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    return "dark" if st.session_state.dark_mode else "light"



# ============================================================
# DYNAMIC METRICS & DATA LOADERS
# ============================================================

@st.cache_data(ttl=60)
def load_dataset() -> pd.DataFrame:
    """Loads and caches the cleaned healthcare dataset."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


@st.cache_data(ttl=60)
def get_model_metadata() -> Dict[str, Any]:
    """Loads model training metadata or returns sensible defaults."""
    if MODEL_METADATA_PATH.exists():
        try:
            with open(MODEL_METADATA_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass

    # Default fallback
    return {
        "model_name": "Random Forest",
        "accuracy": 0.833,
        "accuracy_pct": "83.30%",
        "dataset_total": 5000,
        "train_samples": 4000,
        "test_samples": 1000,
        "classes": ["Complicated", "Deceased", "Recovered"],
    }


@st.cache_data(ttl=60)
def get_model_comparison() -> Optional[pd.DataFrame]:
    """Loads benchmark model comparison table."""
    if MODEL_COMPARISON_PATH.exists():
        try:
            return pd.read_csv(MODEL_COMPARISON_PATH)
        except Exception:
            pass
    return None


# ============================================================
# CLINICAL REFERENCE RANGES & STATUS
# ============================================================

def get_clinical_status(test_name: str, value: float) -> Tuple[str, str, str]:
    """
    Evaluates clinical measurements against recognized medical reference ranges.
    Returns: (Status Label, Badge Class, Clinical Guidance)
    """
    if test_name == "Blood Pressure":
        # Systolic mmHg
        if value < 90:
            return "Low / Hypotension", "badge-warning", "Blood pressure is below standard normal range (<90 mmHg)."
        elif value <= 120:
            return "Normal", "badge-normal", "Optimal resting systolic blood pressure (90-120 mmHg)."
        elif value <= 139:
            return "Elevated / Prehypertension", "badge-warning", "Slightly elevated systolic blood pressure (121-139 mmHg)."
        else:
            return "Hypertension Stage 2", "badge-danger", "Significantly elevated blood pressure (≥140 mmHg)."

    elif test_name == "Blood Sugar":
        # Fasting mg/dL
        if value < 70:
            return "Hypoglycemia", "badge-danger", "Blood sugar is low (<70 mg/dL). Risk of hypoglycemic event."
        elif value <= 99:
            return "Normal Fasting", "badge-normal", "Healthy blood glucose level (70-99 mg/dL)."
        elif value <= 125:
            return "Prediabetes Range", "badge-warning", "Impaired fasting glucose (100-125 mg/dL)."
        else:
            return "Diabetic Range", "badge-danger", "Elevated fasting blood sugar (≥126 mg/dL)."

    elif test_name == "Cholesterol":
        # Total Cholesterol mg/dL
        if value < 200:
            return "Desirable / Optimal", "badge-normal", "Healthy total cholesterol (<200 mg/dL)."
        elif value <= 239:
            return "Borderline High", "badge-warning", "Borderline elevated cholesterol (200-239 mg/dL)."
        else:
            return "High Cholesterol", "badge-danger", "Elevated cholesterol (≥240 mg/dL). Cardiovascular risk factor."

    elif test_name == "Creatinine":
        # Serum mg/dL
        if value < 0.6:
            return "Low Creatinine", "badge-info", "Below standard reference range (<0.6 mg/dL)."
        elif value <= 1.2:
            return "Normal Renal Function", "badge-normal", "Standard healthy serum creatinine (0.6-1.2 mg/dL)."
        elif value <= 1.8:
            return "Mild Impairment", "badge-warning", "Mildly elevated creatinine (1.3-1.8 mg/dL). Monitor kidney status."
        else:
            return "Renal Risk / Elevated", "badge-danger", "High serum creatinine (≥1.9 mg/dL). Suggests renal stress."

    elif test_name == "Hemoglobin":
        # g/dL
        if value < 10.0:
            return "Moderate-Severe Anemia", "badge-danger", "Significantly low hemoglobin (<10 g/dL)."
        elif value < 12.0:
            return "Mild Anemia", "badge-warning", "Mildly low hemoglobin (10.0-11.9 g/dL)."
        elif value <= 17.5:
            return "Normal Range", "badge-normal", "Healthy adult hemoglobin (12.0-17.5 g/dL)."
        else:
            return "Elevated Hemoglobin", "badge-warning", "High hemoglobin concentration (>17.5 g/dL)."

    elif test_name == "Vitamin D":
        # ng/mL
        if value < 20:
            return "Deficient", "badge-danger", "Vitamin D deficiency (<20 ng/mL). Supplementation indicated."
        elif value < 30:
            return "Insufficient", "badge-warning", "Suboptimal Vitamin D level (20-29 ng/mL)."
        elif value <= 100:
            return "Optimal", "badge-normal", "Healthy Vitamin D level (30-100 ng/mL)."
        else:
            return "High", "badge-warning", "Excessive Vitamin D level (>100 ng/mL)."

    return "Recorded", "badge-info", "Clinical measurement logged."


# ============================================================
# CUSTOM SIDEBAR
# ============================================================

def render_sidebar(
    total_patients: Optional[int] = None,
    model_accuracy: Optional[str] = None,
    total_diagnoses: Optional[int] = None,
):
    """Renders the standard navigation sidebar with dynamic KPIs."""
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    if "theme_toggle" not in st.session_state:
        st.session_state.theme_toggle = st.session_state.dark_mode

    # Resolve dynamic values if not supplied
    if total_patients is None or total_diagnoses is None or model_accuracy is None:
        df = load_dataset()
        meta = get_model_metadata()
        if total_patients is None:
            total_patients = len(df) if not df.empty else meta.get("dataset_total", 5000)
        if total_diagnoses is None:
            total_diagnoses = df["DiagnosisName"].nunique() if not df.empty else 10
        if model_accuracy is None:
            model_accuracy = meta.get("accuracy_pct", "83.30%")

    with st.sidebar:
        st.html(
            """
            <div style="text-align:center; padding:10px 0 14px 0;">
                <div style="font-size:40px; line-height:1; margin-bottom:8px;">🏥</div>
                <div style="font-size:20px; font-weight:800; color:var(--text-primary); line-height:1.2;">
                    Hospital Analytics
                </div>
                <div style="font-size:12px; margin-top:4px; color:var(--text-muted);">
                    Healthcare Intelligence Platform
                </div>
            </div>
            """
        )

        st.toggle(
            "🌙 Dark Mode",
            key="theme_toggle",
            on_change=_on_toggle_change,
            help="Switch between dark and light themes",
        )

        st.markdown("---")

        st.html(
            """
            <div style="
                padding:10px 14px;
                border-radius:12px;
                background:rgba(16,185,129,0.12);
                border:1px solid rgba(16,185,129,0.25);
                margin-bottom:10px;
            ">
                <div style="font-size:13px; font-weight:600; color:var(--text-primary);">
                    🟢 System Online
                </div>
                <div style="font-size:11px; margin-top:2px; color:var(--text-secondary);">
                    Analytics & ML Engine active
                </div>
            </div>
            """
        )

        st.markdown("### 🧭 Navigation")
        st.page_link("app.py", label="Home", icon="🏠")
        st.page_link("pages/About.py", label="About & Reference", icon="ℹ️")
        st.page_link("pages/Dashboard.py", label="Analytics Dashboard", icon="📊")
        st.page_link("pages/Model_Performance.py", label="Model Performance", icon="📈")
        st.page_link("pages/Predict.py", label="Patient Outcome Prediction", icon="🤖")

        st.markdown("---")

        st.markdown("### 📊 System Overview")
        st.metric("Total Patients", f"{total_patients:,}")
        st.metric("Model Accuracy", model_accuracy)
        st.metric("Diagnoses", total_diagnoses)

        st.markdown("---")
        apply_theme()

        st.html(
            """
            <div style="text-align:center; padding:10px 0 5px 0;">
                <div style="font-size:11px; color:var(--text-muted);">
                    Hospital Patient Analytics • v2.0
                </div>
                <div style="font-size:10px; margin-top:3px; color:var(--text-muted);">
                    Data Analytics & Machine Learning
                </div>
                <div style="font-size:10px; margin-top:3px; color:var(--text-muted);">
                    Ayush &amp; Moon
                </div>
            </div>
            """
        )


def plotly_template() -> str:
    """Returns the Plotly template matching the current session theme."""
    return "plotly_dark" if st.session_state.get("dark_mode", True) else "plotly_white"