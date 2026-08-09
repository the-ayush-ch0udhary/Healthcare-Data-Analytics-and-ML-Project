"""
Shared theme and sidebar engine for Hospital Patient Analytics.

Features:
- Instant Dark / Light mode toggle
- Persistent theme across pages
- Custom sidebar with top toggle placement
- Custom navigation
- System overview
- st.html based sidebar branding/footer
- Shared assets/style.css
- Plotly theme support
"""

from pathlib import Path
import streamlit as st


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent.parent
STYLE_CSS_PATH = APP_DIR / "assets" / "style.css"


# ============================================================
# DARK THEME
# ============================================================

DARK_THEME = {
    "bg-gradient": "linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)",

    "text-primary": "#f8fafc",
    "text-secondary": "#cbd5e1",
    "text-muted": "#94a3b8",

    "card-bg": "rgba(255, 255, 255, 0.045)",
    "card-border": "rgba(255, 255, 255, 0.10)",
    "card-shadow": "rgba(0, 0, 0, 0.35)",
    "card-hover-border": "rgba(99, 102, 241, 0.50)",
    "card-hover-shadow": "rgba(99, 102, 241, 0.20)",

    "sidebar-bg": "#0f172a",
    "sidebar-border": "rgba(255, 255, 255, 0.10)",

    "heading-start": "#38bdf8",
    "heading-end": "#818cf8",

    "metric-value": "#38bdf8",
    "metric-label": "#cbd5e1",

    "button-start": "#6366f1",
    "button-end": "#a855f7",
    "button-hover-start": "#4f46e5",
    "button-hover-end": "#9333ea",
    "button-hover-glow": "rgba(129, 140, 248, 0.40)",

    "input-bg": "#1e293b",
    "input-border": "rgba(255, 255, 255, 0.20)",

    "divider": "rgba(255, 255, 255, 0.10)",

    "code-bg": "rgba(15, 23, 42, 0.80)",
    "code-text": "#e2e8f0",

    "scrollbar-track": "rgba(255, 255, 255, 0.03)",
    "scrollbar-thumb": "rgba(99, 102, 241, 0.45)",
    "scrollbar-thumb-hover": "rgba(99, 102, 241, 0.75)",

    "tab-bg": "rgba(255, 255, 255, 0.04)",
    "tab-active-bg": "rgba(99, 102, 241, 0.25)",

    "alert-text": "#f8fafc",
    "alert-success-bg": "rgba(16, 185, 129, 0.18)",
    "alert-info-bg": "rgba(59, 130, 246, 0.18)",
    "alert-warning-bg": "rgba(245, 158, 11, 0.18)",
}


# ============================================================
# LIGHT THEME
# ============================================================

LIGHT_THEME = {
    "bg-gradient": "linear-gradient(135deg, #f8fafc 0%, #eef2ff 50%, #f8fafc 100%)",

    "text-primary": "#0f172a",
    "text-secondary": "#1e293b",
    "text-muted": "#475569",

    "card-bg": "rgba(255, 255, 255, 0.92)",
    "card-border": "rgba(15, 23, 42, 0.12)",
    "card-shadow": "rgba(15, 23, 42, 0.07)",
    "card-hover-border": "rgba(79, 70, 229, 0.40)",
    "card-hover-shadow": "rgba(79, 70, 229, 0.12)",

    "sidebar-bg": "#ffffff",
    "sidebar-border": "rgba(15, 23, 42, 0.10)",

    "heading-start": "#4f46e5",
    "heading-end": "#9333ea",

    "metric-value": "#3730a3",
    "metric-label": "#334155",

    "button-start": "#6366f1",
    "button-end": "#a855f7",
    "button-hover-start": "#4f46e5",
    "button-hover-end": "#9333ea",
    "button-hover-glow": "rgba(99, 102, 241, 0.30)",

    "input-bg": "#ffffff",
    "input-border": "#cbd5e1",

    "divider": "rgba(15, 23, 42, 0.12)",

    "code-bg": "#f1f5f9",
    "code-text": "#0f172a",

    "scrollbar-track": "rgba(15, 23, 42, 0.04)",
    "scrollbar-thumb": "rgba(79, 70, 229, 0.35)",
    "scrollbar-thumb-hover": "rgba(79, 70, 229, 0.60)",

    "tab-bg": "rgba(15, 23, 42, 0.04)",
    "tab-active-bg": "rgba(79, 70, 229, 0.12)",

    "alert-text": "#0f172a",
    "alert-success-bg": "#e8f7ee",
    "alert-info-bg": "#eaf2ff",
    "alert-warning-bg": "#fff8df",
}


# ============================================================
# CSS VARIABLE BUILDER
# ============================================================

def _build_variable_block(palette: dict) -> str:
    lines = [f"    --{key}: {value};" for key, value in palette.items()]
    return ":root {\n" + "\n".join(lines) + "\n}"


# ============================================================
# STREAMLIT CSS OVERRIDES
# ============================================================

def _build_streamlit_overrides() -> str:
    return """

    /* ========================================================
       GLOBAL APP
       ======================================================== */

    .stApp {
        background: var(--bg-gradient) !important;
        color: var(--text-primary) !important;
        transition: background 0.3s ease, color 0.3s ease !important;
    }


    /* ========================================================
       HIDE DEFAULT STREAMLIT NAVIGATION
       ======================================================== */

    [data-testid="stSidebarNav"] {
        display: none !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--sidebar-border) !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--text-primary);
    }

    [data-testid="stSidebar"] a {
        color: var(--text-primary) !important;
        text-decoration: none !important;
    }

    [data-testid="stSidebar"] a:hover {
        color: var(--heading-start) !important;
    }


    /* ========================================================
       TEXT & LABELS
       ======================================================== */

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    label,
    label p,
    label span {
        color: var(--text-primary) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        color: var(--metric-value) !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] span {
        color: var(--metric-label) !important;
    }


    /* ========================================================
       INPUTS & FORM CONTROLS (NUMBER INPUT FIX)
       ======================================================== */

    div[data-baseweb="input"],
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"],
    div[data-baseweb="select"] > div {
        background-color: var(--input-bg) !important;
        border-color: var(--input-border) !important;
        color: var(--text-primary) !important;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="base-input"] input,
    div[data-baseweb="select"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        color: var(--text-primary) !important;
        -webkit-text-fill-color: var(--text-primary) !important;
        background-color: transparent !important;
        font-weight: 500 !important;
    }

    div[data-baseweb="select"] span {
        color: var(--text-primary) !important;
    }

    /* Number Input Buttons & Steppers */
    [data-testid="stNumberInput"] button {
        background-color: var(--input-bg) !important;
        border-color: var(--input-border) !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stNumberInput"] button:hover {
        background-color: var(--tab-active-bg) !important;
    }

    [data-testid="stNumberInput"] svg {
        fill: var(--text-primary) !important;
        color: var(--text-primary) !important;
    }

    /* Selectbox Dropdown Menu Popover */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: var(--sidebar-bg) !important;
        border: 1px solid var(--input-border) !important;
    }

    li[role="option"],
    div[data-baseweb="option"] {
        background-color: var(--sidebar-bg) !important;
        color: var(--text-primary) !important;
    }

    li[role="option"]:hover,
    div[data-baseweb="option"]:hover {
        background-color: var(--tab-active-bg) !important;
    }


    /* ========================================================
       SLIDER
       ======================================================== */

    [data-testid="stSlider"] {
        color: var(--text-primary) !important;
    }

    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"] {
        color: var(--text-secondary) !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(
            135deg,
            var(--button-start),
            var(--button-end)
        ) !important;
        color: #ffffff !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }


    /* ========================================================
       TOGGLE & CHECKBOX
       ======================================================== */

    [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
    [data-testid="stSidebar"] [data-testid="stCheckbox"] p,
    [data-testid="stWidgetLabel"] p {
        color: var(--text-primary) !important;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    [data-testid="stAlert"] {
        color: var(--alert-text) !important;
        border-radius: 12px !important;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] strong,
    [data-testid="stAlert"] li {
        color: var(--alert-text) !important;
    }


    /* ========================================================
       CONTAINERS & CARDS
       ======================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card-bg) !important;
        border-color: var(--card-border) !important;
        border-radius: 16px !important;
    }


    /* ========================================================
       TABS
       ======================================================== */

    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        background: var(--tab-bg) !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--text-primary) !important;
        background: var(--tab-active-bg) !important;
    }


    /* ========================================================
       DATAFRAMES
       ======================================================== */

    [data-testid="stDataFrame"] {
        border: 1px solid var(--card-border) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
    }


    /* ========================================================
       CODE
       ======================================================== */

    [data-testid="stCodeBlock"] pre,
    [data-testid="stCodeBlock"] code {
        background: var(--code-bg) !important;
        color: var(--code-text) !important;
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border-color: var(--divider) !important;
    }


    /* ========================================================
       SCROLLBAR
       ======================================================== */

    * {
        scrollbar-width: thin;
        scrollbar-color: var(--scrollbar-thumb) var(--scrollbar-track);
    }

    *::-webkit-scrollbar {
        width: 9px;
        height: 9px;
    }

    *::-webkit-scrollbar-track {
        background: var(--scrollbar-track);
    }

    *::-webkit-scrollbar-thumb {
        background: var(--scrollbar-thumb);
        border-radius: 10px;
    }

    *::-webkit-scrollbar-thumb:hover {
        background: var(--scrollbar-thumb-hover);
    }

    """


# ============================================================
# LOAD CUSTOM CSS
# ============================================================

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


# ============================================================
# STATE SYNCHRONIZATION CALLBACK
# ============================================================

def _on_toggle_change():
    """Callback triggered instantly when the user toggles the switch."""
    st.session_state.dark_mode = st.session_state.theme_toggle


# ============================================================
# APPLY THEME
# ============================================================

def apply_theme() -> str:
    """
    Applies the current theme CSS variables and Streamlit overrides.
    Should be called at the very top of each page script.
    """
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    palette = (
        DARK_THEME if st.session_state.dark_mode else LIGHT_THEME
    )

    css = (
        _build_variable_block(palette)
        + "\n"
        + _build_streamlit_overrides()
        + "\n"
        + _load_style_css()
    )

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    return "dark" if st.session_state.dark_mode else "light"


# ============================================================
# CUSTOM SIDEBAR
# ============================================================

def render_sidebar(
    total_patients=None,
    model_accuracy="83.3%",
    total_diagnoses=None
):
    # Initialize state variables safely
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    if "theme_toggle" not in st.session_state:
        st.session_state.theme_toggle = st.session_state.dark_mode

    with st.sidebar:

        # ====================================================
        # BRAND
        # ====================================================

        st.html(
            """
            <div style="
                text-align:center;
                padding:10px 0 10px 0;
            ">
                <div style="
                    font-size:42px;
                    line-height:1;
                    margin-bottom:8px;
                ">
                    🏥
                </div>

                <div style="
                    font-size:21px;
                    font-weight:800;
                    color:var(--text-primary);
                    line-height:1.2;
                ">
                    Hospital Analytics
                </div>

                <div style="
                    font-size:12px;
                    margin-top:5px;
                    color:var(--text-secondary);
                ">
                    Healthcare Intelligence Platform
                </div>
            </div>
            """
        )

        # ====================================================
        # THEME SWITCH (TOP OF SIDEBAR)
        # ====================================================

        st.toggle(
            "🌙 Dark Mode",
            key="theme_toggle",
            on_change=_on_toggle_change,
            help="Switch between dark and light theme"
        )

        st.markdown("---")

        # ====================================================
        # SYSTEM STATUS
        # ====================================================

        st.html(
            """
            <div style="
                padding:12px 15px;
                border-radius:12px;
                background:rgba(16,185,129,0.16);
                border:1px solid rgba(16,185,129,0.25);
                margin-bottom:12px;
            ">
                <div style="
                    font-size:14px;
                    font-weight:600;
                    color:var(--text-primary);
                ">
                    🟢 System Online
                </div>

                <div style="
                    font-size:11px;
                    margin-top:3px;
                    color:var(--text-secondary);
                ">
                    Analytics platform operational
                </div>
            </div>
            """
        )

        st.markdown("---")

        # ====================================================
        # NAVIGATION
        # ====================================================

        st.markdown("### 🧭 Navigation")

        st.page_link(
            "app.py",
            label="Home",
            icon="🏠"
        )

        st.page_link(
            "pages/About.py",
            label="About",
            icon="ℹ️"
        )

        st.page_link(
            "pages/Dashboard.py",
            label="Dashboard",
            icon="📊"
        )

        st.page_link(
            "pages/Model_Performance.py",
            label="Model Performance",
            icon="📈"
        )

        st.page_link(
            "pages/Predict.py",
            label="Predict",
            icon="🤖"
        )

        st.markdown("---")

        # ====================================================
        # SYSTEM OVERVIEW
        # ====================================================

        st.markdown("### 📊 System Overview")

        if total_patients is not None:
            st.metric(
                "Patients",
                f"{total_patients:,}"
            )

        st.metric(
            "Model Accuracy",
            model_accuracy
        )

        if total_diagnoses is not None:
            st.metric(
                "Diagnoses",
                total_diagnoses
            )

        st.markdown("---")

        # Re-apply CSS variables in sidebar frame
        apply_theme()

        # ====================================================
        # SIDEBAR FOOTER
        # ====================================================

        st.html(
            """
            <div style="
                text-align:center;
                padding:10px 0 5px 0;
            ">
                <div style="
                    font-size:11px;
                    color:var(--text-muted);
                ">
                    Hospital Patient Analytics
                </div>

                <div style="
                    font-size:10px;
                    margin-top:4px;
                    color:var(--text-muted);
                ">
                    v1.0 • Data Analytics & ML
                </div>

                <div style="
                    font-size:10px;
                    margin-top:4px;
                    color:var(--text-muted);
                ">
                    Developed by Ayush & Moon
                </div>
            </div>
            """
        )


# ============================================================
# PLOTLY THEME
# ============================================================

def plotly_template() -> str:
    return (
        "plotly_dark"
        if st.session_state.get("dark_mode", True)
        else "plotly_white"
    )