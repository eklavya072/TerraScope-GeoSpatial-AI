import streamlit as st
import sys
import os
import numpy as np
from PIL import Image
import io
import base64
from config import DATA_CONFIG

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from model_handler import ModelHandler
from visualizer import Visualizer

st.set_page_config(page_title="TerraScope", layout="wide", initial_sidebar_state="expanded")

# --- INIT SESSION STATE ---
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'model_handler' not in st.session_state:
    mh = ModelHandler()
    if mh.load_model():
        mh.load_class_indices()
    st.session_state.model_handler = mh
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None
if 'classification_fig' not in st.session_state:
    st.session_state.classification_fig = None
if 'classification_status' not in st.session_state:
    st.session_state.classification_status = 'idle'

mh = st.session_state.model_handler
viz = st.session_state.visualizer

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0&display=swap');

    :root {
        --primary: #012d1d;
        --primary-container: #1b4332;
        --on-primary-container: #86af99;
        --surface: #f8f9ff;
        --surface-container: #e5eeff;
        --surface-container-high: #dce9ff;
        --surface-container-low: #eff4ff;
        --on-surface: #0b1c30;
        --on-surface-variant: #414844;
        --outline-variant: #c1c8c2;
        --on-background: #0b1c30;
    }

    * { font-family: 'Inter', sans-serif; box-sizing: border-box; }

    #MainMenu, .stAppToolbar, .stDecoration, .stAppDeployButton, footer,
    [data-testid="stToolbar"], [data-testid="stToolbarActions"], [data-testid="stStatusWidget"],
    .stApp header button[title="View app info"], .stApp header button[aria-label="View app info"],
    .stApp header button, .stApp header [role="button"], .stApp header .stButton,
    button[title="View app info"], button[aria-label="View app info"],
    .stButton[kind="header"], .stButton[kind="header"] *,
    header button, header [role="button"], header .stButton {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    header[data-testid="stHeader"] { display: none !important; height: 0 !important; visibility: hidden !important; }
    button[kind="header"] { display: none !important; visibility: hidden !important; }
    .stApp header { display: none !important; visibility: hidden !important; }

    .stApp { background: var(--surface) !important; }
    .block-container {
        max-width: 1200px !important;
        padding: 0 32px 48px !important;
        margin: 0 auto !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--primary) !important;
        width: 240px !important;
        min-width: 240px !important;
        border-right: 1px solid var(--outline-variant) !important;
    }
    section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown p { color: inherit !important; }

    section[data-testid="stSidebar"] .stButton button {
        all: unset !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 12px !important;
        padding: 14px 16px !important;
        border-radius: 8px !important;
        color: rgba(255,255,255,0.7) !important;
        font-size: 17px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        width: calc(100% - 24px) !important;
        margin: 0 auto 4px !important;
        background: transparent !important;
        line-height: 1.2 !important;
        box-sizing: border-box !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background: rgba(255,255,255,0.05) !important;
        color: #fff !important;
    }
    section[data-testid="stSidebar"] .stButton button[data-testid="baseButton-primary"] {
        background: rgba(134,175,153,0.15) !important;
        color: #c1ecd4 !important;
    }
    section[data-testid="stSidebar"] .stButton button p {
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        margin: 0 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        color: inherit !important;
    }
    section[data-testid="stSidebar"] .stButton { margin: 0 !important; padding: 0 !important; }
    section[data-testid="stSidebar"] .stButton button:focus { box-shadow: none !important; outline: none !important; }

    .st-key-nav_Home .stButton button p::before,
    .st-key-nav_Charts .stButton button p::before,
    .st-key-nav_Classes .stButton button p::before {
        font-family: 'Material Symbols Outlined';
        font-size: 20px;
        font-weight: normal;
        line-height: 1;
    }
    .st-key-nav_Home .stButton button p::before { content: 'home'; }
    .st-key-nav_Charts .stButton button p::before { content: 'bar_chart'; }
    .st-key-nav_Classes .stButton button p::before { content: 'category'; }

    /* Top utility bar */
    .top-bar {
        height: 64px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        border-bottom: 1px solid var(--outline-variant);
        background: var(--surface);
        margin: 0 -32px 32px;
        padding: 0 32px;
        position: sticky;
        top: 0;
        z-index: 10;
    }
    .top-bar-btn {
        width: 32px; height: 32px;
        display: flex; align-items: center; justify-content: center;
        border-radius: 999px; color: var(--on-surface-variant);
        cursor: pointer;
    }
    .top-bar-btn:hover { background: var(--surface-container); }

    /* Hero */
    .hero-wrap {
        position: relative;
        width: 100%;
        aspect-ratio: 16/8;
        border-radius: 16px;
        overflow: hidden;
        background: var(--on-background);
        margin-bottom: 32px;
        border: 1px solid rgba(193,200,194,0.2);
        box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25);
    }
    .hero-wrap img { width: 100%; height: 100%; object-fit: cover; opacity: 0.9; }
    .hero-gradient {
        position: absolute; inset: 0;
        background: linear-gradient(to top, rgba(0,0,0,0.6), transparent);
        display: flex; flex-direction: column; justify-content: flex-end;
        padding: 32px;
    }
    .hero-badge {
        background: rgba(0,0,0,0.4);
        backdrop-filter: blur(12px);
        padding: 8px 16px;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.15);
        transition: all 0.3s ease;
    }
    .hero-badge:hover {
        background: rgba(0,0,0,0.5);
        border-color: rgba(255,255,255,0.25);
    }
    .hero-badge-label {
        color: #fff !important;
        font-size: 10px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.1em; margin: 0;
    }
    .hero-badge-value {
        color: #fff !important; font-size: 13px; font-weight: 500; margin: 0;
    }
    .hero-compass {
        position: absolute; top: 16px; right: 16px;
        width: 36px; height: 36px;
        display: flex; align-items: center; justify-content: center;
        color: rgba(255,255,255,0.9);
    }
    .hero-fullscreen {
        width: 40px; height: 40px;
        display: flex; align-items: center; justify-content: center;
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px; color: #fff;
        backdrop-filter: blur(4px);
    }

    /* Upload area */
    .upload-label-box {
        background: var(--primary-container) !important;
        border: 2px solid var(--primary-container) !important;
        border-radius: 12px !important;
        height: 56px !important;
        display: flex !important;
        align-items: center !important;
        padding: 0 20px !important;
        color: #fff !important;
        font-size: 16px !important;
        font-weight: 700 !important;
    }
    .upload-plus-btn, .st-key-upload [data-testid="stFileUploader"] {
        background: var(--primary-container) !important;
        border: 2px solid var(--primary-container) !important;
        border-radius: 12px !important;
        height: 56px !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: #fff !important;
        font-size: 30px !important;
        font-weight: 200 !important;
        cursor: pointer !important;
    }
    .upload-plus-btn:hover, .st-key-upload [data-testid="stFileUploader"]:hover {
        opacity: 0.85 !important;
    }
    .st-key-upload [data-testid="stFileUploader"] {
        position: relative !important;
        margin-top: -56px !important;
        opacity: 0 !important;
        z-index: 2 !important;
    }
    .st-key-upload [data-testid="stFileUploader"] section {
        all: unset !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
        height: 100% !important;
    }
    .st-key-upload [data-testid="stFileUploader"] button {
        all: unset !important;
        width: 100% !important;
        height: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
    }
    .st-key-upload [data-testid="stFileUploader"] small,
    .st-key-upload [data-testid="stFileUploaderDropzoneInstructions"],
    .st-key-upload [data-testid="stFileUploaderFileName"] {
        display: none !important;
    }
    .st-key-upload [data-testid="stFileUploaderDropzone"] {
        all: unset !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
        height: 100% !important;
    }
    .st-key-upload [data-testid="stFileUploaderLabel"] { display: none !important; }
    .st-key-clear_file .stButton button {
        all: unset !important;
        cursor: pointer !important;
        color: var(--on-surface-variant) !important;
        font-size: 18px !important;
        padding: 4px 8px !important;
        position: absolute !important;
        right: 72px !important;
        top: 14px !important;
        z-index: 11 !important;
    }
    .st-key-clear_file .stButton button:hover { color: #ba1a1a !important; }
    .st-key-clear_file .stButton button p { margin: 0 !important; font-size: 18px !important; }

    /* Section headings */
    .section-title {
        font-size: 20px; font-weight: 700;
        color: var(--on-surface);
        margin: 0 0 16px;
    }
    .page-title {
        font-size: 40px; font-weight: 900;
        color: var(--on-surface);
        letter-spacing: -0.02em;
        margin: 0 0 4px;
        padding-top: 16px;
    }
    .page-subtitle {
        color: var(--on-surface-variant);
        font-size: 14px; font-weight: 500;
        margin: 0 0 32px;
    }

    /* Preview box */
    .preview-box {
        aspect-ratio: 1;
        border-radius: 16px;
        border: 1px solid var(--outline-variant);
        background: var(--on-background);
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
    }
    .preview-box:hover {
        border-color: var(--primary-container);
        box-shadow: 0 4px 12px rgba(1,45,29,0.1);
    }
    .preview-empty {
        display: flex; flex-direction: column;
        align-items: center; gap: 16px;
        color: rgba(65,72,68,0.7);
    }
    .preview-box img { width: 100%; height: 100%; object-fit: cover; }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-classify_btn) {
        background: #fff !important;
        border: 1px solid var(--outline-variant) !important;
        border-radius: 16px !important;
        padding: 20px 24px 24px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.02) !important;
        min-height: 300px;
        transition: all 0.3s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-classify_btn):hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08), 0 8px 24px rgba(0,0,0,0.04) !important;
    }
    .status-row {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 12px;
    }
    .status-label {
        font-size: 11px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.05em;
        color: var(--on-surface-variant);
    }
    .badge {
        font-size: 10px; font-weight: 700;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 999px;
    }
    .badge-idle { background: var(--surface-container-high); color: var(--on-surface-variant); }
    .badge-processing { background: #fff3cd; color: #856404; }
    .badge-complete { background: #d4edda; color: #155724; }
    .status-msg {
        display: flex; align-items: center; gap: 12px;
        color: var(--on-surface-variant); font-size: 14px;
        margin-bottom: 24px;
    }
    .result-box {
        background: var(--primary-container);
        color: #c1ecd4;
        padding: 16px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 16px;
        border: 1px solid rgba(134,175,153,0.2);
        box-shadow: 0 2px 8px rgba(1,45,29,0.1);
    }
    .confidence-text {
        color: var(--on-surface);
        font-weight: 700;
        font-size: 16px;
        margin: 0 0 16px;
    }
    .chart-toolbar {
        display: flex; align-items: center; gap: 16px;
        color: rgba(65,72,68,0.7);
        margin-bottom: 16px;
        padding-bottom: 16px;
        border-bottom: 1px solid rgba(193,200,194,0.3);
    }
    .chart-toolbar span {
        font-family: 'Material Symbols Outlined';
        font-size: 20px;
        cursor: default;
    }
    .chart-section-title {
        font-size: 12px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.05em;
        color: rgba(65,72,68,0.7);
        margin: 0 0 8px;
    }
    .results-divider {
        border-top: 1px solid rgba(193,200,194,0.3);
        padding-top: 24px;
        margin-top: 24px;
    }

    /* Classify button */
    .st-key-classify_btn .stButton button {
        background: var(--primary) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 16px 24px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        width: 100% !important;
        box-shadow: 0 10px 15px -3px rgba(1,45,29,0.15) !important;
        transition: background 0.2s !important;
    }
    .st-key-classify_btn .stButton button:hover {
        background: var(--primary-container) !important;
        color: #fff !important;
        border: none !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 20px -3px rgba(1,45,29,0.2) !important;
    }
    .st-key-classify_btn .stButton button p {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        margin: 0 !important;
        color: #fff !important;
        font-weight: 700 !important;
    }
    .st-key-classify_btn .stButton button p::before {
        font-family: 'Material Symbols Outlined';
        content: 'analytics';
        font-size: 20px;
        font-weight: normal;
    }

    /* Plotly chart */
    .stPlotlyChart { margin-top: 0 !important; }

    /* Charts / Classes pages - FORCE WHITE TEXT */
    .stTabs [data-baseweb="tab-list"] button *,
    .stTabs [data-baseweb="tab-list"] button,
    .stTabs [data-baseweb="tab"] *,
    .stTabs [data-baseweb="tab"] {
        color: #ffffff !important;
    }
    h1, h2, h3, h4, h5, h6, p, li, .stText, .stWrite, .stSubheader,
    .stExpander summary, .stExpander summary span {
        color: var(--on-surface) !important;
    }
    .stExpander svg, .stExpander svg * { fill: var(--on-surface) !important; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--primary) !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: #ffffff !important;
        border-color: rgba(255,255,255,0.3) !important;
    }
    .stTabs [role="tablist"] [role="indicator"],
    .stTabs [data-baseweb="tab-list"] [role="indicator"],
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs div[role="tablist"] > div:last-child,
    [data-testid="stTabs"] div[role="tablist"] > div:not([role="tab"]) {
        display: none !important;
        height: 0 !important;
        background: transparent !important;
        border: none !important;
    }
    div[data-testid="stExpander"] {
        background: var(--primary) !important;
        border: 1px solid #ffffff !important;
        border-radius: 12px !important;
    }
    div[data-testid="stExpander"]:not(:hover) {
        background: var(--primary) !important;
        border-color: #ffffff !important;
    }
    div[data-testid="stExpander"]:hover {
        background: var(--primary) !important;
        border-color: #ffffff !important;
    }
    div[data-testid="stExpander"]:focus,
    div[data-testid="stExpander"]:focus-within,
    div[data-testid="stExpander"]:active,
    div[data-testid="stExpander"][aria-expanded="true"],
    div[data-testid="stExpander"] summary:focus,
    div[data-testid="stExpander"] summary:hover,
    div[data-testid="stExpander"] summary:active,
    div[data-testid="stExpander"] summary {
        background: var(--primary) !important;
        border-color: #ffffff !important;
    }
    div[data-testid="stExpander"] > div > div > div > p,
    div[data-testid="stExpander"] > div > div > div > div > p,
    div[data-testid="stExpander"] > div > div > div > div > div > p,
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary span,
    div[data-testid="stExpander"] > div > div > div > div > div > div > p,
    div[data-testid="stExpander"] > div > div > div > div > div > div > li,
    div[data-testid="stExpander"] > div > div > div > div > div > div > ul > li {
        color: #fff !important;
    }
    div[data-testid="stExpander"] svg {
        fill: #fff !important;
    }
    /* Force all text in expanders to be white */
    div[data-testid="stExpander"] * {
        color: #fff !important;
    }
    div[data-testid="stExpander"] a,
    div[data-testid="stExpander"] .stMarkdown a {
        color: #86af99 !important;
    }
    /* Prevent background color change on expand */
    div[data-testid="stExpander"] > div {
        background: transparent !important;
    }
    div[data-testid="stExpander"] > div > div {
        background: transparent !important;
    }

    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    .spin { animation: spin 1s linear infinite; display: inline-block; }

    .material-symbols-outlined {
        font-family: 'Material Symbols Outlined';
        font-weight: normal;
        font-style: normal;
        font-size: 24px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-smoothing: antialiased;
    }
</style>
""", unsafe_allow_html=True)


def style_confidence_chart(fig):
    fig.update_layout(
        title=dict(
            text="",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=10, b=60),
        height=280,
        font=dict(family="Inter, sans-serif", color="#000"),
        xaxis=dict(
            title=dict(text="Land Cover Class", font=dict(size=11, color="#000")),
            tickangle=-45,
            tickfont=dict(size=10, color="#000"),
            showgrid=False,
            linecolor="#c1c8c2",
        ),
        yaxis=dict(
            title=dict(text="Confidence Score", font=dict(size=11, color="#000")),
            tickformat=".0%",
            tickfont=dict(color="#000"),
            showgrid=True,
            gridcolor="rgba(193,200,194,0.4)",
            linecolor="#c1c8c2",
        ),
        coloraxis_colorbar=dict(
            title=dict(text="Confidence Score", font=dict(size=10, color="#000")),
            tickformat=".0%",
            tickfont=dict(color="#000"),
            thickness=12,
            len=0.9,
        ),
    )
    fig.update_traces(marker_line_width=0)
    return fig


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 24px 4px;display:flex;align-items:center;gap:12px;">
        <div style="width:64px;height:64px;background:rgba(134,175,153,0.25);border-radius:10px;display:flex;align-items:center;justify-content:center;">
            <span class="material-symbols-outlined" style="color:#1b4332;font-size:38px;">public</span>
        </div>
        <h1 style="color:#fff;font-size:28px;font-weight:700;margin:0;letter-spacing:-0.02em;">TerraScope</h1>
    </div>
    <div style="padding:4px 28px 8px;text-align:center;">
        <p style="color:rgba(134,175,153,0.6);font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;margin:0;">Navigation</p>
    </div>
    """, unsafe_allow_html=True)

    for label in ["Home", "Charts", "Classes"]:
        active = st.session_state.page == label
        if st.button(label, key=f"nav_{label}", use_container_width=True, type="primary" if active else "secondary"):
            st.session_state.page = label
            st.rerun()


# --- TOP UTILITY BAR ---
st.markdown("""
<div class="top-bar">
    <div class="top-bar-btn">
        <span class="material-symbols-outlined" style="font-size:20px;">more_vert</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ===========================
# HOME
# ===========================
if st.session_state.page == "Home":
    st.markdown("""
    <div style="margin-bottom:32px;">
        <h2 class="page-title">TerraScope Geospatial AI</h2>
        <p class="page-subtitle">Welcome to the Land Cover Classification System. Analyze multi-spectral data with AI precision.</p>
    </div>
    """, unsafe_allow_html=True)

    sat_path = "assets/satellite.jpg"
    if os.path.exists(sat_path):
        with open(sat_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <div class="hero-wrap">
            <img src="data:image/jpeg;base64,{b64}" alt="Live Satellite Feed">
            <div class="hero-compass">
                <span class="material-symbols-outlined" style="font-size:28px;">explore</span>
            </div>
            <div class="hero-gradient">
                <div style="display:flex;align-items:center;justify-content:space-between;">
                    <div style="display:flex;gap:16px;flex-wrap:wrap;">
                        <div class="hero-badge">
                            <p class="hero-badge-label" style="color:#fff;">Active Sensor</p>
                            <p class="hero-badge-value" style="color:#fff;">Sentinel-2B Multispectral</p>
                        </div>
                    </div>
                    <div class="hero-fullscreen">
                        <span class="material-symbols-outlined" style="font-size:20px;">fullscreen</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    has_image = False
    image_bytes = None

    upl_col1, upl_col2 = st.columns([4, 1])
    with upl_col1:
        st.markdown('<div class="upload-label-box">Upload your image</div>', unsafe_allow_html=True)
    with upl_col2:
        st.markdown('<div class="upload-plus-btn" id="upload-plus">+</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "", type=DATA_CONFIG['allowed_formats'], key="upload", label_visibility="collapsed"
        )

    if uploaded_file is not None:
        raw = uploaded_file.read()
        if ('uploaded_image' not in st.session_state or
            st.session_state.uploaded_image is None or
            uploaded_file.name != st.session_state.get('uploaded_image_name')):
            st.session_state.uploaded_image = raw
            st.session_state.uploaded_image_name = uploaded_file.name
            st.session_state.classification_result = None
            st.session_state.classification_fig = None
            st.session_state.classification_status = 'idle'
        has_image = True
        image_bytes = st.session_state.uploaded_image
    elif 'uploaded_image' in st.session_state and st.session_state.uploaded_image is not None:
        has_image = True
        image_bytes = st.session_state.uploaded_image

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<h3 class="section-title">Image Preview</h3>', unsafe_allow_html=True)
        if has_image:
            img_b64 = base64.b64encode(image_bytes).decode()
            st.markdown(f"""
            <div class="preview-box">
                <img src="data:image/jpeg;base64,{img_b64}" alt="Uploaded preview">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="preview-box">
                <div class="preview-empty">
                    <span class="material-symbols-outlined" style="font-size:40px;">image_not_supported</span>
                    <p style="font-size:14px;font-weight:500;margin:0;">No image uploaded</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<h3 class="section-title">Classification</h3>', unsafe_allow_html=True)

        result = st.session_state.classification_result
        status = st.session_state.classification_status
        show_results = result is not None and status == 'complete'

        status_map = {
            'idle': ('badge-idle', 'IDLE', 'Ready to begin analysis...', 'progress_activity', 'spin'),
            'processing': ('badge-processing', 'PROCESSING', 'Analyzing image features...', 'hourglass_top', 'spin'),
            'complete': ('badge-complete', 'COMPLETE', 'Classification finished successfully.', 'check_circle', ''),
        }
        status_badge, status_text, status_msg, status_icon, icon_class = status_map[status]

        with st.container(border=True):
            st.markdown(f"""
                <div class="status-row">
                    <span class="status-label">Status</span>
                    <span class="badge {status_badge}">{status_text}</span>
                </div>
                <div class="status-msg">
                    <span class="material-symbols-outlined {icon_class}" style="font-size:20px;">{status_icon}</span>
                    <span>{status_msg}</span>
                </div>
            """, unsafe_allow_html=True)

            if status == 'processing' and has_image:
                try:
                    with st.spinner("Classifying image..."):
                        pred_result = mh.predict(image_bytes)
                        class_names = [
                            mh.class_indices.get(str(i), f"Class_{i}")
                            for i in range(len(pred_result['all_predictions']))
                        ]
                        fig = style_confidence_chart(
                            viz.plot_confidence_bar(class_names, pred_result['all_predictions'])
                        )
                        st.session_state.classification_result = pred_result
                        st.session_state.classification_fig = fig
                        st.session_state.classification_status = 'complete'
                        st.rerun()
                except Exception as e:
                    st.error(f"Classification failed: {e}")
                    st.session_state.classification_status = 'idle'
            elif has_image:
                if st.button("Run Classification", key="classify_btn", use_container_width=True):
                    st.session_state.classification_status = 'processing'
                    st.rerun()

            if show_results:
                result = st.session_state.classification_result
                st.markdown(f"""
                <div class="results-divider">
                    <div class="result-box">Classification Result: {result["class_name"]}</div>
                    <p class="confidence-text">Confidence: <span style="font-weight:400;color:#414844;">{result["confidence"]:.2%}</span></p>
                    <div class="chart-toolbar">
                        <span>photo_camera</span>
                        <span>zoom_in</span>
                        <span>open_with</span>
                        <span>add_box</span>
                        <span>indeterminate_check_box</span>
                        <span>fullscreen_exit</span>
                        <span>home</span>
                        <span>fullscreen</span>
                    </div>
                    <p class="chart-section-title">Classification Confidence Scores</p>
                </div>
                """, unsafe_allow_html=True)
                if st.session_state.classification_fig is not None:
                    st.plotly_chart(
                        st.session_state.classification_fig,
                        use_container_width=True,
                        config={'displayModeBar': False},
                    )


# ===========================
# CHARTS
# ===========================
elif st.session_state.page == "Charts":
    st.markdown("""
    <h2 class="page-title" style="font-size:28px;margin-bottom:8px;">Charts and Visualizations</h2>
    <p class="page-subtitle" style="margin-bottom:24px;">Model evaluation metrics and image analysis tools.</p>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Model Evaluation", "Image Analysis"])
    with tab1:
        st.subheader("Model Performance")
        if os.path.exists("assets/model_performance.jpg"):
            st.image("assets/model_performance.jpg", caption="Training Progress Over Time", use_container_width=True)

    with tab2:
        st.subheader("Image Analysis")
        if 'uploaded_image' in st.session_state and st.session_state.uploaded_image is not None:
            u = st.session_state.uploaded_image
            st.image(u, caption="Uploaded Image", width=400)
            try:
                img = Image.open(io.BytesIO(u))
                arr = np.array(img)
                st.subheader("RGB Color Histograms")
                figs = viz.plot_rgb_histograms(arr)
                ca, cb, cc = st.columns(3)
                for i, c in enumerate([ca, cb, cc]):
                    with c:
                        st.pyplot(figs[i])
                at = st.selectbox("Select Analysis Type", ["Image Statistics", "Edge Detection", "Intensity Map"])
                if at == "Image Statistics":
                    for k, v in viz.image_statistics(arr).items():
                        st.write(f"**{k}:** {v}")
                elif at == "Edge Detection":
                    st.image(viz.edge_detection(arr), caption="Edge Detection", width=400)
                elif at == "Intensity Map":
                    st.plotly_chart(viz.intensity_map(arr), use_container_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
        else:
            st.warning("Please upload an image on the Home page first.")


# ===========================
# CLASSES
# ===========================
elif st.session_state.page == "Classes":
    st.markdown("""
    <h2 class="page-title" style="font-size:28px;margin-bottom:8px;">Terrascope Classes</h2>
    <p class="page-subtitle" style="margin-bottom:24px;">Reference guide for all supported EuroSAT classification categories.</p>
    """, unsafe_allow_html=True)

    class_info = [
        ("AnnualCrop", "Agricultural areas where crops are planted and harvested within a single year.", "assets/annualcrop.jpeg"),
        ("Forest", "Areas dominated by trees, forming a continuous canopy.", "assets/forest.jpeg"),
        ("HerbaceousVegetation", "Areas covered by non-woody plants and grasses.", "assets/herbaceous_vegetation.jpeg"),
        ("Highway", "Major roads and transportation infrastructure.", "assets/highway.jpeg"),
        ("Industrial", "Areas containing factories, warehouses, and industrial facilities.", "assets/industrial.jpeg"),
        ("Pasture", "Land used for grazing livestock.", "assets/pasture.avif"),
        ("PermanentCrop", "Agricultural areas with long-term crops like orchards and vineyards.", "assets/permanent_crop.jpeg"),
        ("Residential", "Areas containing houses and residential buildings.", "assets/residential.png"),
        ("River", "Natural watercourses and their immediate surroundings.", "assets/river.jpeg"),
        ("SeaLake", "Large bodies of water including seas and lakes.", "assets/sealake.jpeg"),
    ]
    
    cols = st.columns(2)
    for idx, (name, desc, image_path) in enumerate(class_info):
        with cols[idx % 2]:
            with st.expander(name):
                st.write(desc)
                if os.path.exists(image_path):
                    st.image(image_path, caption=f"Example of {name}", use_container_width=True)
                else:
                    st.info(f"Demo image for {name} not found")

