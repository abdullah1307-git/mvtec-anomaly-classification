import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import requests
from pathlib import Path
import pandas as pd

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AnomalyVision — Industrial Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS + ANIMATIONS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
[data-testid="stSidebar"] { display: none; }

/* ---- APP BACKGROUND ---- */
.stApp {
    background: #080c14;
    background-image:
        radial-gradient(ellipse at 20% 10%, rgba(59,130,246,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 90%, rgba(139,92,246,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(6,182,212,0.03) 0%, transparent 60%);
}

/* ---- TOP NAV ---- */
.topnav {
    background: rgba(8,12,20,0.95);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid #1e2d3d;
    padding: 0 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 56px;
    margin: -1rem -1rem 2rem -1rem;
    position: sticky;
    top: 0;
    z-index: 999;
}

/* ---- HERO ---- */
.hero {
    background: linear-gradient(180deg, #0d1117 0%, #080c14 100%);
    border: 1px solid #1e2d3d;
    border-radius: 20px;
    padding: 64px 48px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 32px;
}

.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.7), transparent);
}

.hero::after {
    content: '';
    position: absolute;
    top: -180px; left: 50%;
    transform: translateX(-50%);
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.1) 0%, transparent 70%);
    pointer-events: none;
    animation: pulse-glow 4s ease-in-out infinite;
}

@keyframes pulse-glow {
    0%, 100% { opacity: 0.6; transform: translateX(-50%) scale(1); }
    50% { opacity: 1; transform: translateX(-50%) scale(1.1); }
}

/* ---- TYPING ANIMATION ---- */
.typing-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.8rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #93c5fd 40%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: inline-block;
    overflow: hidden;
    border-right: 3px solid #3b82f6;
    white-space: nowrap;
    animation: typing 2s steps(20, end) forwards, blink 0.8s step-end infinite;
    max-width: 100%;
}

@keyframes typing {
    from { width: 0; }
    to { width: 100%; }
}

@keyframes blink {
    50% { border-color: transparent; }
}

/* ---- BADGE ---- */
.badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.25);
    color: #60a5fa;
    padding: 5px 14px;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 20px;
    margin-left: auto;
    margin-right: auto;
    font-family: 'JetBrains Mono', monospace;
    animation: fadeIn 0.5s ease both;
}

.badge .dot {
    width: 6px; height: 6px;
    background: #3b82f6;
    border-radius: 50%;
    animation: ping 1.5s ease-in-out infinite;
}

@keyframes ping {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.5); opacity: 0.5; }
}

/* ---- STAT BOXES with counter animation ---- */
.stats-row {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 32px;
}

.stat-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid #1e2d3d;
    border-radius: 14px;
    padding: 16px 24px;
    text-align: center;
    min-width: 110px;
    transition: border-color 0.3s, transform 0.3s, background 0.3s;
    animation: slideUp 0.6s ease both;
}

.stat-box:hover {
    border-color: rgba(59,130,246,0.5);
    transform: translateY(-4px);
    background: rgba(59,130,246,0.08);
}

.stat-v {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #3b82f6;
    line-height: 1;
}

.stat-l {
    font-size: 0.66rem;
    color: #6e7681;
    margin-top: 5px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 500;
}

/* ---- ANIMATIONS ---- */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.95); }
    to { opacity: 1; transform: scale(1); }
}

.anim-fade { animation: fadeIn 0.5s ease both; }
.anim-slide { animation: slideUp 0.5s ease both; }
.anim-scale { animation: scaleIn 0.4s ease both; }

/* ---- CARDS ---- */
.gcard {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 14px;
    transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
}

.gcard:hover {
    border-color: #2d4a6e;
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.clabel {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #484f58;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 14px;
}

/* ---- EYEBROW ---- */
.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 8px;
}

/* ---- SECTION TITLE ---- */
.sec-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #f0f6fc;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
}

.sec-body { color: #8b949e; font-size: 0.9rem; line-height: 1.7; }

/* ---- KPI ---- */
.kpi {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
}
.kpi:hover {
    border-color: rgba(59,130,246,0.4);
    transform: translateY(-3px);
    box-shadow: 0 6px 24px rgba(59,130,246,0.1);
}
.kpi-v { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #3b82f6; line-height: 1; }
.kpi-l { font-size: 0.68rem; color: #6e7681; margin-top: 6px; letter-spacing: 0.04em; }

/* ---- RESULT CARDS ---- */
.res-pass {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(16,185,129,0.04));
    border: 2px solid rgba(16,185,129,0.5);
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    animation: scaleIn 0.4s ease both;
    box-shadow: 0 0 40px rgba(16,185,129,0.1), inset 0 1px 0 rgba(16,185,129,0.2);
}

.res-fail {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(239,68,68,0.04));
    border: 2px solid rgba(239,68,68,0.5);
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    animation: scaleIn 0.4s ease both;
    box-shadow: 0 0 40px rgba(239,68,68,0.1), inset 0 1px 0 rgba(239,68,68,0.2);
}

.res-icon { font-size: 3.5rem; line-height: 1; margin-bottom: 12px; }
.res-verdict { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin: 0; }
.res-conf { font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 600; margin-top: 8px; }
.res-sub { font-size: 0.82rem; color: #8b949e; margin-top: 4px; }

/* ---- COMPARISON SLIDER ---- */
.slider-container {
    position: relative;
    width: 100%;
    overflow: hidden;
    border-radius: 12px;
    cursor: ew-resize;
    user-select: none;
}

.slider-before, .slider-after {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
}

.slider-divider {
    position: absolute;
    top: 0; bottom: 0;
    width: 2px;
    background: white;
    cursor: ew-resize;
    z-index: 10;
}

.slider-handle {
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 40px; height: 40px;
    background: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.5);
    font-size: 1rem;
    cursor: ew-resize;
}

/* ---- SAMPLE IMAGE HOVER ---- */
.sample-card {
    position: relative;
    border-radius: 10px;
    overflow: hidden;
    border: 2px solid #1e2d3d;
    transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
    cursor: pointer;
}

.sample-card:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}

.sample-card.normal { border-color: rgba(16,185,129,0.4); }
.sample-card.anomaly { border-color: rgba(239,68,68,0.3); }

.sample-card.normal:hover { border-color: #10b981; box-shadow: 0 12px 40px rgba(16,185,129,0.15); }
.sample-card.anomaly:hover { border-color: #ef4444; box-shadow: 0 12px 40px rgba(239,68,68,0.15); }

.sample-overlay {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    background: linear-gradient(transparent, rgba(0,0,0,0.85));
    padding: 20px 10px 8px;
    transform: translateY(100%);
    transition: transform 0.25s ease;
}

.sample-card:hover .sample-overlay { transform: translateY(0); }

/* ---- BUTTONS ---- */
.stButton > button,
.stButton > button:focus,
div[data-testid="stButton"] > button {
    background-color: #2563eb !important;
    background-image: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 18px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: filter 0.15s ease, transform 0.15s ease !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.4) !important;
    letter-spacing: 0.01em !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

.stButton > button:hover,
div[data-testid="stButton"] > button:hover {
    filter: brightness(1.15) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.5) !important;
}

.stButton > button:active,
div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
    filter: brightness(0.95) !important;
}

.stButton > button:disabled,
div[data-testid="stButton"] > button:disabled {
    background-color: #1e2d3d !important;
    background-image: none !important;
    color: #484f58 !important;
    box-shadow: none !important;
    transform: none !important;
    filter: none !important;
    cursor: not-allowed !important;
}

/* ---- INFO BOX ---- */
.infobox {
    background: rgba(59,130,246,0.07);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #8b949e;
    line-height: 1.6;
}

/* ---- TECH BADGE ---- */
.tbadge {
    display: inline-block;
    background: rgba(255,255,255,0.04);
    border: 1px solid #1e2d3d;
    border-radius: 7px;
    padding: 5px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #8b949e;
    margin: 3px;
    transition: all 0.2s;
}
.tbadge:hover { border-color: #3b82f6; color: #60a5fa; background: rgba(59,130,246,0.08); }

/* ---- CAT PILL ---- */
.cpill {
    display: inline-block;
    background: rgba(139,92,246,0.08);
    border: 1px solid rgba(139,92,246,0.2);
    color: #a78bfa;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 0.75rem;
    margin: 3px;
    transition: all 0.2s;
}
.cpill:hover { background: rgba(139,92,246,0.18); border-color: rgba(139,92,246,0.4); }

/* ---- MODEL CARDS ---- */
.mcard {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 14px;
    padding: 24px;
    height: 100%;
    transition: transform 0.25s, border-color 0.25s, box-shadow 0.25s;
}
.mcard:hover { transform: translateY(-4px); box-shadow: 0 10px 40px rgba(0,0,0,0.4); }
.mcard.best { border-color: rgba(16,185,129,0.4); }
.mcard.mid  { border-color: rgba(59,130,246,0.3); }

/* ---- PIPELINE STEP ---- */
.pstep {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 16px 10px;
    text-align: center;
    transition: all 0.25s;
}
.pstep:hover { border-color: rgba(59,130,246,0.4); transform: translateY(-3px); box-shadow: 0 8px 24px rgba(59,130,246,0.1); }

/* ---- FOOTER ---- */
.foot {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 16px;
    padding: 28px 36px;
    margin-top: 48px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 20px;
}

/* ---- DIVIDER ---- */
.divider { border: none; border-top: 1px solid #1e2d3d; margin: 32px 0; }

/* ---- SKELETON LOADING ---- */
.skeleton {
    background: linear-gradient(90deg, #1e2d3d 25%, #2d4a6e 50%, #1e2d3d 75%);
    background-size: 200% 100%;
    animation: skeleton-loading 1.5s infinite;
    border-radius: 8px;
}

@keyframes skeleton-loading {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ---- STEP INDICATOR ---- */
.step-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 10px;
    margin-bottom: 20px;
}

.step-num-active {
    width: 24px; height: 24px;
    background: #3b82f6;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700; color: white;
    flex-shrink: 0;
}

.step-num-done {
    width: 24px; height: 24px;
    background: #10b981;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700; color: white;
    flex-shrink: 0;
}

.step-num-inactive {
    width: 24px; height: 24px;
    background: #1e2d3d;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700; color: #484f58;
    flex-shrink: 0;
}

/* ---- HOW IT WORKS STEPS ---- */
.how-step {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 14px 0;
    border-bottom: 1px solid #1e2d3d;
}
.how-step:last-child { border-bottom: none; }

/* ---- RESULT SECTION ---- */
.result-section {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 16px;
    padding: 28px;
    margin-top: 8px;
    animation: slideUp 0.4s ease both;
}

/* ---- STREAMLIT OVERRIDES ---- */
h1,h2,h3,h4,h5,h6 { color: #f0f6fc !important; }
p, li { color: #8b949e; }
.stProgress > div > div { background: linear-gradient(90deg, #3b82f6, #818cf8) !important; }
[data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 2px dashed #1e2d3d !important;
    border-radius: 12px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(59,130,246,0.4) !important; }
.stSelectbox > div > div { background: #0d1117 !important; border: 1px solid #1e2d3d !important; color: #f0f6fc !important; }
.stDataFrame { border: 1px solid #1e2d3d; border-radius: 12px; overflow: hidden; }
.stSuccess { background: rgba(16,185,129,0.1) !important; border: 1px solid rgba(16,185,129,0.3) !important; border-radius: 10px !important; color: #10b981 !important; }
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
HF_BASE     = "https://huggingface.co/abdullah130704/mvtec-anomaly-model/resolve/main"
MODEL_URL   = f"{HF_BASE}/best_model_weights.pth"
SAMPLE_BASE = f"{HF_BASE}/sample_images"

CATEGORIES = [
    'bottle','cable','capsule','carpet','grid',
    'hazelnut','leather','metal_nut','pill','screw',
    'tile','toothbrush','transistor','wood','zipper'
]

IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

SAMPLE_FILES = {
    'bottle':     ['normal_01.png','anomaly_01_broken_large.png','anomaly_02_broken_small.png'],
    'cable':      ['normal_01.png','anomaly_01_bent_wire.png','anomaly_02_cable_swap.png'],
    'capsule':    ['normal_01.png','anomaly_01_crack.png','anomaly_02_faulty_imprint.png'],
    'carpet':     ['normal_01.png','anomaly_01_color.png','anomaly_02_cut.png'],
    'grid':       ['normal_01.png','anomaly_01_bent.png','anomaly_02_broken.png'],
    'hazelnut':   ['normal_01.png','anomaly_01_crack.png','anomaly_02_cut.png'],
    'leather':    ['normal_01.png','anomaly_01_color.png','anomaly_02_cut.png'],
    'metal_nut':  ['normal_01.png','anomaly_01_bent.png','anomaly_02_color.png'],
    'pill':       ['normal_01.png','anomaly_01_color.png','anomaly_02_combined.png'],
    'screw':      ['normal_01.png','anomaly_01_manipulated_front.png','anomaly_02_scratch_head.png'],
    'tile':       ['normal_01.png','anomaly_01_crack.png','anomaly_02_glue_strip.png'],
    'toothbrush': ['normal_01.png','anomaly_01_defective.png', None],
    'transistor': ['normal_01.png','anomaly_01_bent_lead.png','anomaly_02_cut_lead.png'],
    'wood':       ['normal_01.png','anomaly_01_color.png','anomaly_02_combined.png'],
    'zipper':     ['normal_01.png','anomaly_01_broken_teeth.png','anomaly_02_combined.png'],
}

PAGES = ["🏠 Live Demo", "📋 Project Info", "📊 Results", "🔬 Explainability", "👤 About"]

# ============================================================
# SESSION STATE
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "🏠 Live Demo"
if "selected_sample_img" not in st.session_state:
    st.session_state.selected_sample_img = None
if "selected_sample_name" not in st.session_state:
    st.session_state.selected_sample_name = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# ============================================================
# TOP NAV
# ============================================================
nav_cols = st.columns([1.5, 5, 0.3])
with nav_cols[0]:
    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:800;
                background:linear-gradient(135deg,#3b82f6,#818cf8);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                background-clip:text; padding:14px 0; white-space:nowrap;">
        🔍 AnomalyVision
    </div>
    """, unsafe_allow_html=True)

with nav_cols[1]:
    btn_cols = st.columns(len(PAGES))
    for i, (col, pg) in enumerate(zip(btn_cols, PAGES)):
        with col:
            is_active = st.session_state.page == pg
            label = pg.split(" ", 1)[1]
            if st.button(label, key=f"nav_{i}", use_container_width=True):
                st.session_state.page = pg
                st.session_state.selected_sample_img = None
                st.session_state.selected_sample_name = None
                st.session_state.analysis_done = False
                st.rerun()

st.markdown("<hr style='border-color:#1e2d3d; margin:0 0 24px;'>", unsafe_allow_html=True)
page = st.session_state.page

# ============================================================
# MODEL
# ============================================================
class ConvNeXtTinyModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = convnext_tiny(weights=None)
        nf = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Linear(nf,256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256,num_classes)
        )
    def forward(self,x): return self.backbone(x)


@st.cache_resource(show_spinner=False)
def load_model():
    path = Path("best_model_weights.pth")
    if not path.exists():
        with st.spinner("⬇️ Downloading model weights (112 MB, first load only)..."):
            r = requests.get(MODEL_URL, stream=True)
            if r.status_code == 200:
                with open(path,"wb") as f:
                    for chunk in r.iter_content(8192): f.write(chunk)
            else:
                return None
    m = ConvNeXtTinyModel(2)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


def preprocess(pil_img):
    t = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    return t(pil_img.convert("RGB")).unsqueeze(0)


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.grads = self.acts = None
        layer.register_forward_hook(lambda m,i,o: setattr(self,'acts',o.detach()))
        layer.register_full_backward_hook(lambda m,gi,go: setattr(self,'grads',go[0].detach()))

    def generate(self, tensor, cls=1):
        self.model.eval()
        tensor = tensor.requires_grad_(True)
        out = self.model(tensor)
        self.model.zero_grad()
        out[0,cls].backward()
        w   = self.grads.mean(dim=[2,3], keepdim=True)
        cam = torch.relu((w*self.acts).sum(1,keepdim=True)).squeeze().cpu().numpy()
        mn,mx = cam.min(), cam.max()
        if mx-mn > 0: cam = (cam-mn)/(mx-mn)
        return cv2.resize(cam, (IMG_SIZE,IMG_SIZE))


def make_gradcam_images(pil_img, heatmap):
    """Returns original numpy array and overlay numpy array."""
    img = np.array(pil_img.resize((IMG_SIZE,IMG_SIZE))).astype(np.float32)/255
    h   = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    h   = cv2.cvtColor(h, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    ov  = np.clip(0.45*h + 0.55*img, 0, 1)
    return img, ov


def run_prediction(model, pil_img):
    tensor = preprocess(pil_img)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out,1)[0]
        pred  = out.argmax(1).item()
        conf  = probs[pred].item()
    p_norm = probs[0].item()
    p_anom = probs[1].item()
    layer  = list(model.backbone.features.children())[-1]
    gc     = GradCAM(model, layer)
    hm     = gc.generate(tensor.clone().requires_grad_(True), cls=pred)
    orig_np, overlay_np = make_gradcam_images(pil_img, hm)
    return pred, conf, p_norm, p_anom, orig_np, overlay_np


@st.cache_data(show_spinner=False)
def fetch_sample(cat, fname):
    r = requests.get(f"{SAMPLE_BASE}/{cat}/{fname}")
    if r.status_code == 200:
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return None


def make_confidence_html(p_norm, p_anom):
    """Create confidence bars using pure HTML - no library needed."""
    return f"""
    <div style="padding: 8px 0;">
        <div style="margin-bottom: 16px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#10b981; font-size:0.85rem; font-weight:600;
                             font-family:'Space Grotesk',sans-serif;">✅ Normal</span>
                <span style="color:#f0f6fc; font-size:0.85rem; font-weight:700;
                             font-family:'Space Grotesk',sans-serif;">{p_norm*100:.1f}%</span>
            </div>
            <div style="background:#1e2d3d; border-radius:100px; height:12px; overflow:hidden;">
                <div style="width:{p_norm*100:.1f}%; background:linear-gradient(90deg,#10b981,#34d399);
                            height:100%; border-radius:100px; transition:width 0.8s ease;
                            box-shadow: 0 0 8px rgba(16,185,129,0.4);"></div>
            </div>
        </div>
        <div style="margin-bottom: 8px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#ef4444; font-size:0.85rem; font-weight:600;
                             font-family:'Space Grotesk',sans-serif;">⚠️ Anomalous</span>
                <span style="color:#f0f6fc; font-size:0.85rem; font-weight:700;
                             font-family:'Space Grotesk',sans-serif;">{p_anom*100:.1f}%</span>
            </div>
            <div style="background:#1e2d3d; border-radius:100px; height:12px; overflow:hidden;">
                <div style="width:{p_anom*100:.1f}%; background:linear-gradient(90deg,#ef4444,#f87171);
                            height:100%; border-radius:100px; transition:width 0.8s ease;
                            box-shadow: 0 0 8px rgba(239,68,68,0.4);"></div>
            </div>
        </div>
    </div>
    """


def footer():
    st.markdown("""
    <div class="foot">
        <div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1rem;
                        font-weight:700; color:#f0f6fc;">🔍 AnomalyVision</div>
            <div style="font-size:0.78rem; color:#6e7681; margin-top:4px; line-height:1.6;">
                Machine Learning & Smart Systems — Final Project<br>
                University of Europe for Applied Sciences, Potsdam · SS26
            </div>
        </div>
        <div style="text-align:right;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:0.95rem;
                        font-weight:600; color:#f0f6fc;">Abdullah Rashid</div>
            <div style="font-size:0.78rem; color:#6e7681; margin-top:5px; line-height:1.8;">
                📧 <a href="mailto:abdullahrashid130704@outlook.com"
                      style="color:#3b82f6; text-decoration:none;">abdullahrashid130704@outlook.com</a><br>
                📞 +49 15 510 337 507<br>
                🔗 <a href="https://linkedin.com/in/abdullahr2004" target="_blank"
                      style="color:#3b82f6; text-decoration:none;">linkedin.com/in/abdullahr2004</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 1 — LIVE DEMO
# ============================================================
if page == "🏠 Live Demo":

    # Hero with typing animation
    st.markdown("""
    <div class="hero">
        <div style="display:flex; justify-content:center; margin-bottom:20px;">
            <div class="badge"><span class="dot"></span> LIVE INFERENCE ENGINE ACTIVE</div>
        </div>
        <div style="text-align:center;">
            <div class="typing-title">Detect Defects in Milliseconds</div>
        </div>
        <p style="font-size:1.05rem; color:#8b949e; line-height:1.7; margin-top:16px;
                  max-width:580px; margin-left:auto; margin-right:auto;">
            Upload any industrial product image and get an instant AI-powered
            quality assessment with visual Grad-CAM explanations.
        </p>
        <div class="stats-row">
            <div class="stat-box" style="animation-delay:0.1s">
                <div class="stat-v">90.66%</div><div class="stat-l">Accuracy</div>
            </div>
            <div class="stat-box" style="animation-delay:0.2s">
                <div class="stat-v">0.9544</div><div class="stat-l">ROC-AUC</div>
            </div>
            <div class="stat-box" style="animation-delay:0.3s">
                <div class="stat-v">4.4ms</div><div class="stat-l">Inference</div>
            </div>
            <div class="stat-box" style="animation-delay:0.4s">
                <div class="stat-v">15</div><div class="stat-l">Categories</div>
            </div>
            <div class="stat-box" style="animation-delay:0.5s">
                <div class="stat-v">5K+</div><div class="stat-l">Images Trained</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("Model could not be loaded. Please refresh the page.")
        st.stop()

    # ---- How it works steps ----
    st.markdown("""
    <div style="display:flex; gap:12px; margin-bottom:28px; flex-wrap:wrap;">
        <div style="flex:1; min-width:180px; background:#0d1117; border:1px solid #1e2d3d;
                    border-radius:12px; padding:16px; display:flex; gap:12px; align-items:flex-start;">
            <div style="width:28px;height:28px;background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.3);
                        border-radius:50%;display:flex;align-items:center;justify-content:center;
                        font-size:0.8rem;font-weight:700;color:#3b82f6;flex-shrink:0;">1</div>
            <div>
                <div style="color:#f0f6fc;font-size:0.85rem;font-weight:600;">Upload or Select</div>
                <div style="color:#6e7681;font-size:0.76rem;margin-top:2px;">Upload your image or pick a sample below</div>
            </div>
        </div>
        <div style="flex:1; min-width:180px; background:#0d1117; border:1px solid #1e2d3d;
                    border-radius:12px; padding:16px; display:flex; gap:12px; align-items:flex-start;">
            <div style="width:28px;height:28px;background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.3);
                        border-radius:50%;display:flex;align-items:center;justify-content:center;
                        font-size:0.8rem;font-weight:700;color:#3b82f6;flex-shrink:0;">2</div>
            <div>
                <div style="color:#f0f6fc;font-size:0.85rem;font-weight:600;">Click Analyze</div>
                <div style="color:#6e7681;font-size:0.76rem;margin-top:2px;">AI processes the image in milliseconds</div>
            </div>
        </div>
        <div style="flex:1; min-width:180px; background:#0d1117; border:1px solid #1e2d3d;
                    border-radius:12px; padding:16px; display:flex; gap:12px; align-items:flex-start;">
            <div style="width:28px;height:28px;background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.3);
                        border-radius:50%;display:flex;align-items:center;justify-content:center;
                        font-size:0.8rem;font-weight:700;color:#3b82f6;flex-shrink:0;">3</div>
            <div>
                <div style="color:#f0f6fc;font-size:0.85rem;font-weight:600;">View Results</div>
                <div style="color:#6e7681;font-size:0.76rem;margin-top:2px;">See prediction, confidence, and Grad-CAM</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Upload + Preview ----
    up_col, prev_col = st.columns([1,1], gap="large")

    with up_col:
        st.markdown("<div class='clabel'>STEP 1 — UPLOAD YOUR IMAGE</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop image here",
            type=["jpg","jpeg","png","bmp"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            st.session_state.selected_sample_img  = None
            st.session_state.selected_sample_name = None
            st.session_state.analysis_done = False

        st.markdown("""
        <div class="infobox" style="margin-top:12px;">
            <strong style="color:#3b82f6;">Works with any industrial product.</strong>
            Covers all 15 MVTec AD categories: bottles, cables, capsules, carpet,
            grid, hazelnut, leather, metal nuts, pills, screws, tiles,
            toothbrushes, transistors, wood, and zippers.
        </div>
        """, unsafe_allow_html=True)

    with prev_col:
        st.markdown("<div class='clabel'>IMAGE PREVIEW</div>", unsafe_allow_html=True)
        if uploaded_file:
            st.image(Image.open(uploaded_file), use_column_width=True)
        elif st.session_state.selected_sample_img is not None:
            st.image(st.session_state.selected_sample_img, use_column_width=True)
            st.markdown(f"""
            <div style="text-align:center; color:#60a5fa; font-size:0.78rem; margin-top:6px;
                        font-family:'JetBrains Mono',monospace;">
                ✓ Loaded: {st.session_state.selected_sample_name}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#0d1117; border:2px dashed #1e2d3d; border-radius:12px;
                        padding:70px 20px; text-align:center; color:#484f58; min-height:220px;">
                <div style="font-size:2.5rem;">🖼️</div>
                <div style="font-size:0.82rem; margin-top:10px;">
                    Upload an image or select a sample below
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Sample Images ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='clabel'>OR TRY A SAMPLE IMAGE — SELECT CATEGORY AND CLICK TO LOAD</div>",
                unsafe_allow_html=True)

    sample_cat_display = st.selectbox(
        "Choose category",
        [c.replace('_',' ').title() for c in CATEGORIES],
        label_visibility="collapsed",
        key="sample_cat_select"
    )
    cat_key = sample_cat_display.lower().replace(' ','_')
    files   = SAMPLE_FILES.get(cat_key, ['normal_01.png', None, None])
    labels  = ['✅ Normal', '⚠️ Anomaly Type 1', '⚠️ Anomaly Type 2']
    border_classes = ['normal', 'anomaly', 'anomaly']
    colors  = ['#10b981', '#ef4444', '#ef4444']

    s_cols = st.columns(3, gap="medium")
    for i, (fn, lbl, bc, clr) in enumerate(zip(files, labels, border_classes, colors)):
        with s_cols[i]:
            if fn:
                img = fetch_sample(cat_key, fn)
                if img:
                    st.image(img, use_column_width=True)
                    st.markdown(f"""
                    <div style="text-align:center; color:{clr}; font-size:0.75rem;
                                font-weight:600; margin:5px 0 6px;">{lbl}</div>
                    """, unsafe_allow_html=True)
                    if st.button(f"Load this image", key=f"use_{i}_{cat_key}"):
                        st.session_state.selected_sample_img  = img
                        st.session_state.selected_sample_name = f"{sample_cat_display} — {lbl}"
                        st.session_state.analysis_done = False
                        st.toast(f"✅ Loaded: {sample_cat_display} {lbl}", icon="🖼️")
                        st.rerun()
            else:
                st.markdown("""
                <div style="background:#0d1117; border:1px dashed #1e2d3d; border-radius:10px;
                            padding:50px 10px; text-align:center; color:#484f58; font-size:0.76rem;">
                    Only 1 anomaly type for this category
                </div>
                """, unsafe_allow_html=True)

    # ---- Determine image ----
    analyze_img   = None
    analyze_label = "Unknown"
    if uploaded_file is not None:
        analyze_img   = Image.open(uploaded_file).convert("RGB")
        analyze_label = uploaded_file.name
    elif st.session_state.selected_sample_img is not None:
        analyze_img   = st.session_state.selected_sample_img
        analyze_label = st.session_state.selected_sample_name

    # ---- Analyze button ----
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns([1,1,1])
    with b2:
        go = st.button("🔍  Analyze Image", disabled=(analyze_img is None), use_container_width=True)

    # ---- Run analysis ----
    if go and analyze_img:
        with st.spinner("🧠 Running AI analysis..."):
            pred, conf, p_norm, p_anom, orig_np, overlay_np = run_prediction(model, analyze_img)
        st.session_state.analysis_done    = True
        st.session_state.analysis_results = (pred, conf, p_norm, p_anom, orig_np, overlay_np, analyze_label)

    # ---- Show results ----
    if st.session_state.analysis_done and st.session_state.analysis_results:
        pred, conf, p_norm, p_anom, orig_np, overlay_np, analyze_label = st.session_state.analysis_results

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="anim-slide">
            <div class="eyebrow">ANALYSIS COMPLETE</div>
            <div class="sec-title" style="margin-bottom:4px;">Prediction Results</div>
            <div style="color:#6e7681; font-size:0.8rem; margin-bottom:24px;
                        font-family:'JetBrains Mono',monospace;">
                {analyze_label} &nbsp;·&nbsp; Model: ConvNeXtTiny
            </div>
        </div>
        """, unsafe_allow_html=True)

        r1, r2, r3 = st.columns([1.1, 1.4, 1], gap="large")

        with r1:
            if pred == 0:
                st.markdown(f"""
                <div class="res-pass anim-scale">
                    <div class="res-icon">✅</div>
                    <div class="res-verdict" style="color:#10b981;">NORMAL</div>
                    <div class="res-sub">No defect detected</div>
                    <div class="res-conf" style="color:#10b981;">{conf*100:.1f}% confident</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="res-fail anim-scale">
                    <div class="res-icon">⚠️</div>
                    <div class="res-verdict" style="color:#ef4444;">ANOMALOUS</div>
                    <div class="res-sub">Defect detected</div>
                    <div class="res-conf" style="color:#ef4444;">{conf*100:.1f}% confident</div>
                </div>
                """, unsafe_allow_html=True)

        with r2:
            st.markdown("<div class='clabel'>CONFIDENCE SCORES</div>", unsafe_allow_html=True)
            st.markdown(make_confidence_html(p_norm, p_anom), unsafe_allow_html=True)

            dominant_color = "#10b981" if pred == 0 else "#ef4444"
            dominant_label = "NORMAL" if pred == 0 else "ANOMALOUS"
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03); border:1px solid #1e2d3d;
                        border-radius:10px; padding:12px; text-align:center; margin-top:4px;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.62rem;
                            color:#484f58; margin-bottom:4px; letter-spacing:0.1em;">FINAL VERDICT</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
                            font-weight:800; color:{dominant_color};">{dominant_label}</div>
                <div style="font-size:0.76rem; color:#6e7681; margin-top:2px;">
                    {conf*100:.1f}% confidence
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown("<div class='clabel'>MODEL STATS</div>", unsafe_allow_html=True)
            for v, l in [("ConvNeXtTiny","Architecture"),("90.66%","Test Accuracy"),
                         ("0.9544","ROC-AUC"),("4.4ms","Inference"),("15.7M","Parameters")]:
                st.markdown(f"""
                <div class="kpi" style="margin-bottom:8px; padding:10px 14px;">
                    <div style="font-family:'Space Grotesk',sans-serif; font-size:0.92rem;
                                font-weight:700; color:#3b82f6;">{v}</div>
                    <div class="kpi-l">{l}</div>
                </div>
                """, unsafe_allow_html=True)

        # ---- Interactive Grad-CAM Slider ----
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='clabel'>GRAD-CAM VISUAL EXPLANATION — DRAG SLIDER TO COMPARE</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="infobox" style="margin-bottom:16px;">
            <strong style="color:#3b82f6;">Interactive comparison:</strong>
            Use the slider below to compare the original image with the Grad-CAM heatmap overlay.
            🔴 <strong>Red/yellow</strong> = high model attention.
            🔵 <strong>Blue</strong> = low attention areas.
        </div>
        """, unsafe_allow_html=True)

        # Slider using Streamlit's built-in slider
        slider_val = st.slider(
            "← Original Image | Grad-CAM Overlay →",
            min_value=0, max_value=100, value=50,
            key="gradcam_slider"
        )

        # Blend based on slider value
        blend_ratio = slider_val / 100
        blended = (1 - blend_ratio) * orig_np + blend_ratio * overlay_np
        blended = np.clip(blended, 0, 1)

        # Show blended image
        st.image(blended, use_column_width=True,
                 caption=f"{'Original' if slider_val < 10 else 'Grad-CAM Overlay' if slider_val > 90 else f'Blend: {slider_val}% Grad-CAM'}")

        # Side by side comparison
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.image(orig_np, use_column_width=True, caption="Original Image")
        with comp_col2:
            st.image(overlay_np, use_column_width=True, caption="Full Grad-CAM Overlay")

    # Footer always shows at bottom
    st.markdown("<br>", unsafe_allow_html=True)
    footer()


# ============================================================
# PAGE 2 — PROJECT INFO
# ============================================================
elif page == "📋 Project Info":

    st.markdown("""
    <div class="hero">
        <div style="display:flex;justify-content:center;margin-bottom:20px;"><div class="badge"><span class="dot"></span> RESEARCH PROJECT · SS26</div></div>
        <div class="typing-title">Project Overview</div>
        <p style="font-size:1.05rem; color:#8b949e; line-height:1.7; margin-top:14px;
                  max-width:600px; margin-left:auto; margin-right:auto;">
            Explainable Multi-Product Industrial Anomaly Classification
            using CNN Transfer Learning and Cross-Category Evaluation
        </p>
    </div>
    """, unsafe_allow_html=True)

    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.markdown("""
        <div class="gcard">
            <div style="font-size:2rem; margin-bottom:12px;">🏭</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                        color:#f0f6fc; margin-bottom:10px; font-size:0.95rem;">
                Manual Inspection is Broken
            </div>
            <div class="sec-body" style="font-size:0.84rem;">
                Manufacturing lines require inspecting thousands of products per hour.
                Human inspection is slow, expensive, and prone to fatigue errors.
                One missed defect can cause costly recalls and safety hazards.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="gcard">
            <div style="font-size:2rem; margin-bottom:12px;">🤖</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                        color:#f0f6fc; margin-bottom:10px; font-size:0.95rem;">
                AI as the Solution
            </div>
            <div class="sec-body" style="font-size:0.84rem;">
                Deep learning inspects images in milliseconds with consistent accuracy.
                This project builds, compares, and explains three CNN architectures
                to find the best balance of accuracy, speed, and trustworthiness.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Dataset
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">DATASET</div>
    <div class="sec-title">MVTec Anomaly Detection</div>
    <div class="sec-body" style="margin-bottom:20px;">
        The gold standard benchmark for industrial anomaly detection.
    </div>
    """, unsafe_allow_html=True)

    dcols = st.columns(4)
    for col, v, l in zip(dcols,
        ["5,354","15","73","2"],
        ["Total Images","Categories","Defect Types","Classes"]):
        with col:
            st.markdown(f"""<div class="kpi"><div class="kpi-v">{v}</div><div class="kpi-l">{l}</div></div>""",
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="gcard">
        <div class="clabel">ALL 15 PRODUCT CATEGORIES</div>
        <span class="cpill">Bottle</span><span class="cpill">Cable</span>
        <span class="cpill">Capsule</span><span class="cpill">Carpet</span>
        <span class="cpill">Grid</span><span class="cpill">Hazelnut</span>
        <span class="cpill">Leather</span><span class="cpill">Metal Nut</span>
        <span class="cpill">Pill</span><span class="cpill">Screw</span>
        <span class="cpill">Tile</span><span class="cpill">Toothbrush</span>
        <span class="cpill">Transistor</span><span class="cpill">Wood</span>
        <span class="cpill">Zipper</span>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">METHODOLOGY</div>
    <div class="sec-title">Research Pipeline</div>
    <div class="sec-body" style="margin-bottom:20px;">A systematic 6-stage pipeline from raw data to deployed AI.</div>
    """, unsafe_allow_html=True)

    pcols = st.columns(6)
    for col, (num, name, desc, clr) in zip(pcols, [
        ("1","Data Loading","15 categories · 5K+ images","#3b82f6"),
        ("2","Preprocessing","Resize · Normalize · Augment","#3b82f6"),
        ("3","Training","3 CNN models · Early stopping","#3b82f6"),
        ("4","Evaluation","F1 · AUC · Confusion Matrix","#3b82f6"),
        ("5","Explainability","Grad-CAM · SHAP analysis","#3b82f6"),
        ("6","Deployment","Streamlit · Web App","#10b981"),
    ]):
        with col:
            st.markdown(f"""
            <div class="pstep">
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.3rem;
                            font-weight:800; color:{clr};">{num}</div>
                <div style="font-size:0.78rem; font-weight:600; color:#f0f6fc; margin-top:6px;">{name}</div>
                <div style="font-size:0.68rem; color:#6e7681; margin-top:4px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Models
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">MODELS</div>
    <div class="sec-title">Three Architectures Compared</div>
    <div class="sec-body" style="margin-bottom:20px;">From a simple baseline to state-of-the-art transfer learning.</div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3, gap="large")
    for col, icon, name, tag, tagcol, desc, params, acc, acc_col in [
        (m1,"🏗️","Custom CNN","BASELINE","#484f58",
         "4 convolutional blocks built from scratch. No pretrained weights. Serves as our performance baseline.",
         "422,530 params · 1.63 MB","63.77%","#8b949e"),
        (m2,"⚡","EfficientNetV2S","TRANSFER LEARNING","#3b82f6",
         "Pretrained on ImageNet. Two-stage training: frozen backbone, then partial fine-tuning.",
         "15.2M params · 79.1 MB","86.93%","#3b82f6"),
        (m3,"🏆","ConvNeXtTiny","BEST MODEL ★","#10b981",
         "Modern ConvNet pretrained on ImageNet. Outperformed all models across every metric.",
         "15.7M params · 106.95 MB","90.66%","#10b981"),
    ]:
        with col:
            cls = "best" if "BEST" in tag else "mid" if "TRANSFER" in tag else ""
            st.markdown(f"""
            <div class="mcard {cls}">
                <div style="font-size:2rem;">{icon}</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1rem;
                            font-weight:700; color:#f0f6fc; margin:10px 0 4px;">{name}</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem;
                            color:{tagcol}; letter-spacing:0.1em; margin-bottom:12px;">{tag}</div>
                <div class="sec-body" style="font-size:0.82rem;">{desc}</div>
                <div style="margin-top:16px; padding-top:14px; border-top:1px solid #1e2d3d;">
                    <div style="color:#6e7681; font-size:0.72rem;">{params}</div>
                    <div style="font-family:'Space Grotesk',sans-serif; font-size:1.3rem;
                                font-weight:800; color:{acc_col}; margin-top:4px;">{acc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Tech
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">TECHNOLOGY</div>
    <div class="sec-title">Tech Stack</div>
    <div class="gcard" style="margin-top:12px;">
        <span class="tbadge">Python 3.10</span><span class="tbadge">PyTorch</span>
        <span class="tbadge">Torchvision</span><span class="tbadge">ConvNeXtTiny</span>
        <span class="tbadge">EfficientNetV2S</span><span class="tbadge">Grad-CAM</span>
        <span class="tbadge">SHAP</span><span class="tbadge">OpenCV</span>
        <span class="tbadge">Scikit-learn</span><span class="tbadge">Matplotlib</span>
        <span class="tbadge">Plotly</span><span class="tbadge">Streamlit</span>
        <span class="tbadge">Kaggle GPU T4×2</span><span class="tbadge">Hugging Face</span>
    </div>
    """, unsafe_allow_html=True)

    footer()


# ============================================================
# PAGE 3 — RESULTS
# ============================================================
elif page == "📊 Results":

    st.markdown("""
    <div class="hero">
        <div style="display:flex;justify-content:center;margin-bottom:20px;"><div class="badge"><span class="dot"></span> EXPERIMENTAL RESULTS</div></div>
        <div class="typing-title">Model Performance</div>
        <p style="font-size:1.05rem; color:#8b949e; line-height:1.7; margin-top:14px;
                  max-width:560px; margin-left:auto; margin-right:auto;">
            Complete evaluation of all 3 models on the held-out test set.
            ConvNeXtTiny achieved best results across every metric.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    kcols = st.columns(5)
    for col, v, l in zip(kcols,
        ["90.66%","0.9544","0.8706","0.8916","4.4ms"],
        ["Best Accuracy","Best ROC-AUC","Best Macro F1","Best PR-AUC","Fastest Inference"]):
        with col:
            st.markdown(f"""<div class="kpi"><div class="kpi-v">{v}</div><div class="kpi-l">{l}</div></div>""",
                        unsafe_allow_html=True)

    # Interactive Plotly table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">COMPARISON TABLE</div>
    <div class="sec-title">Full Metrics Breakdown</div>
    """, unsafe_allow_html=True)

    df = pd.DataFrame({
        "Model":           ["Custom CNN","EfficientNetV2S","ConvNeXtTiny ⭐"],
        "Accuracy":        [0.6377,0.8693,0.9066],
        "Macro F1":        [0.5499,0.8142,0.8706],
        "ROC-AUC":         [0.5775,0.9025,0.9544],
        "PR-AUC":          [0.2821,0.8159,0.8916],
        "Inference (ms)":  [0.790,3.154,4.401],
        "Params (M)":      [0.42,15.22,15.67],
        "Size (MB)":       [1.63,79.10,106.95],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ---- Metric selector bar chart (matplotlib) ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">INTERACTIVE CHART</div>
    <div class="sec-title">Select Metric to Compare</div>
    """, unsafe_allow_html=True)

    metric = st.selectbox(
        "Choose metric",
        ["Accuracy","Macro F1","ROC-AUC","PR-AUC","Inference (ms)","Params (M)","Size (MB)"],
        label_visibility="collapsed"
    )

    models_list   = ["Custom CNN", "EfficientNetV2S", "ConvNeXtTiny"]
    metric_values = {
        "Accuracy":        [0.6377, 0.8693, 0.9066],
        "Macro F1":        [0.5499, 0.8142, 0.8706],
        "ROC-AUC":         [0.5775, 0.9025, 0.9544],
        "PR-AUC":          [0.2821, 0.8159, 0.8916],
        "Inference (ms)":  [0.790,  3.154,  4.401],
        "Params (M)":      [0.42,   15.22,  15.67],
        "Size (MB)":       [1.63,   79.10,  106.95],
    }
    bar_model_colors = ["#6e7681", "#3b82f6", "#10b981"]
    vals = metric_values[metric]

    fig_bar, ax_bar = plt.subplots(figsize=(8, 3.5))
    fig_bar.patch.set_facecolor('#0d1117')
    ax_bar.set_facecolor('#0d1117')
    bars = ax_bar.bar(models_list, vals, color=bar_model_colors,
                      edgecolor='none', width=0.5)
    ax_bar.tick_params(colors='#8b949e', labelsize=10)
    for sp in ax_bar.spines.values(): sp.set_color('#1e2d3d')
    ax_bar.set_ylabel(metric, color='#8b949e', fontsize=9)
    ax_bar.yaxis.set_tick_params(labelcolor='#8b949e')
    for bar, val in zip(bars, vals):
        ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                    f'{val:.4f}', ha='center', fontsize=9, color='#8b949e')
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close()

    # ---- Radar chart (matplotlib) ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">RADAR CHART</div>
    <div class="sec-title">Multi-Metric Comparison</div>
    <div class="sec-body" style="margin-bottom:16px;">
        A spider chart comparing all 3 models across key performance dimensions.
    </div>
    """, unsafe_allow_html=True)

    radar_labels  = ['Accuracy', 'Macro F1', 'ROC-AUC', 'PR-AUC', 'Speed']
    radar_data    = {
        'Custom CNN':      [0.6377, 0.5499, 0.5775, 0.2821, 1.00],
        'EfficientNetV2S': [0.8693, 0.8142, 0.9025, 0.8159, 0.25],
        'ConvNeXtTiny':    [0.9066, 0.8706, 0.9544, 0.8916, 0.18],
    }
    radar_colors_list = ['#6e7681', '#3b82f6', '#10b981']
    N = len(radar_labels)
    angles = [n/float(N)*2*np.pi for n in range(N)]
    angles += angles[:1]

    fig_r, ax_r = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
    fig_r.patch.set_facecolor('#0d1117')
    ax_r.set_facecolor('#0d1117')
    ax_r.spines['polar'].set_color('#1e2d3d')
    ax_r.grid(color='#1e2d3d', linewidth=0.8)

    for (model_n, scores), clr in zip(radar_data.items(), radar_colors_list):
        vals_r = scores + scores[:1]
        ax_r.plot(angles, vals_r, color=clr, linewidth=2, label=model_n)
        ax_r.fill(angles, vals_r, color=clr, alpha=0.08)

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(radar_labels, color='#8b949e', fontsize=9)
    ax_r.set_ylim(0, 1)
    ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_r.set_yticklabels(['0.25','0.5','0.75','1.0'], color='#6e7681', fontsize=7)
    ax_r.tick_params(colors='#8b949e')
    legend = ax_r.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1),
                         facecolor='#0d1117', edgecolor='#1e2d3d',
                         labelcolor='#8b949e', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_r)
    plt.close()

    # ---- Per-category bar chart (matplotlib) ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">CATEGORY ANALYSIS</div>
    <div class="sec-title">Per-Category Accuracy — ConvNeXtTiny</div>
    <div class="sec-body" style="margin-bottom:16px;">
        🟢 above 90% &nbsp;·&nbsp; 🔵 80–90% &nbsp;·&nbsp; 🔴 below 80%
    </div>
    """, unsafe_allow_html=True)

    cat_names = ["Carpet","Leather","Bottle","Hazelnut","Tile","Zipper",
                 "Grid","Wood","Pill","Capsule","Metal Nut","Cable",
                 "Transistor","Screw","Toothbrush"]
    cat_accs  = [1.00,0.98,0.95,0.96,0.94,0.95,
                 0.93,0.92,0.90,0.91,0.89,0.88,
                 0.87,0.82,0.70]
    cat_colors = ['#10b981' if v>=0.90 else '#3b82f6' if v>=0.80 else '#ef4444'
                  for v in cat_accs]

    fig_c, ax_c = plt.subplots(figsize=(13, 4))
    fig_c.patch.set_facecolor('#0d1117')
    ax_c.set_facecolor('#0d1117')
    bars_c = ax_c.bar(cat_names, cat_accs, color=cat_colors, edgecolor='none', width=0.6)
    ax_c.axhline(y=np.mean(cat_accs), color='#f59e0b', linestyle='--', linewidth=1.5,
                 label=f'Mean: {np.mean(cat_accs):.2f}')
    ax_c.set_ylim(0, 1.13)
    ax_c.tick_params(axis='x', rotation=30, labelsize=9, colors='#8b949e')
    ax_c.tick_params(axis='y', labelsize=9, colors='#8b949e')
    for sp in ax_c.spines.values(): sp.set_color('#1e2d3d')
    ax_c.set_ylabel('Accuracy', color='#8b949e', fontsize=9)
    for bar, val in zip(bars_c, cat_accs):
        ax_c.text(bar.get_x()+bar.get_width()/2, val+0.012,
                  f'{val:.0%}', ha='center', fontsize=8, color='#8b949e')
    ax_c.legend(facecolor='#0d1117', edgecolor='#1e2d3d', labelcolor='#f59e0b', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_c)
    plt.close()

    # RQs
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">KEY FINDINGS</div>
    <div class="sec-title">Research Questions Answered</div>
    """, unsafe_allow_html=True)

    for rq, q, a in [
        ("RQ1","Which architecture achieves strongest performance?",
         "ConvNeXtTiny — 90.66% accuracy, 0.9544 ROC-AUC, best across all metrics."),
        ("RQ2","How much does performance vary between categories?",
         "30% range — Carpet achieved 100%, Toothbrush was hardest at 70%."),
        ("RQ3","Do Grad-CAM and SHAP overlap with anomalous regions?",
         "Yes — correct predictions consistently highlight the actual defect areas."),
        ("RQ4","Which categories produce highest false-negative rates?",
         "Toothbrush and Screw — fine-grained texture defects are hardest to detect."),
        ("RQ5","Can a compact model give a practical trade-off?",
         "Custom CNN at 0.790ms/image is 5.6× faster but 27% less accurate than ConvNeXtTiny."),
    ]:
        st.markdown(f"""
        <div class="gcard" style="padding:16px 20px; margin-bottom:8px;">
            <div style="display:flex; gap:14px; align-items:flex-start;">
                <div style="background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.25);
                            color:#3b82f6; padding:4px 12px; border-radius:8px;
                            font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                            font-weight:600; white-space:nowrap;">{rq}</div>
                <div>
                    <div style="color:#f0f6fc; font-size:0.86rem; font-weight:500;">{q}</div>
                    <div style="color:#10b981; font-size:0.82rem; margin-top:5px;">→ {a}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    footer()


# ============================================================
# PAGE 4 — EXPLAINABILITY
# ============================================================
elif page == "🔬 Explainability":

    st.markdown("""
    <div class="hero">
        <div style="display:flex;justify-content:center;margin-bottom:20px;"><div class="badge"><span class="dot"></span> EXPLAINABLE AI · XAI</div></div>
        <div class="typing-title">Why Trust the Model?</div>
        <p style="font-size:1.05rem; color:#8b949e; line-height:1.7; margin-top:14px;
                  max-width:560px; margin-left:auto; margin-right:auto;">
            High accuracy alone is not enough. We use Grad-CAM and SHAP
            to verify the model is looking at the right things.
        </p>
    </div>
    """, unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    for col, icon, title, body in zip([g1,g2,g3],
        ["🎯","❓","⚠️"],
        ["Correct Predictions","Wrong Predictions","Limitations"],
        [
            "For correctly classified anomalous images, Grad-CAM highlights the actual defect — scratches, cracks, or contamination visible on the product surface.",
            "For misclassified images, Grad-CAM reveals the model focused on background or irrelevant areas, explaining why the prediction failed.",
            "Grad-CAM produces coarse spatial maps and cannot pinpoint exact pixel-level regions. Always combine with SHAP for a complete picture."
        ]):
        with col:
            st.markdown(f"""
            <div class="gcard" style="text-align:center; padding:22px 18px;">
                <div style="font-size:1.8rem; margin-bottom:10px;">{icon}</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                            color:#f0f6fc; font-size:0.9rem; margin-bottom:8px;">{title}</div>
                <div class="sec-body" style="font-size:0.8rem;">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">SHAP</div>
    <div class="sec-title">SHapley Additive exPlanations</div>
    <div class="sec-body" style="margin-bottom:20px;">
        SHAP assigns each pixel a value showing how much it pushed the prediction
        toward Anomalous (red) or toward Normal (blue).
    </div>
    """, unsafe_allow_html=True)

    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.markdown("""
        <div class="gcard">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <div style="width:14px; height:14px; background:#ef4444; border-radius:3px;"></div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                            color:#f0f6fc;">Red — Supports Anomalous</div>
            </div>
            <div class="sec-body" style="font-size:0.82rem;">
                Red pixels push the model toward Anomalous classification. These
                correspond to actual defect areas — scratches, cracks, contamination.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="gcard">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <div style="width:14px; height:14px; background:#3b82f6; border-radius:3px;"></div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                            color:#f0f6fc;">Blue — Supports Normal</div>
            </div>
            <div class="sec-body" style="font-size:0.82rem;">
                Blue pixels work against the anomalous classification — these are
                clean regions the model recognizes as defect-free.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">IMPORTANCE</div>
    <div class="sec-title">Why Explainability Matters</div>
    <div class="gcard" style="margin-top:12px;">
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:24px;">
            <div>
                <div style="font-size:1.5rem; margin-bottom:8px;">🛡️</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                            color:#f0f6fc; font-size:0.88rem; margin-bottom:6px;">Trust</div>
                <div class="sec-body" style="font-size:0.8rem;">
                    Factory operators need to trust the system before replacing
                    human inspectors. Visual explanations build that trust.
                </div>
            </div>
            <div>
                <div style="font-size:1.5rem; margin-bottom:8px;">🐛</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                            color:#f0f6fc; font-size:0.88rem; margin-bottom:6px;">Debugging</div>
                <div class="sec-body" style="font-size:0.8rem;">
                    When the model fails, explanations show whether it was looking
                    at background patterns instead of actual product features.
                </div>
            </div>
            <div>
                <div style="font-size:1.5rem; margin-bottom:8px;">📋</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                            color:#f0f6fc; font-size:0.88rem; margin-bottom:6px;">Compliance</div>
                <div class="sec-body" style="font-size:0.8rem;">
                    In regulated industries, AI decisions must be auditable.
                    Grad-CAM and SHAP provide the required decision trail.
                </div>
            </div>
        </div>
    </div>
    <div class="infobox" style="margin-top:14px; border-color:rgba(245,158,11,0.25);
                                background:rgba(245,158,11,0.05);">
        <strong style="color:#f59e0b;">⚠️ Important:</strong>
        Attractive heatmaps are not proof of correct reasoning. Always combine
        explainability tools with rigorous quantitative evaluation metrics.
    </div>
    """, unsafe_allow_html=True)

    footer()


# ============================================================
# PAGE 5 — ABOUT
# ============================================================
elif page == "👤 About":

    st.markdown("""
    <div class="hero">
        <div style="display:flex;justify-content:center;margin-bottom:20px;"><div class="badge"><span class="dot"></span> ABOUT</div></div>
        <div class="typing-title">Abdullah Rashid</div>
        <p style="font-size:1.05rem; color:#8b949e; line-height:1.7; margin-top:14px;">
            3rd Semester · Bachelor of Software Engineering<br>
            University of Europe for Applied Sciences, Potsdam · SS26
        </p>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns([1,1], gap="large")

    with a1:
        st.markdown("""
        <div class="eyebrow">CONTACT</div>
        <div class="sec-title" style="margin-bottom:12px;">Get in Touch</div>
        <div class="gcard">
            <div style="display:flex; flex-direction:column; gap:18px;">
        """, unsafe_allow_html=True)

        for icon, label, val, href in [
            ("📧","EMAIL","abdullahrashid130704@outlook.com","mailto:abdullahrashid130704@outlook.com"),
            ("📞","PHONE","+49 15 510 337 507","tel:+4915510337507"),
            ("🔗","LINKEDIN","linkedin.com/in/abdullahr2004","https://linkedin.com/in/abdullahr2004"),
        ]:
            st.markdown(f"""
            <a href="{href}" target="_blank" style="text-decoration:none; display:flex; align-items:center; gap:14px;">
                <div style="width:40px; height:40px; min-width:40px; background:rgba(59,130,246,0.1);
                            border:1px solid rgba(59,130,246,0.2); border-radius:10px;
                            display:flex; align-items:center; justify-content:center; font-size:1.1rem;">{icon}</div>
                <div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.62rem;
                                color:#484f58; letter-spacing:0.1em; margin-bottom:2px;">{label}</div>
                    <div style="color:#3b82f6; font-size:0.85rem;">{val}</div>
                </div>
            </a>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="eyebrow">PROJECT LINKS</div>
        <div class="sec-title" style="margin-bottom:12px;">Resources</div>
        <div class="gcard">
        """, unsafe_allow_html=True)

        for icon, title, sub, url in [
            ("💻","GitHub Repository","Source code, notebook, all figures",
             "https://github.com/abdullah1307-git/mvtec-anomaly-classification"),
            ("📓","Kaggle Notebook","Full executed training notebook with results",
             "https://www.kaggle.com/code/abdullah130704/notebook8b312b2661"),
            ("📊","MVTec AD Dataset","15 categories · 5,354 images · Kaggle",
             "https://www.kaggle.com/datasets/ipythonx/mvtec-ad"),
            ("🤗","Hugging Face Model","ConvNeXtTiny weights · 112 MB",
             "https://huggingface.co/abdullah130704/mvtec-anomaly-model"),
        ]:
            st.markdown(f"""
            <a href="{url}" target="_blank" style="text-decoration:none; display:block; margin-bottom:8px;">
                <div style="display:flex; align-items:center; gap:14px; padding:12px 14px;
                            background:rgba(255,255,255,0.03); border:1px solid #1e2d3d;
                            border-radius:10px; transition:all 0.2s;">
                    <div style="font-size:1.3rem;">{icon}</div>
                    <div style="flex:1;">
                        <div style="color:#f0f6fc; font-size:0.86rem; font-weight:500;">{title}</div>
                        <div style="color:#6e7681; font-size:0.73rem; margin-top:2px;">{sub}</div>
                    </div>
                    <div style="color:#484f58; font-size:0.9rem;">↗</div>
                </div>
            </a>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">ACADEMIC CONTEXT</div>
    <div class="sec-title" style="margin-bottom:12px;">Course Information</div>
    <div class="gcard">
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:20px;">
    """, unsafe_allow_html=True)

    for label, val in [
        ("COURSE","Machine Learning & Smart Systems"),
        ("UNIVERSITY","UE for Applied Sciences, Potsdam"),
        ("SEMESTER","Summer Semester 2026 · 3rd"),
        ("PROGRAMME","Bachelor of Software Engineering"),
    ]:
        st.markdown(f"""
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.62rem;
                            color:#484f58; letter-spacing:0.1em; margin-bottom:6px;">{label}</div>
                <div style="color:#f0f6fc; font-size:0.84rem; font-weight:500;">{val}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    footer()
