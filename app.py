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
import os
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
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS — Full Design System
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---- Reset & Base ---- */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

/* ---- App Background ---- */
.stApp {
    background: #080c14;
    background-image:
        radial-gradient(ellipse at 20% 20%, rgba(59,130,246,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(139,92,246,0.04) 0%, transparent 50%);
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2d3d !important;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] .stRadio label { 
    padding: 10px 14px !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}

/* ---- Typography ---- */
.display-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.8rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #93c5fd 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.display-sub {
    font-size: 1.15rem;
    color: #8b949e;
    line-height: 1.7;
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto;
}

.section-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 8px;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f0f6fc;
    margin-bottom: 6px;
    letter-spacing: -0.02em;
}

.section-body {
    color: #8b949e;
    font-size: 0.92rem;
    line-height: 1.7;
}

/* ---- Hero ---- */
.hero-wrap {
    background: linear-gradient(180deg, #0d1117 0%, #080c14 100%);
    border: 1px solid #1e2d3d;
    border-radius: 20px;
    padding: 64px 48px;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 32px;
    animation: fadeInUp 0.6s ease both;
}

.hero-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.5), transparent);
}

.hero-wrap::after {
    content: '';
    position: absolute;
    top: -200px; left: 50%;
    transform: translateX(-50%);
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
    pointer-events: none;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.25);
    color: #60a5fa;
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 24px;
    font-family: 'JetBrains Mono', monospace;
}

/* ---- Stat Pills ---- */
.stat-row {
    display: flex;
    justify-content: center;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 32px;
}

.stat-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 14px 24px;
    text-align: center;
    min-width: 120px;
    transition: border-color 0.2s, transform 0.2s;
}

.stat-pill:hover {
    border-color: rgba(59,130,246,0.4);
    transform: translateY(-2px);
}

.stat-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #3b82f6;
    line-height: 1;
}

.stat-label {
    font-size: 0.72rem;
    color: #6e7681;
    margin-top: 4px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---- Cards ---- */
.glass-card {
    background: rgba(13,17,23,0.8);
    border: 1px solid #1e2d3d;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
    transition: border-color 0.2s, transform 0.2s;
}

.glass-card:hover { border-color: #2d4a6e; }

.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    color: #484f58;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 16px;
}

/* ---- Result Cards ---- */
.result-pass {
    background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(16,185,129,0.03) 100%);
    border: 1.5px solid rgba(16,185,129,0.4);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    animation: pulseGreen 2s ease infinite;
}

.result-fail {
    background: linear-gradient(135deg, rgba(239,68,68,0.08) 0%, rgba(239,68,68,0.03) 100%);
    border: 1.5px solid rgba(239,68,68,0.4);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    animation: pulseRed 2s ease infinite;
}

.result-verdict {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 8px 0 4px;
}

.result-sub { font-size: 0.85rem; color: #8b949e; }

/* ---- Metric Cards ---- */
.kpi-card {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s, transform 0.15s;
}

.kpi-card:hover {
    border-color: rgba(59,130,246,0.35);
    transform: translateY(-2px);
}

.kpi-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #3b82f6;
    line-height: 1;
}

.kpi-lbl {
    font-size: 0.72rem;
    color: #6e7681;
    margin-top: 6px;
    font-weight: 500;
    letter-spacing: 0.04em;
}

/* ---- Info Box ---- */
.info-pill {
    background: rgba(59,130,246,0.07);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 0.84rem;
    color: #8b949e;
    line-height: 1.6;
}

/* ---- Model Cards ---- */
.model-card {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 14px;
    padding: 24px;
    height: 100%;
    transition: border-color 0.2s, transform 0.2s;
}

.model-card:hover {
    transform: translateY(-3px);
}

.model-card.best { border-color: rgba(16,185,129,0.35); }
.model-card.mid  { border-color: rgba(59,130,246,0.25); }
.model-card.base { border-color: #1e2d3d; }

/* ---- Pipeline Steps ---- */
.pipeline-step {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    position: relative;
    transition: border-color 0.2s, transform 0.2s;
}

.pipeline-step:hover {
    border-color: rgba(59,130,246,0.3);
    transform: translateY(-2px);
}

.step-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #3b82f6;
    line-height: 1;
}

.step-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: #f0f6fc;
    margin-top: 6px;
}

.step-desc {
    font-size: 0.72rem;
    color: #6e7681;
    margin-top: 3px;
}

/* ---- Category Pills ---- */
.cat-pill {
    display: inline-block;
    background: rgba(139,92,246,0.08);
    border: 1px solid rgba(139,92,246,0.2);
    color: #a78bfa;
    padding: 5px 14px;
    border-radius: 100px;
    font-size: 0.76rem;
    font-weight: 500;
    margin: 3px;
    transition: background 0.2s;
}

.cat-pill:hover { background: rgba(139,92,246,0.15); }

/* ---- Footer ---- */
.site-footer {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 16px;
    padding: 32px 40px;
    margin-top: 48px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 20px;
}

.footer-brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f0f6fc;
}

.footer-tagline {
    font-size: 0.8rem;
    color: #6e7681;
    margin-top: 4px;
    line-height: 1.5;
}

.footer-contact {
    text-align: right;
}

.footer-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #f0f6fc;
}

.footer-details {
    font-size: 0.8rem;
    color: #6e7681;
    margin-top: 6px;
    line-height: 1.7;
}

.footer-link { color: #3b82f6; text-decoration: none; }
.footer-link:hover { color: #60a5fa; }

/* ---- Tech Badge ---- */
.tech-badge {
    display: inline-block;
    background: rgba(255,255,255,0.04);
    border: 1px solid #1e2d3d;
    border-radius: 8px;
    padding: 6px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #8b949e;
    margin: 4px;
    transition: border-color 0.2s;
}

.tech-badge:hover { border-color: #3b82f6; color: #60a5fa; }

/* ---- Animations ---- */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes pulseGreen {
    0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
    50%       { box-shadow: 0 0 20px 2px rgba(16,185,129,0.1); }
}

@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    50%       { box-shadow: 0 0 20px 2px rgba(239,68,68,0.1); }
}

@keyframes scanLine {
    0%   { top: 0; }
    100% { top: 100%; }
}

.animate-in { animation: fadeInUp 0.5s ease both; }

/* ---- Streamlit overrides ---- */
h1, h2, h3, h4, h5, h6 { color: #f0f6fc !important; }
p, li { color: #8b949e; }
.stProgress > div > div { background: linear-gradient(90deg, #3b82f6, #818cf8) !important; }
[data-testid="stFileUploader"] {
    background: #0d1117;
    border: 2px dashed #1e2d3d;
    border-radius: 12px;
}
.stSelectbox > div > div {
    background: #0d1117 !important;
    border: 1px solid #1e2d3d !important;
    color: #f0f6fc !important;
}
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    padding: 12px 24px;
    font-size: 0.95rem;
    transition: opacity 0.2s, transform 0.2s;
    width: 100%;
}
div[data-testid="stButton"] button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}
div[data-testid="stButton"] button:disabled {
    background: #1e2d3d;
    color: #484f58;
}
.stDataFrame { border: 1px solid #1e2d3d; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================
HF_BASE    = "https://huggingface.co/abdullah130704/mvtec-anomaly-model/resolve/main"
MODEL_URL  = f"{HF_BASE}/best_model_weights.pth"
SAMPLE_BASE = f"{HF_BASE}/sample_images"

CATEGORIES = [
    'bottle','cable','capsule','carpet','grid',
    'hazelnut','leather','metal_nut','pill','screw',
    'tile','toothbrush','transistor','wood','zipper'
]

IMG_SIZE       = 224
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]

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
        w = self.grads.mean(dim=[2,3], keepdim=True)
        cam = torch.relu((w*self.acts).sum(1,keepdim=True)).squeeze().cpu().numpy()
        mn, mx = cam.min(), cam.max()
        if mx-mn > 0: cam = (cam-mn)/(mx-mn)
        return cv2.resize(cam, (IMG_SIZE,IMG_SIZE))


def gradcam_figure(pil_img, heatmap):
    img = np.array(pil_img.resize((IMG_SIZE,IMG_SIZE))).astype(np.float32)/255
    h   = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    h   = cv2.cvtColor(h, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    ov  = np.clip(0.45*h + 0.55*img, 0, 1)
    fig, ax = plt.subplots(1,2,figsize=(10,4))
    fig.patch.set_facecolor('#0d1117')
    for a in ax: a.set_facecolor('#0d1117')
    ax[0].imshow(img);  ax[0].set_title('Original Image',       color='#f0f6fc', fontsize=11, pad=10); ax[0].axis('off')
    ax[1].imshow(ov);   ax[1].set_title('Grad-CAM — Model Focus', color='#f0f6fc', fontsize=11, pad=10); ax[1].axis('off')
    plt.tight_layout(pad=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
    buf.seek(0); plt.close()
    return buf


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
    fig    = gradcam_figure(pil_img, hm)
    return pred, conf, p_norm, p_anom, fig


@st.cache_data(show_spinner=False)
def fetch_sample(cat, fname):
    url = f"{SAMPLE_BASE}/{cat}/{fname}"
    r = requests.get(url)
    if r.status_code == 200:
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return None


# ============================================================
# SHARED COMPONENTS
# ============================================================
def footer():
    st.markdown("""
    <div class="site-footer">
        <div>
            <div class="footer-brand">🔍 AnomalyVision</div>
            <div class="footer-tagline">
                Machine Learning & Smart Systems — Final Project<br>
                University of Europe for Applied Sciences, Potsdam
            </div>
        </div>
        <div class="footer-contact">
            <div class="footer-name">Abdullah Rashid</div>
            <div class="footer-details">
                📧 <a href="mailto:abdullahrashid130704@outlook.com" class="footer-link">abdullahrashid130704@outlook.com</a><br>
                📞 +49 15 510 337 507<br>
                🔗 <a href="https://linkedin.com/in/abdullahr2004" class="footer-link" target="_blank">linkedin.com/in/abdullahr2004</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:24px 0 16px; text-align:center;">
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1.3rem; font-weight:800;
                    background:linear-gradient(135deg,#3b82f6,#818cf8);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text;">
            🔍 AnomalyVision
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                    color:#484f58; margin-top:4px; letter-spacing:0.08em;">
            INDUSTRIAL AI SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2d3d; margin:0 0 16px;'>", unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Live Demo",
        "📋  Project Info",
        "📊  Results",
        "🔬  Explainability",
        "👤  About"
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1e2d3d; margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                color:#484f58; letter-spacing:0.08em; margin-bottom:10px;">
        MODEL STATUS
    </div>
    <div style="font-size:0.8rem; color:#8b949e; line-height:2;">
        <span style="color:#10b981;">●</span> ConvNeXtTiny · Active<br>
        <span style="color:#3b82f6;">◆</span> Accuracy · 90.66%<br>
        <span style="color:#3b82f6;">◆</span> ROC-AUC · 0.9544<br>
        <span style="color:#3b82f6;">◆</span> Categories · 15<br>
        <span style="color:#3b82f6;">◆</span> Inference · 4.4ms
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2d3d; margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                color:#484f58; letter-spacing:0.08em; margin-bottom:10px;">
        TECH STACK
    </div>
    <div style="font-size:0.76rem; color:#6e7681; line-height:1.9;">
        PyTorch · ConvNeXt<br>
        Grad-CAM · SHAP<br>
        Streamlit · Python
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 1 — LIVE DEMO
# ============================================================
if page == "🏠  Live Demo":

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">● LIVE INFERENCE ENGINE ACTIVE</div>
        <div class="display-title">Detect Defects<br>in Milliseconds</div>
        <p class="display-sub" style="margin-top:16px;">
            Upload any industrial product image and get an instant AI-powered quality assessment.
            Powered by ConvNeXtTiny trained on 15 product categories.
        </p>
        <div class="stat-row">
            <div class="stat-pill"><div class="stat-value">90.66%</div><div class="stat-label">Accuracy</div></div>
            <div class="stat-pill"><div class="stat-value">0.9544</div><div class="stat-label">ROC-AUC</div></div>
            <div class="stat-pill"><div class="stat-value">4.4ms</div><div class="stat-label">Inference</div></div>
            <div class="stat-pill"><div class="stat-value">15</div><div class="stat-label">Categories</div></div>
            <div class="stat-pill"><div class="stat-value">5K+</div><div class="stat-label">Images Trained</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("Model could not be loaded. Please refresh the page.")
        st.stop()

    # Upload section
    up_col, prev_col = st.columns([1,1], gap="large")

    with up_col:
        st.markdown("<div class='card-label'>UPLOAD YOUR IMAGE</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop image here",
            type=["jpg","jpeg","png","bmp"],
            label_visibility="collapsed"
        )
        st.markdown("""
        <div class="info-pill" style="margin-top:14px;">
            <strong style="color:#3b82f6;">Works with any industrial product.</strong>
            The model covers bottles, cables, capsules, carpet, grid, hazelnut,
            leather, metal nuts, pills, screws, tiles, toothbrushes, transistors,
            wood, and zippers. Upload any image to get started.
        </div>
        """, unsafe_allow_html=True)

    with prev_col:
        st.markdown("<div class='card-label'>PREVIEW</div>", unsafe_allow_html=True)
        if uploaded_file:
            st.image(Image.open(uploaded_file), use_column_width=True)
        else:
            st.markdown("""
            <div style="background:#0d1117; border:2px dashed #1e2d3d; border-radius:12px;
                        padding:60px 20px; text-align:center; color:#484f58;">
                <div style="font-size:2.5rem;">🖼️</div>
                <div style="font-size:0.85rem; margin-top:10px;">Image preview will appear here</div>
            </div>
            """, unsafe_allow_html=True)

    # Sample images
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='card-label'>OR TRY A SAMPLE IMAGE</div>", unsafe_allow_html=True)

    sample_cat = st.selectbox(
        "Choose a product category to browse samples",
        [c.replace('_',' ').title() for c in CATEGORIES],
        label_visibility="collapsed",
        key="demo_cat"
    )
    cat_key = sample_cat.lower().replace(' ','_')
    files   = SAMPLE_FILES.get(cat_key, ['normal_01.png', None, None])
    labels  = ['✅ Normal', '⚠️ Anomaly Type 1', '⚠️ Anomaly Type 2']
    colors  = ['#10b981', '#ef4444', '#ef4444']

    s_cols = st.columns(3)
    selected_sample = None
    selected_sample_label = None

    for i,(fn,lbl,col) in enumerate(zip(files, labels, colors)):
        with s_cols[i]:
            if fn:
                img = fetch_sample(cat_key, fn)
                if img:
                    st.image(img, use_column_width=True)
                    st.markdown(f"<div style='text-align:center; color:{col}; font-size:0.78rem; font-weight:600; margin:4px 0 8px;'>{lbl}</div>", unsafe_allow_html=True)
                    if st.button(f"Use this", key=f"s{i}"):
                        selected_sample = img
                        selected_sample_label = lbl
            else:
                st.markdown("""
                <div style="background:#0d1117; border:1px dashed #1e2d3d; border-radius:10px;
                            padding:40px 10px; text-align:center; color:#484f58; font-size:0.78rem;">
                    Only 1 anomaly type for this category
                </div>
                """, unsafe_allow_html=True)

    # Determine image to analyze
    analyze_img   = None
    analyze_label = None
    if selected_sample is not None:
        analyze_img   = selected_sample
        analyze_label = f"{sample_cat} — {selected_sample_label}"
        st.success(f"Sample loaded: {analyze_label}")
    elif uploaded_file is not None:
        analyze_img   = Image.open(uploaded_file).convert("RGB")
        analyze_label = uploaded_file.name

    # Analyze button
    st.markdown("<br>", unsafe_allow_html=True)
    b1,b2,b3 = st.columns([1,1,1])
    with b2:
        go = st.button("🔍  Analyze Image", disabled=(analyze_img is None), use_container_width=True)

    # Results
    if go and analyze_img:
        with st.spinner("Running AI analysis..."):
            pred, conf, p_norm, p_anom, cam_fig = run_prediction(model, analyze_img)

        st.markdown("<hr style='border-color:#1e2d3d; margin:32px 0;'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="section-eyebrow">ANALYSIS COMPLETE</div>
        <div class="section-title">Prediction Results</div>
        <div class="section-body" style="margin-bottom:24px;">Image: {analyze_label} &nbsp;·&nbsp; Model: ConvNeXtTiny</div>
        """, unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)

        with r1:
            if pred == 0:
                st.markdown(f"""
                <div class="result-pass">
                    <div style="font-size:3rem; line-height:1;">✅</div>
                    <div class="result-verdict" style="color:#10b981;">NORMAL</div>
                    <div class="result-sub">No defect detected</div>
                    <div style="margin-top:12px; font-family:'Space Grotesk',sans-serif;
                                font-size:1.3rem; font-weight:700; color:#10b981;">
                        {conf*100:.1f}% confident
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-fail">
                    <div style="font-size:3rem; line-height:1;">⚠️</div>
                    <div class="result-verdict" style="color:#ef4444;">ANOMALOUS</div>
                    <div class="result-sub">Defect detected</div>
                    <div style="margin-top:12px; font-family:'Space Grotesk',sans-serif;
                                font-size:1.3rem; font-weight:700; color:#ef4444;">
                        {conf*100:.1f}% confident
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with r2:
            st.markdown("<div class='card-label'>CONFIDENCE BREAKDOWN</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin-bottom:6px; display:flex; justify-content:space-between;">
                <span style="color:#10b981; font-size:0.85rem; font-weight:600;">Normal</span>
                <span style="color:#f0f6fc; font-size:0.85rem;">{p_norm*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(p_norm)
            st.markdown(f"""
            <div style="margin:16px 0 6px; display:flex; justify-content:space-between;">
                <span style="color:#ef4444; font-size:0.85rem; font-weight:600;">Anomalous</span>
                <span style="color:#f0f6fc; font-size:0.85rem;">{p_anom*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(p_anom)

        with r3:
            st.markdown("<div class='card-label'>MODEL INFO</div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="kpi-card" style="margin-bottom:10px;">
                <div class="kpi-val" style="font-size:1rem;">ConvNeXtTiny</div>
                <div class="kpi-lbl">Architecture</div>
            </div>
            <div class="kpi-card" style="margin-bottom:10px;">
                <div class="kpi-val">90.66%</div>
                <div class="kpi-lbl">Test Accuracy</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val">0.9544</div>
                <div class="kpi-lbl">ROC-AUC Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card-label'>GRAD-CAM VISUAL EXPLANATION</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-pill" style="margin-bottom:14px;">
            <strong style="color:#3b82f6;">Grad-CAM</strong> reveals where the model looked.
            🔴 <strong>Red/yellow</strong> regions had the highest influence on this prediction.
            🔵 <strong>Blue</strong> regions had little to no influence.
            This makes the AI decision transparent and trustworthy.
        </div>
        """, unsafe_allow_html=True)
        st.image(cam_fig, use_column_width=True)

    footer()


# ============================================================
# PAGE 2 — PROJECT INFO
# ============================================================
elif page == "📋  Project Info":

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">RESEARCH PROJECT · SS26</div>
        <div class="display-title">Project Overview</div>
        <p class="display-sub" style="margin-top:16px;">
            Explainable Multi-Product Industrial Anomaly Classification
            using CNN Transfer Learning and Cross-Category Evaluation
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Problem statement
    st.markdown("""
    <div class="section-eyebrow">THE PROBLEM</div>
    <div class="section-title">Why Automated Defect Detection?</div>
    """, unsafe_allow_html=True)

    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:2rem; margin-bottom:12px;">🏭</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                        color:#f0f6fc; font-size:1rem; margin-bottom:10px;">
                Manual Inspection is Broken
            </div>
            <div class="section-body">
                In modern manufacturing lines, human inspectors must examine thousands
                of products per hour. This is slow, expensive, inconsistent, and prone
                to fatigue-related errors. A single missed defect can cause product
                recalls, safety hazards, and massive financial losses.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-size:2rem; margin-bottom:12px;">🤖</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                        color:#f0f6fc; font-size:1rem; margin-bottom:10px;">
                AI as the Solution
            </div>
            <div class="section-body">
                Deep learning models can inspect images in milliseconds with consistent
                accuracy. This project builds and compares three CNN architectures to
                find the best balance of accuracy, speed, and explainability for
                real industrial deployment.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Dataset
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-eyebrow">DATASET</div>
    <div class="section-title">MVTec Anomaly Detection</div>
    <div class="section-body" style="margin-bottom:24px;">
        The MVTec AD dataset is the gold standard benchmark for industrial anomaly detection,
        containing high-resolution images of 15 different industrial products and textures.
    </div>
    """, unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    for col, val, lbl in zip(
        [d1,d2,d3,d4],
        ["5,354","15","73","2"],
        ["Total Images","Categories","Defect Types","Classes"]
    ):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-val">{val}</div>
                <div class="kpi-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="card-label">ALL 15 PRODUCT CATEGORIES</div>
        <div style="margin-top:8px;">
            <span class="cat-pill">Bottle</span>
            <span class="cat-pill">Cable</span>
            <span class="cat-pill">Capsule</span>
            <span class="cat-pill">Carpet</span>
            <span class="cat-pill">Grid</span>
            <span class="cat-pill">Hazelnut</span>
            <span class="cat-pill">Leather</span>
            <span class="cat-pill">Metal Nut</span>
            <span class="cat-pill">Pill</span>
            <span class="cat-pill">Screw</span>
            <span class="cat-pill">Tile</span>
            <span class="cat-pill">Toothbrush</span>
            <span class="cat-pill">Transistor</span>
            <span class="cat-pill">Wood</span>
            <span class="cat-pill">Zipper</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Methodology pipeline
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-eyebrow">METHODOLOGY</div>
    <div class="section-title">Research Pipeline</div>
    <div class="section-body" style="margin-bottom:24px;">
        A systematic 6-stage pipeline from raw data to deployed explainable model.
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("1","Data Loading","15 categories, 5K+ images, binary labels"),
        ("2","Preprocessing","Resize 224×224, normalize, augment"),
        ("3","Model Training","3 CNNs with early stopping"),
        ("4","Evaluation","F1, AUC, confusion matrix"),
        ("5","Explainability","Grad-CAM + SHAP analysis"),
        ("6","Deployment","Streamlit web application"),
    ]
    cols = st.columns(6)
    for col, (num, name, desc) in zip(cols, steps):
        with col:
            color = "#10b981" if num == "6" else "#3b82f6"
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-num" style="color:{color};">{num}</div>
                <div class="step-name">{name}</div>
                <div class="step-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Models
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-eyebrow">MODELS</div>
    <div class="section-title">Three Architectures Compared</div>
    <div class="section-body" style="margin-bottom:24px;">
        From a simple custom baseline to state-of-the-art transfer learning models.
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3, gap="large")
    with m1:
        st.markdown("""
        <div class="model-card base">
            <div style="font-size:2.2rem;">🏗️</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.05rem;
                        font-weight:700; color:#f0f6fc; margin:10px 0 4px;">Custom CNN</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                        color:#484f58; letter-spacing:0.1em; margin-bottom:12px;">BASELINE</div>
            <div class="section-body" style="font-size:0.83rem;">
                4 convolutional blocks built from scratch.
                No pretrained weights. Serves as our
                performance lower bound.
            </div>
            <div style="margin-top:16px; padding-top:16px; border-top:1px solid #1e2d3d;">
                <div style="color:#6e7681; font-size:0.75rem;">422,530 params · 1.63 MB</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
                            font-weight:700; color:#f0f6fc; margin-top:4px;">63.77%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="model-card mid">
            <div style="font-size:2.2rem;">⚡</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.05rem;
                        font-weight:700; color:#f0f6fc; margin:10px 0 4px;">EfficientNetV2S</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                        color:#3b82f6; letter-spacing:0.1em; margin-bottom:12px;">TRANSFER LEARNING</div>
            <div class="section-body" style="font-size:0.83rem;">
                Pretrained on ImageNet. Two-stage training:
                frozen backbone then partial fine-tuning
                of last 2 blocks.
            </div>
            <div style="margin-top:16px; padding-top:16px; border-top:1px solid #1e2d3d;">
                <div style="color:#6e7681; font-size:0.75rem;">15.2M params · 79.10 MB</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
                            font-weight:700; color:#3b82f6; margin-top:4px;">86.93%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown("""
        <div class="model-card best">
            <div style="font-size:2.2rem;">🏆</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.05rem;
                        font-weight:700; color:#f0f6fc; margin:10px 0 4px;">ConvNeXtTiny</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                        color:#10b981; letter-spacing:0.1em; margin-bottom:12px;">BEST MODEL ★</div>
            <div class="section-body" style="font-size:0.83rem;">
                Modern ConvNet architecture pretrained on
                ImageNet. Outperformed all models across
                every evaluation metric.
            </div>
            <div style="margin-top:16px; padding-top:16px; border-top:1px solid #1e2d3d;">
                <div style="color:#6e7681; font-size:0.75rem;">15.7M params · 106.95 MB</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
                            font-weight:700; color:#10b981; margin-top:4px;">90.66%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tech stack
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-eyebrow">TECHNOLOGY</div>
    <div class="section-title">Tech Stack</div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <span class="tech-badge">Python 3.10</span>
        <span class="tech-badge">PyTorch</span>
        <span class="tech-badge">Torchvision</span>
        <span class="tech-badge">ConvNeXtTiny</span>
        <span class="tech-badge">EfficientNetV2S</span>
        <span class="tech-badge">Grad-CAM</span>
        <span class="tech-badge">SHAP</span>
        <span class="tech-badge">OpenCV</span>
        <span class="tech-badge">Scikit-learn</span>
        <span class="tech-badge">Matplotlib</span>
        <span class="tech-badge">Seaborn</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">Kaggle GPU T4×2</span>
        <span class="tech-badge">Hugging Face</span>
    </div>
    """, unsafe_allow_html=True)

    footer()


# ============================================================
# PAGE 3 — RESULTS
# ============================================================
elif page == "📊  Results":

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">EXPERIMENTAL RESULTS</div>
        <div class="display-title">Model Performance</div>
        <p class="display-sub" style="margin-top:16px;">
            Complete evaluation of all 3 models on the held-out test set.
            ConvNeXtTiny achieved best results across every metric.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPIs
    k1,k2,k3,k4,k5 = st.columns(5)
    for col, val, lbl in zip(
        [k1,k2,k3,k4,k5],
        ["90.66%","0.9544","0.8706","0.8916","4.4ms"],
        ["Best Accuracy","Best ROC-AUC","Best Macro F1","Best PR-AUC","Best Inference"]
    ):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-val">{val}</div>
                <div class="kpi-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    # Full table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-eyebrow">COMPARISON TABLE</div>
    <div class="section-title">Full Metrics Breakdown</div>
    """, unsafe_allow_html=True)

    df = pd.DataFrame({
        "Model":           ["Custom CNN","EfficientNetV2S","ConvNeXtTiny ⭐"],
        "Accuracy":        [0.6377, 0.8693, 0.9066],
        "Macro Precision": [0.5409, 0.8318, 0.8697],
        "Macro Recall":    [0.5481, 0.7949, 0.8636],
        "Macro F1":        [0.5499, 0.8142, 0.8706],
        "Weighted F1":     [0.6531, 0.8674, 0.9044],
        "ROC-AUC":         [0.5775, 0.9025, 0.9544],
        "PR-AUC":          [0.2821, 0.8159, 0.8916],
        "Inference (ms)":  [0.790,  3.154,  4.401 ],
        "Params (M)":      [0.42,   15.22,  15.67 ],
        "Size (MB)":       [1.63,   79.10,  106.95],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Per category
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-eyebrow">CATEGORY ANALYSIS</div>
    <div class="section-title">Per-Category Performance</div>
    <div class="section-body" style="margin-bottom:24px;">
        ConvNeXtTiny's accuracy broken down by each of the 15 product categories.
    </div>
    """, unsafe_allow_html=True)

    cat_data = {
        "Category":  ["Bottle","Cable","Capsule","Carpet","Grid","Hazelnut",
                      "Leather","Metal Nut","Pill","Screw","Tile","Toothbrush",
                      "Transistor","Wood","Zipper"],
        "Accuracy":  [0.95, 0.88, 0.91, 1.00, 0.93, 0.96,
                      0.98, 0.89, 0.90, 0.82, 0.94, 0.70,
                      0.87, 0.92, 0.95],
    }
    cat_df = pd.DataFrame(cat_data).sort_values("Accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    colors_bar = ['#10b981' if v >= 0.90 else '#3b82f6' if v >= 0.80 else '#ef4444'
                  for v in cat_df["Accuracy"]]
    bars = ax.bar(cat_df["Category"], cat_df["Accuracy"], color=colors_bar,
                  edgecolor='none', width=0.65)
    ax.axhline(y=cat_df["Accuracy"].mean(), color='#f59e0b', linestyle='--',
               linewidth=1.5, label=f'Mean = {cat_df["Accuracy"].mean():.2f}')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy", color='#8b949e')
    ax.tick_params(colors='#8b949e', axis='both')
    ax.spines[['top','right','left','bottom']].set_color('#1e2d3d')
    ax.set_facecolor('#0d1117')
    for spine in ax.spines.values(): spine.set_color('#1e2d3d')
    ax.tick_params(axis='x', rotation=35)
    for bar, val in zip(bars, cat_df["Accuracy"]):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.0%}',
                ha='center', fontsize=8, color='#8b949e')
    ax.legend(facecolor='#0d1117', labelcolor='#f59e0b', edgecolor='#1e2d3d')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Research questions
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-eyebrow">RESEARCH QUESTIONS</div>
    <div class="section-title">Key Findings</div>
    """, unsafe_allow_html=True)

    rqs = [
        ("RQ1","Which architecture achieves strongest performance?",
         "ConvNeXtTiny — 90.66% accuracy, 0.9544 ROC-AUC, best across all metrics.","#3b82f6"),
        ("RQ2","How much does performance vary between categories?",
         "Range of 30% — Carpet achieved 100%, Toothbrush was hardest at 70%.","#3b82f6"),
        ("RQ3","Do Grad-CAM and SHAP overlap with anomalous regions?",
         "Yes — correct predictions consistently focus on actual defect areas.","#3b82f6"),
        ("RQ4","Which categories produce highest false-negative rates?",
         "Toothbrush and Screw — fine-grained defects are hardest to detect.","#3b82f6"),
        ("RQ5","Can a compact model give a practical trade-off?",
         "Custom CNN at 0.790ms/image is 5x faster but 27% less accurate.","#3b82f6"),
    ]

    for rq, q, a, c in rqs:
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom:10px; padding:20px 24px;">
            <div style="display:flex; gap:16px; align-items:flex-start;">
                <div style="background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.25);
                            color:#3b82f6; padding:5px 12px; border-radius:8px;
                            font-family:'JetBrains Mono',monospace; font-size:0.75rem;
                            font-weight:600; white-space:nowrap;">{rq}</div>
                <div>
                    <div style="color:#f0f6fc; font-size:0.88rem; font-weight:500;">{q}</div>
                    <div style="color:#10b981; font-size:0.83rem; margin-top:6px;">→ {a}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    footer()


# ============================================================
# PAGE 4 — EXPLAINABILITY
# ============================================================
elif page == "🔬  Explainability":

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">EXPLAINABLE AI · XAI</div>
        <div class="display-title">Why Should You<br>Trust the Model?</div>
        <p class="display-sub" style="margin-top:16px;">
            A high accuracy alone is not enough. We use Grad-CAM and SHAP
            to verify the model is looking at the right things.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # GradCAM explanation
    st.markdown("""
    <div class="section-eyebrow">GRAD-CAM</div>
    <div class="section-title">Gradient-weighted Class Activation Maps</div>
    <div class="section-body" style="margin-bottom:24px;">
        Grad-CAM uses the gradients flowing into the last convolutional layer to produce
        a coarse localization map highlighting the important regions in the image for
        predicting the concept. Red and yellow mean high importance, blue means low.
    </div>
    """, unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:2rem;">🎯</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                        color:#f0f6fc; margin:10px 0 8px;">Correct Prediction</div>
            <div class="section-body" style="font-size:0.82rem;">
                For correctly classified anomalous images, Grad-CAM highlights
                the actual defect region — scratch, crack, or contamination area.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:2rem;">❓</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                        color:#f0f6fc; margin:10px 0 8px;">Wrong Prediction</div>
            <div class="section-body" style="font-size:0.82rem;">
                For misclassified images, Grad-CAM often reveals the model focused
                on background or irrelevant regions — exposing why it failed.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with g3:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:2rem;">🔍</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                        color:#f0f6fc; margin:10px 0 8px;">Limitations</div>
            <div class="section-body" style="font-size:0.82rem;">
                Grad-CAM produces coarse maps and cannot pinpoint exact pixel-level
                regions. It shows where the model looks, not why at a fine-grained level.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # SHAP explanation
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-eyebrow">SHAP</div>
    <div class="section-title">SHapley Additive exPlanations</div>
    <div class="section-body" style="margin-bottom:24px;">
        SHAP assigns each pixel a value showing how much it contributed to the prediction.
        Red regions push the prediction toward Anomalous. Blue regions push toward Normal.
        The magnitude shows how strongly each region influenced the final decision.
    </div>
    """, unsafe_allow_html=True)

    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.markdown("""
        <div class="glass-card">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:14px;">
                <div style="width:16px; height:16px; background:#ef4444; border-radius:3px;"></div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                            color:#f0f6fc; font-size:0.95rem;">Red Regions — Support Anomalous</div>
            </div>
            <div class="section-body" style="font-size:0.83rem;">
                Pixels highlighted in red are pushing the model to classify this image
                as Anomalous. These typically correspond to the actual defect areas —
                scratches, cracks, contamination, or structural damage.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown("""
        <div class="glass-card">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:14px;">
                <div style="width:16px; height:16px; background:#3b82f6; border-radius:3px;"></div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                            color:#f0f6fc; font-size:0.95rem;">Blue Regions — Support Normal</div>
            </div>
            <div class="section-body" style="font-size:0.83rem;">
                Pixels highlighted in blue are working against the anomalous classification.
                These are regions the model considers clean and defect-free, pulling
                the prediction toward Normal.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Why XAI matters
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-eyebrow">IMPORTANCE</div>
    <div class="section-title">Why Explainability Matters</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px;">
            <div>
                <div style="font-size:1.5rem; margin-bottom:8px;">🛡️</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                            color:#f0f6fc; font-size:0.9rem; margin-bottom:6px;">Trust</div>
                <div class="section-body" style="font-size:0.8rem;">
                    Factory operators need to trust the system before replacing human inspectors.
                    Visual explanations build that trust.
                </div>
            </div>
            <div>
                <div style="font-size:1.5rem; margin-bottom:8px;">🐛</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                            color:#f0f6fc; font-size:0.9rem; margin-bottom:6px;">Debugging</div>
                <div class="section-body" style="font-size:0.8rem;">
                    When the model fails, explanations show if it was focusing on
                    background patterns instead of actual product features.
                </div>
            </div>
            <div>
                <div style="font-size:1.5rem; margin-bottom:8px;">📋</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                            color:#f0f6fc; font-size:0.9rem; margin-bottom:6px;">Compliance</div>
                <div class="section-body" style="font-size:0.8rem;">
                    In regulated industries, AI decisions must be explainable.
                    Grad-CAM and SHAP provide the required audit trail.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-pill" style="margin-top:16px;">
        <strong style="color:#f59e0b;">⚠️ Important caveat:</strong>
        Attractive heatmaps are not proof of correct reasoning. A model can produce
        visually plausible Grad-CAM maps while still failing on edge cases.
        Explainability tools should be used alongside rigorous quantitative evaluation,
        not as a substitute for it.
    </div>
    """, unsafe_allow_html=True)

    footer()


# ============================================================
# PAGE 5 — ABOUT
# ============================================================
elif page == "👤  About":

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">ABOUT THIS PROJECT</div>
        <div class="display-title">Abdullah Rashid</div>
        <p class="display-sub" style="margin-top:16px;">
            3rd Semester · Bachelor of Software Engineering<br>
            University of Europe for Applied Sciences, Potsdam
        </p>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns([1,1], gap="large")

    with a1:
        st.markdown("""
        <div class="section-eyebrow">CONTACT</div>
        <div class="section-title">Get in Touch</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div style="display:flex; flex-direction:column; gap:16px;">
                <div style="display:flex; align-items:center; gap:14px;">
                    <div style="width:40px; height:40px; background:rgba(59,130,246,0.12);
                                border:1px solid rgba(59,130,246,0.25); border-radius:10px;
                                display:flex; align-items:center; justify-content:center;
                                font-size:1.1rem;">📧</div>
                    <div>
                        <div style="font-size:0.72rem; color:#6e7681; margin-bottom:2px;">EMAIL</div>
                        <a href="mailto:abdullahrashid130704@outlook.com"
                           style="color:#3b82f6; text-decoration:none; font-size:0.88rem;">
                            abdullahrashid130704@outlook.com
                        </a>
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:14px;">
                    <div style="width:40px; height:40px; background:rgba(59,130,246,0.12);
                                border:1px solid rgba(59,130,246,0.25); border-radius:10px;
                                display:flex; align-items:center; justify-content:center;
                                font-size:1.1rem;">📞</div>
                    <div>
                        <div style="font-size:0.72rem; color:#6e7681; margin-bottom:2px;">PHONE</div>
                        <div style="color:#f0f6fc; font-size:0.88rem;">+49 15 510 337 507</div>
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:14px;">
                    <div style="width:40px; height:40px; background:rgba(59,130,246,0.12);
                                border:1px solid rgba(59,130,246,0.25); border-radius:10px;
                                display:flex; align-items:center; justify-content:center;
                                font-size:1.1rem;">🔗</div>
                    <div>
                        <div style="font-size:0.72rem; color:#6e7681; margin-bottom:2px;">LINKEDIN</div>
                        <a href="https://linkedin.com/in/abdullahr2004" target="_blank"
                           style="color:#3b82f6; text-decoration:none; font-size:0.88rem;">
                            linkedin.com/in/abdullahr2004
                        </a>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="section-eyebrow">PROJECT LINKS</div>
        <div class="section-title">Resources</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div style="display:flex; flex-direction:column; gap:14px;">
                <a href="https://github.com/abdullah1307-git/mvtec-anomaly-classification"
                   target="_blank" style="text-decoration:none;">
                    <div style="display:flex; align-items:center; gap:14px; padding:12px;
                                background:rgba(255,255,255,0.03); border:1px solid #1e2d3d;
                                border-radius:10px; transition:border-color 0.2s;">
                        <div style="font-size:1.3rem;">💻</div>
                        <div>
                            <div style="color:#f0f6fc; font-size:0.88rem; font-weight:500;">GitHub Repository</div>
                            <div style="color:#6e7681; font-size:0.75rem; margin-top:2px;">Source code, notebook, figures</div>
                        </div>
                    </div>
                </a>
                <a href="https://www.kaggle.com/code/abdullah130704/notebook8b312b2661"
                   target="_blank" style="text-decoration:none;">
                    <div style="display:flex; align-items:center; gap:14px; padding:12px;
                                background:rgba(255,255,255,0.03); border:1px solid #1e2d3d;
                                border-radius:10px;">
                        <div style="font-size:1.3rem;">📓</div>
                        <div>
                            <div style="color:#f0f6fc; font-size:0.88rem; font-weight:500;">Kaggle Notebook</div>
                            <div style="color:#6e7681; font-size:0.75rem; margin-top:2px;">Full executed training notebook</div>
                        </div>
                    </div>
                </a>
                <a href="https://www.kaggle.com/datasets/ipythonx/mvtec-ad"
                   target="_blank" style="text-decoration:none;">
                    <div style="display:flex; align-items:center; gap:14px; padding:12px;
                                background:rgba(255,255,255,0.03); border:1px solid #1e2d3d;
                                border-radius:10px;">
                        <div style="font-size:1.3rem;">📊</div>
                        <div>
                            <div style="color:#f0f6fc; font-size:0.88rem; font-weight:500;">MVTec AD Dataset</div>
                            <div style="color:#6e7681; font-size:0.75rem; margin-top:2px;">15 categories · 5000+ images</div>
                        </div>
                    </div>
                </a>
                <a href="https://huggingface.co/abdullah130704/mvtec-anomaly-model"
                   target="_blank" style="text-decoration:none;">
                    <div style="display:flex; align-items:center; gap:14px; padding:12px;
                                background:rgba(255,255,255,0.03); border:1px solid #1e2d3d;
                                border-radius:10px;">
                        <div style="font-size:1.3rem;">🤗</div>
                        <div>
                            <div style="color:#f0f6fc; font-size:0.88rem; font-weight:500;">Hugging Face Model</div>
                            <div style="color:#6e7681; font-size:0.75rem; margin-top:2px;">ConvNeXtTiny weights · 112 MB</div>
                        </div>
                    </div>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Course info
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-eyebrow">ACADEMIC CONTEXT</div>
    <div class="section-title">Course Information</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:20px;">
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                            color:#484f58; letter-spacing:0.1em; margin-bottom:6px;">COURSE</div>
                <div style="color:#f0f6fc; font-size:0.88rem; font-weight:500;">
                    Machine Learning & Smart Systems
                </div>
            </div>
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                            color:#484f58; letter-spacing:0.1em; margin-bottom:6px;">UNIVERSITY</div>
                <div style="color:#f0f6fc; font-size:0.88rem; font-weight:500;">
                    UE for Applied Sciences, Potsdam
                </div>
            </div>
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                            color:#484f58; letter-spacing:0.1em; margin-bottom:6px;">SEMESTER</div>
                <div style="color:#f0f6fc; font-size:0.88rem; font-weight:500;">
                    Summer Semester 2026 · 3rd Semester
                </div>
            </div>
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                            color:#484f58; letter-spacing:0.1em; margin-bottom:6px;">PROGRAMME</div>
                <div style="color:#f0f6fc; font-size:0.88rem; font-weight:500;">
                    Bachelor of Software Engineering
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    footer()
