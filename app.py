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
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: #080c14;
    background-image:
        radial-gradient(ellipse at 20% 10%, rgba(59,130,246,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 90%, rgba(139,92,246,0.05) 0%, transparent 50%);
}

/* Hide default sidebar toggle */
[data-testid="collapsedControl"] { display: none; }
[data-testid="stSidebar"] { display: none; }

/* ---- TOP NAV ---- */
.topnav {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(8,12,20,0.92);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid #1e2d3d;
    padding: 0 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 58px;
    margin: -1rem -1rem 2rem -1rem;
}

.topnav-brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    background: linear-gradient(135deg, #3b82f6, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    white-space: nowrap;
}

.topnav-links {
    display: flex;
    gap: 4px;
}

.navlink {
    padding: 7px 16px;
    border-radius: 8px;
    font-size: 0.83rem;
    font-weight: 500;
    color: #8b949e;
    cursor: pointer;
    border: none;
    background: transparent;
    transition: all 0.2s;
    white-space: nowrap;
}

.navlink:hover { color: #f0f6fc; background: rgba(255,255,255,0.06); }
.navlink.active { color: #3b82f6; background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.2); }

/* ---- TYPOGRAPHY ---- */
.display-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.6rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #93c5fd 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.display-sub {
    font-size: 1.05rem;
    color: #8b949e;
    line-height: 1.7;
    max-width: 580px;
    margin: 0 auto;
}

.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 8px;
}

.sec-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #f0f6fc;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
}

.sec-body { color: #8b949e; font-size: 0.9rem; line-height: 1.7; }

/* ---- HERO ---- */
.hero {
    background: linear-gradient(180deg, #0d1117 0%, #080c14 100%);
    border: 1px solid #1e2d3d;
    border-radius: 20px;
    padding: 56px 48px;
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
    width: 500px; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.6), transparent);
}

.hero::after {
    content: '';
    position: absolute;
    top: -150px; left: 50%;
    transform: translateX(-50%);
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(59,130,246,0.09) 0%, transparent 70%);
    pointer-events: none;
}

.badge {
    display: inline-flex;
    align-items: center;
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
    font-family: 'JetBrains Mono', monospace;
}

/* ---- STATS ROW ---- */
.stats-row {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 28px;
}

.stat-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 12px 22px;
    text-align: center;
    min-width: 100px;
}

.stat-v {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #3b82f6;
    line-height: 1;
}

.stat-l {
    font-size: 0.68rem;
    color: #6e7681;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ---- CARDS ---- */
.gcard {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 14px;
    transition: border-color 0.2s;
}

.gcard:hover { border-color: #2d4a6e; }

.clabel {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #484f58;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 14px;
}

/* ---- KPI ---- */
.kpi {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    transition: border-color 0.2s, transform 0.15s;
}
.kpi:hover { border-color: rgba(59,130,246,0.35); transform: translateY(-2px); }
.kpi-v { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #3b82f6; line-height: 1; }
.kpi-l { font-size: 0.68rem; color: #6e7681; margin-top: 6px; letter-spacing: 0.04em; }

/* ---- RESULT CARDS ---- */
.res-pass {
    background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(16,185,129,0.04));
    border: 2px solid rgba(16,185,129,0.5);
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    box-shadow: 0 0 30px rgba(16,185,129,0.08);
}

.res-fail {
    background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.04));
    border: 2px solid rgba(239,68,68,0.5);
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    box-shadow: 0 0 30px rgba(239,68,68,0.08);
}

.res-icon { font-size: 3.5rem; line-height: 1; margin-bottom: 10px; }

.res-verdict {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0;
}

.res-conf {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 8px;
}

.res-sub { font-size: 0.82rem; color: #8b949e; margin-top: 4px; }

/* ---- CONFIDENCE BAR ---- */
.conf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.conf-label { font-size: 0.83rem; font-weight: 600; }
.conf-val { font-size: 0.83rem; color: #f0f6fc; font-family: 'Space Grotesk', sans-serif; font-weight: 700; }

/* ---- BUTTONS ---- */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 10px 20px !important;
    transition: opacity 0.2s, transform 0.15s !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
}

div[data-testid="stButton"] button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

div[data-testid="stButton"] button:disabled {
    background: #1e2d3d !important;
    color: #484f58 !important;
    opacity: 1 !important;
    transform: none !important;
}

/* Sample use buttons - smaller */
div[data-testid="stButton"]:has(button[kind="secondary"]) button {
    background: rgba(59,130,246,0.15) !important;
    border: 1px solid rgba(59,130,246,0.3) !important;
    color: #60a5fa !important;
    font-size: 0.78rem !important;
    padding: 7px 14px !important;
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

/* ---- SAMPLE IMAGE CARD ---- */
.sample-wrap {
    border-radius: 10px;
    overflow: hidden;
    border: 2px solid #1e2d3d;
    transition: border-color 0.2s, transform 0.2s;
    cursor: pointer;
}
.sample-wrap:hover { border-color: #3b82f6; transform: translateY(-2px); }
.sample-wrap.normal { border-color: rgba(16,185,129,0.4); }
.sample-wrap.anomaly { border-color: rgba(239,68,68,0.3); }

.sample-tag {
    padding: 5px 0;
    text-align: center;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ---- SELECTED IMAGE PREVIEW ---- */
.selected-preview {
    background: #0d1117;
    border: 2px solid rgba(59,130,246,0.4);
    border-radius: 12px;
    padding: 10px;
    text-align: center;
}

/* ---- MODEL CARDS ---- */
.mcard {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 14px;
    padding: 24px;
    height: 100%;
    transition: transform 0.2s, border-color 0.2s;
}
.mcard:hover { transform: translateY(-3px); }
.mcard.best { border-color: rgba(16,185,129,0.4); }
.mcard.mid  { border-color: rgba(59,130,246,0.3); }

/* ---- PIPELINE ---- */
.pstep {
    background: #0d1117;
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 16px 12px;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}
.pstep:hover { border-color: rgba(59,130,246,0.35); transform: translateY(-2px); }

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
.tbadge:hover { border-color: #3b82f6; color: #60a5fa; }

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
}

/* ---- STREAMLIT OVERRIDES ---- */
h1,h2,h3,h4,h5,h6 { color: #f0f6fc !important; }
p, li { color: #8b949e; }
.stProgress > div > div { background: linear-gradient(90deg, #3b82f6, #818cf8) !important; }
[data-testid="stFileUploader"] {
    background: #0d1117 !important;
    border: 2px dashed #1e2d3d !important;
    border-radius: 12px !important;
}
.stSelectbox > div > div {
    background: #0d1117 !important;
    border: 1px solid #1e2d3d !important;
    color: #f0f6fc !important;
}
.stDataFrame { border: 1px solid #1e2d3d; border-radius: 12px; }
.stSuccess { background: rgba(16,185,129,0.1) !important; border: 1px solid rgba(16,185,129,0.3) !important; border-radius: 10px !important; }
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


# ============================================================
# TOP NAV
# ============================================================
nav_cols = st.columns([1.2, 5, 0.5])
with nav_cols[0]:
    st.markdown("<div class='topnav-brand' style='padding:16px 0;'>🔍 AnomalyVision</div>",
                unsafe_allow_html=True)

with nav_cols[1]:
    btn_cols = st.columns(len(PAGES))
    for i, (col, pg) in enumerate(zip(btn_cols, PAGES)):
        with col:
            active = "active" if st.session_state.page == pg else ""
            if st.button(pg.split(" ", 1)[1], key=f"nav_{i}",
                         use_container_width=True):
                st.session_state.page = pg
                st.session_state.selected_sample_img = None
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


def gradcam_figure(pil_img, heatmap):
    img = np.array(pil_img.resize((IMG_SIZE,IMG_SIZE))).astype(np.float32)/255
    h   = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    h   = cv2.cvtColor(h, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    ov  = np.clip(0.45*h + 0.55*img, 0, 1)
    fig, axes = plt.subplots(1,2,figsize=(11,4.5))
    fig.patch.set_facecolor('#0d1117')
    for a in axes: a.set_facecolor('#0d1117')
    axes[0].imshow(img)
    axes[0].set_title('Original Image', color='#c9d1d9', fontsize=12, pad=12, fontweight='600')
    axes[0].axis('off')
    axes[1].imshow(ov)
    axes[1].set_title('Grad-CAM Heatmap — Model Focus Areas', color='#c9d1d9', fontsize=12, pad=12, fontweight='600')
    axes[1].axis('off')
    plt.tight_layout(pad=2.5)
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
    r = requests.get(f"{SAMPLE_BASE}/{cat}/{fname}")
    if r.status_code == 200:
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return None


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

    st.markdown("""
    <div class="hero">
        <div class="badge">● LIVE INFERENCE ENGINE ACTIVE</div>
        <div class="display-title">Detect Defects<br>in Milliseconds</div>
        <p class="display-sub" style="margin-top:14px;">
            Upload any industrial product image and get an instant AI-powered
            quality assessment with visual explanations.
        </p>
        <div class="stats-row">
            <div class="stat-box"><div class="stat-v">90.66%</div><div class="stat-l">Accuracy</div></div>
            <div class="stat-box"><div class="stat-v">0.9544</div><div class="stat-l">ROC-AUC</div></div>
            <div class="stat-box"><div class="stat-v">4.4ms</div><div class="stat-l">Inference</div></div>
            <div class="stat-box"><div class="stat-v">15</div><div class="stat-l">Categories</div></div>
            <div class="stat-box"><div class="stat-v">5K+</div><div class="stat-l">Images Trained</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("Model could not be loaded. Please refresh the page.")
        st.stop()

    # ---- Upload section ----
    up_col, prev_col = st.columns([1,1], gap="large")

    with up_col:
        st.markdown("<div class='clabel'>UPLOAD YOUR OWN IMAGE</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop image here",
            type=["jpg","jpeg","png","bmp"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            # Clear any selected sample when user uploads
            st.session_state.selected_sample_img = None
            st.session_state.selected_sample_name = None

        st.markdown("""
        <div class="infobox" style="margin-top:12px;">
            <strong style="color:#3b82f6;">Works with any industrial product.</strong>
            Covers bottles, cables, capsules, carpet, grid, hazelnut, leather,
            metal nuts, pills, screws, tiles, toothbrushes, transistors, wood, and zippers.
        </div>
        """, unsafe_allow_html=True)

    with prev_col:
        st.markdown("<div class='clabel'>IMAGE PREVIEW</div>", unsafe_allow_html=True)
        # Show uploaded file or selected sample
        if uploaded_file:
            display_img = Image.open(uploaded_file)
            st.image(display_img, use_column_width=True)
        elif st.session_state.selected_sample_img is not None:
            st.image(st.session_state.selected_sample_img, use_column_width=True)
            st.markdown(f"""
            <div style="text-align:center; font-size:0.78rem; color:#60a5fa; margin-top:6px;">
                ✓ {st.session_state.selected_sample_name}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#0d1117; border:2px dashed #1e2d3d; border-radius:12px;
                        padding:60px 20px; text-align:center; color:#484f58; min-height:200px;
                        display:flex; flex-direction:column; align-items:center; justify-content:center;">
                <div style="font-size:2.5rem;">🖼️</div>
                <div style="font-size:0.82rem; margin-top:10px;">
                    Upload an image or select a sample below
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Sample images ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='clabel'>OR TRY A SAMPLE IMAGE</div>", unsafe_allow_html=True)

    sample_cat_display = st.selectbox(
        "Choose category",
        [c.replace('_',' ').title() for c in CATEGORIES],
        label_visibility="collapsed",
        key="sample_cat_select"
    )
    cat_key = sample_cat_display.lower().replace(' ','_')
    files   = SAMPLE_FILES.get(cat_key, ['normal_01.png', None, None])
    labels  = ['✅ Normal', '⚠️ Anomaly Type 1', '⚠️ Anomaly Type 2']
    colors  = ['#10b981', '#ef4444', '#ef4444']

    s_cols = st.columns(3, gap="medium")
    for i, (fn, lbl, clr) in enumerate(zip(files, labels, colors)):
        with s_cols[i]:
            if fn:
                img = fetch_sample(cat_key, fn)
                if img:
                    st.image(img, use_column_width=True)
                    st.markdown(f"""
                    <div style="text-align:center; color:{clr}; font-size:0.75rem;
                                font-weight:600; margin:5px 0 6px;">{lbl}</div>
                    """, unsafe_allow_html=True)
                    if st.button(f"Use this image", key=f"use_sample_{i}_{cat_key}"):
                        st.session_state.selected_sample_img  = img
                        st.session_state.selected_sample_name = f"{sample_cat_display} — {lbl}"
                        st.rerun()
            else:
                st.markdown("""
                <div style="background:#0d1117; border:1px dashed #1e2d3d; border-radius:10px;
                            padding:50px 10px; text-align:center; color:#484f58; font-size:0.76rem;">
                    Only 1 anomaly type for this category
                </div>
                """, unsafe_allow_html=True)

    # ---- Determine image to analyze ----
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
        go = st.button(
            "🔍  Analyze Image",
            disabled=(analyze_img is None),
            use_container_width=True
        )

    # ---- Results ----
    if go and analyze_img:
        with st.spinner("Running AI analysis..."):
            pred, conf, p_norm, p_anom, cam_fig = run_prediction(model, analyze_img)

        st.markdown("<hr style='border-color:#1e2d3d; margin:32px 0 24px;'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="eyebrow">ANALYSIS COMPLETE</div>
        <div class="sec-title" style="margin-bottom:4px;">Prediction Results</div>
        <div style="color:#6e7681; font-size:0.82rem; margin-bottom:24px; font-family:'JetBrains Mono',monospace;">
            {analyze_label} &nbsp;·&nbsp; Model: ConvNeXtTiny
        </div>
        """, unsafe_allow_html=True)

        r1, r2, r3 = st.columns([1.1, 1.2, 0.9], gap="large")

        with r1:
            if pred == 0:
                st.markdown(f"""
                <div class="res-pass">
                    <div class="res-icon">✅</div>
                    <div class="res-verdict" style="color:#10b981;">NORMAL</div>
                    <div class="res-sub">No defect detected in this image</div>
                    <div class="res-conf" style="color:#10b981;">{conf*100:.1f}% confident</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="res-fail">
                    <div class="res-icon">⚠️</div>
                    <div class="res-verdict" style="color:#ef4444;">ANOMALOUS</div>
                    <div class="res-sub">Defect detected in this image</div>
                    <div class="res-conf" style="color:#ef4444;">{conf*100:.1f}% confident</div>
                </div>
                """, unsafe_allow_html=True)

        with r2:
            st.markdown("<div class='clabel'>CONFIDENCE BREAKDOWN</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="conf-row" style="margin-top:8px;">
                <span class="conf-label" style="color:#10b981;">✅ Normal</span>
                <span class="conf-val">{p_norm*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(p_norm)

            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="conf-row">
                <span class="conf-label" style="color:#ef4444;">⚠️ Anomalous</span>
                <span class="conf-val">{p_anom*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(p_anom)

            # Decision bar
            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            dominant_color = "#10b981" if pred == 0 else "#ef4444"
            dominant_label = "NORMAL" if pred == 0 else "ANOMALOUS"
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03); border:1px solid #1e2d3d;
                        border-radius:10px; padding:14px; text-align:center;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem;
                            color:#484f58; letter-spacing:0.1em; margin-bottom:6px;">VERDICT</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
                            font-weight:800; color:{dominant_color};">
                    {dominant_label}
                </div>
                <div style="font-size:0.78rem; color:#6e7681; margin-top:4px;">
                    with {conf*100:.1f}% confidence
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown("<div class='clabel'>MODEL STATS</div>", unsafe_allow_html=True)
            for val, lbl in [("ConvNeXtTiny","Architecture"),
                              ("90.66%","Test Accuracy"),
                              ("0.9544","ROC-AUC"),
                              ("4.4ms","Inference Time"),
                              ("15.7M","Parameters")]:
                st.markdown(f"""
                <div class="kpi" style="margin-bottom:8px; padding:12px 16px;">
                    <div style="font-family:'Space Grotesk',sans-serif; font-size:0.95rem;
                                font-weight:700; color:#3b82f6;">{val}</div>
                    <div class="kpi-l">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

        # GradCAM
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='clabel'>GRAD-CAM VISUAL EXPLANATION</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="infobox" style="margin-bottom:16px;">
            <strong style="color:#3b82f6;">How to read this:</strong>
            The heatmap shows <em>where the model looked</em> when making its decision.
            🔴 <strong>Red/yellow</strong> = high attention regions that influenced the prediction.
            🔵 <strong>Blue</strong> = low attention regions that were mostly ignored.
            For anomalous images, red should highlight the actual defect area.
        </div>
        """, unsafe_allow_html=True)
        st.image(cam_fig, use_column_width=True)

    footer()


# ============================================================
# PAGE 2 — PROJECT INFO
# ============================================================
elif page == "📋 Project Info":

    st.markdown("""
    <div class="hero">
        <div class="badge">RESEARCH PROJECT · SS26</div>
        <div class="display-title">Project Overview</div>
        <p class="display-sub" style="margin-top:14px;">
            Explainable Multi-Product Industrial Anomaly Classification
            using CNN Transfer Learning and Cross-Category Evaluation
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Problem
    st.markdown("""
    <div class="eyebrow">THE PROBLEM</div>
    <div class="sec-title">Why Automated Defect Detection?</div>
    <div class="sec-body" style="margin-bottom:24px;"></div>
    """, unsafe_allow_html=True)

    p1, p2 = st.columns(2, gap="large")
    with p1:
        st.markdown("""
        <div class="gcard">
            <div style="font-size:2rem; margin-bottom:12px;">🏭</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                        color:#f0f6fc; margin-bottom:10px;">Manual Inspection is Broken</div>
            <div class="sec-body" style="font-size:0.84rem;">
                In modern manufacturing lines, human inspectors examine thousands
                of products per hour — slow, expensive, inconsistent, and prone to
                fatigue errors. One missed defect can cause recalls and safety hazards.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="gcard">
            <div style="font-size:2rem; margin-bottom:12px;">🤖</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                        color:#f0f6fc; margin-bottom:10px;">AI as the Solution</div>
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
        The gold standard benchmark for industrial anomaly detection, containing
        high-resolution images of 15 industrial products and textures.
    </div>
    """, unsafe_allow_html=True)

    dcols = st.columns(4)
    for col, v, l in zip(dcols, ["5,354","15","73","2"],
                         ["Total Images","Categories","Defect Types","Classes (Normal/Anomaly)"]):
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
    <div class="sec-body" style="margin-bottom:20px;">
        A systematic 6-stage pipeline from raw data to deployed, explainable AI model.
    </div>
    """, unsafe_allow_html=True)

    pcols = st.columns(6)
    steps = [
        ("1","Data Loading","15 categories\n5K+ images","#3b82f6"),
        ("2","Preprocessing","Resize 224×224\nNormalize & Augment","#3b82f6"),
        ("3","Training","3 CNN models\nEarly stopping","#3b82f6"),
        ("4","Evaluation","F1, AUC\nConfusion Matrix","#3b82f6"),
        ("5","Explainability","Grad-CAM\n+ SHAP","#3b82f6"),
        ("6","Deployment","Streamlit\nWeb App","#10b981"),
    ]
    for col, (num, name, desc, clr) in zip(pcols, steps):
        with col:
            st.markdown(f"""
            <div class="pstep">
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.3rem;
                            font-weight:800; color:{clr};">{num}</div>
                <div style="font-size:0.8rem; font-weight:600; color:#f0f6fc; margin-top:6px;">{name}</div>
                <div style="font-size:0.7rem; color:#6e7681; margin-top:4px; white-space:pre-line;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Models
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">MODELS</div>
    <div class="sec-title">Three Architectures Compared</div>
    <div class="sec-body" style="margin-bottom:20px;">From a simple custom baseline to state-of-the-art transfer learning.</div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3, gap="large")
    with m1:
        st.markdown("""
        <div class="mcard">
            <div style="font-size:2rem;">🏗️</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1rem; font-weight:700;
                        color:#f0f6fc; margin:10px 0 4px;">Custom CNN</div>
            <div class="eyebrow" style="color:#484f58;">BASELINE</div>
            <div class="sec-body" style="font-size:0.82rem; margin-top:8px;">
                4 convolutional blocks built from scratch. No pretrained weights.
                Serves as our performance lower bound.
            </div>
            <div style="margin-top:16px; padding-top:14px; border-top:1px solid #1e2d3d;">
                <div style="color:#6e7681; font-size:0.72rem;">422,530 params · 1.63 MB</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.3rem;
                            font-weight:800; color:#8b949e; margin-top:4px;">63.77%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="mcard mid">
            <div style="font-size:2rem;">⚡</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1rem; font-weight:700;
                        color:#f0f6fc; margin:10px 0 4px;">EfficientNetV2S</div>
            <div class="eyebrow">TRANSFER LEARNING</div>
            <div class="sec-body" style="font-size:0.82rem; margin-top:8px;">
                Pretrained on ImageNet. Two-stage training: frozen backbone then
                partial fine-tuning of last 2 blocks.
            </div>
            <div style="margin-top:16px; padding-top:14px; border-top:1px solid #1e2d3d;">
                <div style="color:#6e7681; font-size:0.72rem;">15.2M params · 79.1 MB</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.3rem;
                            font-weight:800; color:#3b82f6; margin-top:4px;">86.93%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="mcard best">
            <div style="font-size:2rem;">🏆</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1rem; font-weight:700;
                        color:#f0f6fc; margin:10px 0 4px;">ConvNeXtTiny</div>
            <div class="eyebrow" style="color:#10b981;">BEST MODEL ★</div>
            <div class="sec-body" style="font-size:0.82rem; margin-top:8px;">
                Modern ConvNet architecture pretrained on ImageNet.
                Outperformed all models across every evaluation metric.
            </div>
            <div style="margin-top:16px; padding-top:14px; border-top:1px solid #1e2d3d;">
                <div style="color:#6e7681; font-size:0.72rem;">15.7M params · 106.95 MB</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-size:1.3rem;
                            font-weight:800; color:#10b981; margin-top:4px;">90.66%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Tech stack
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
        <span class="tbadge">Seaborn</span><span class="tbadge">Streamlit</span>
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
        <div class="badge">EXPERIMENTAL RESULTS</div>
        <div class="display-title">Model Performance</div>
        <p class="display-sub" style="margin-top:14px;">
            Complete evaluation of all 3 models on the held-out test set.
            ConvNeXtTiny achieved best results across every metric.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    kcols = st.columns(5)
    for col, v, l in zip(kcols,
        ["90.66%","0.9544","0.8706","0.8916","4.4ms"],
        ["Best Accuracy","Best ROC-AUC","Best Macro F1","Best PR-AUC","Best Inference"]):
        with col:
            st.markdown(f"""<div class="kpi"><div class="kpi-v">{v}</div><div class="kpi-l">{l}</div></div>""",
                        unsafe_allow_html=True)

    # Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">COMPARISON TABLE</div>
    <div class="sec-title">Full Metrics Breakdown</div>
    """, unsafe_allow_html=True)

    df = pd.DataFrame({
        "Model":           ["Custom CNN","EfficientNetV2S","ConvNeXtTiny ⭐"],
        "Accuracy":        [0.6377,0.8693,0.9066],
        "Macro Precision": [0.5409,0.8318,0.8697],
        "Macro Recall":    [0.5481,0.7949,0.8636],
        "Macro F1":        [0.5499,0.8142,0.8706],
        "Weighted F1":     [0.6531,0.8674,0.9044],
        "ROC-AUC":         [0.5775,0.9025,0.9544],
        "PR-AUC":          [0.2821,0.8159,0.8916],
        "Inference (ms)":  [0.790,3.154,4.401],
        "Params (M)":      [0.42,15.22,15.67],
        "Size (MB)":       [1.63,79.10,106.95],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Category bar chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">CATEGORY ANALYSIS</div>
    <div class="sec-title">Per-Category Accuracy — ConvNeXtTiny</div>
    <div class="sec-body" style="margin-bottom:20px;">
        🟢 Green = above 90% &nbsp;|&nbsp; 🔵 Blue = 80–90% &nbsp;|&nbsp; 🔴 Red = below 80%
    </div>
    """, unsafe_allow_html=True)

    cat_data = pd.DataFrame({
        "Category": ["Carpet","Leather","Bottle","Hazelnut","Tile","Zipper",
                     "Grid","Wood","Pill","Capsule","Metal Nut","Cable",
                     "Transistor","Screw","Toothbrush"],
        "Accuracy": [1.00,0.98,0.95,0.96,0.94,0.95,
                     0.93,0.92,0.90,0.91,0.89,0.88,
                     0.87,0.82,0.70],
    })

    fig, ax = plt.subplots(figsize=(14, 4.5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    bar_colors = ['#10b981' if v >= 0.90 else '#3b82f6' if v >= 0.80 else '#ef4444'
                  for v in cat_data["Accuracy"]]
    bars = ax.bar(cat_data["Category"], cat_data["Accuracy"],
                  color=bar_colors, edgecolor='none', width=0.6)
    ax.axhline(y=cat_data["Accuracy"].mean(), color='#f59e0b', linestyle='--',
               linewidth=1.5, label=f'Mean = {cat_data["Accuracy"].mean():.2f}', alpha=0.8)
    ax.set_ylim(0, 1.12)
    ax.tick_params(colors='#6e7681', axis='both', labelsize=9)
    ax.tick_params(axis='x', rotation=30)
    for sp in ax.spines.values(): sp.set_color('#1e2d3d')
    ax.set_ylabel("Accuracy", color='#6e7681', fontsize=9)
    for bar, val in zip(bars, cat_data["Accuracy"]):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.012,
                f'{val:.0%}', ha='center', fontsize=8, color='#8b949e')
    ax.legend(facecolor='#0d1117', labelcolor='#f59e0b', edgecolor='#1e2d3d', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # RQs
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">KEY FINDINGS</div>
    <div class="sec-title">Research Questions Answered</div>
    """, unsafe_allow_html=True)

    rqs = [
        ("RQ1","Which architecture achieves strongest performance?",
         "ConvNeXtTiny — 90.66% accuracy, 0.9544 ROC-AUC, best across all metrics."),
        ("RQ2","How much does performance vary between categories?",
         "30% range — Carpet achieved 100%, Toothbrush was hardest at 70%."),
        ("RQ3","Do Grad-CAM and SHAP overlap with anomalous regions?",
         "Yes — correct predictions consistently focus on the actual defect areas."),
        ("RQ4","Which categories produce highest false-negative rates?",
         "Toothbrush and Screw — fine-grained texture defects are hardest to detect."),
        ("RQ5","Can a compact model give a practical trade-off?",
         "Custom CNN at 0.790ms/image is 5.6× faster but 27% less accurate."),
    ]
    for rq, q, a in rqs:
        st.markdown(f"""
        <div class="gcard" style="padding:18px 22px; margin-bottom:10px;">
            <div style="display:flex; gap:14px; align-items:flex-start;">
                <div style="background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.25);
                            color:#3b82f6; padding:4px 12px; border-radius:8px;
                            font-family:'JetBrains Mono',monospace; font-size:0.72rem;
                            font-weight:600; white-space:nowrap; align-self:flex-start;">{rq}</div>
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
        <div class="badge">EXPLAINABLE AI · XAI</div>
        <div class="display-title">Why Trust<br>the Model?</div>
        <p class="display-sub" style="margin-top:14px;">
            High accuracy alone is not enough. We use Grad-CAM and SHAP
            to verify the model looks at the right parts of every image.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Grad-CAM
    st.markdown("""
    <div class="eyebrow">GRAD-CAM</div>
    <div class="sec-title">Gradient-weighted Class Activation Maps</div>
    <div class="sec-body" style="margin-bottom:20px;">
        Grad-CAM uses gradients flowing into the last convolutional layer to produce
        a heatmap highlighting which image regions drove the prediction.
        Red = high influence, Blue = low influence.
    </div>
    """, unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    for col, icon, title, body in zip([g1,g2,g3],
        ["🎯","❓","⚠️"],
        ["Correct Predictions","Wrong Predictions","Limitations"],
        [
            "For correctly classified anomalous images, Grad-CAM highlights the actual defect region — scratch, crack, or contamination area.",
            "For misclassified images, Grad-CAM often reveals the model was focused on background or irrelevant regions — exposing why it failed.",
            "Grad-CAM produces coarse spatial maps and cannot pinpoint exact pixel-level regions. Use alongside SHAP for a complete picture."
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

    # SHAP
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">SHAP</div>
    <div class="sec-title">SHapley Additive exPlanations</div>
    <div class="sec-body" style="margin-bottom:20px;">
        SHAP assigns each pixel a value showing how much it contributed to the final prediction.
        Positive values (red) push toward Anomalous. Negative values (blue) push toward Normal.
    </div>
    """, unsafe_allow_html=True)

    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.markdown("""
        <div class="gcard">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <div style="width:14px; height:14px; background:#ef4444; border-radius:3px; flex-shrink:0;"></div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                            color:#f0f6fc; font-size:0.9rem;">Red — Supports Anomalous</div>
            </div>
            <div class="sec-body" style="font-size:0.82rem;">
                Red pixels push the model toward classifying the image as Anomalous.
                These typically correspond to actual defect areas — scratches, cracks,
                contamination, or structural damage visible in the product.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="gcard">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <div style="width:14px; height:14px; background:#3b82f6; border-radius:3px; flex-shrink:0;"></div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                            color:#f0f6fc; font-size:0.9rem;">Blue — Supports Normal</div>
            </div>
            <div class="sec-body" style="font-size:0.82rem;">
                Blue pixels work against the anomalous classification — they represent
                clean, defect-free regions the model recognizes as normal, pulling
                the prediction back toward the Normal class.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Why XAI matters
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">IMPORTANCE</div>
    <div class="sec-title">Why Explainability Matters</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="gcard">
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:24px;">
            <div>
                <div style="font-size:1.4rem; margin-bottom:8px;">🛡️</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                            color:#f0f6fc; font-size:0.88rem; margin-bottom:6px;">Trust</div>
                <div class="sec-body" style="font-size:0.8rem;">
                    Factory operators need to trust the system before replacing human inspectors.
                    Visual explanations build that trust by showing the reasoning.
                </div>
            </div>
            <div>
                <div style="font-size:1.4rem; margin-bottom:8px;">🐛</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                            color:#f0f6fc; font-size:0.88rem; margin-bottom:6px;">Debugging</div>
                <div class="sec-body" style="font-size:0.8rem;">
                    When the model fails, explanations reveal if it focused on
                    background patterns instead of actual product features.
                </div>
            </div>
            <div>
                <div style="font-size:1.4rem; margin-bottom:8px;">📋</div>
                <div style="font-family:'Space Grotesk',sans-serif; font-weight:600;
                            color:#f0f6fc; font-size:0.88rem; margin-bottom:6px;">Compliance</div>
                <div class="sec-body" style="font-size:0.8rem;">
                    In regulated industries, AI decisions must be auditable.
                    Grad-CAM and SHAP provide the required decision audit trail.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="infobox" style="margin-top:16px; border-color:rgba(245,158,11,0.25); background:rgba(245,158,11,0.06);">
        <strong style="color:#f59e0b;">⚠️ Important caveat:</strong>
        Attractive heatmaps are not proof of correct reasoning. A model can produce
        visually plausible Grad-CAM maps while still failing on edge cases.
        Always use explainability tools alongside rigorous quantitative evaluation.
    </div>
    """, unsafe_allow_html=True)

    footer()


# ============================================================
# PAGE 5 — ABOUT
# ============================================================
elif page == "👤 About":

    st.markdown("""
    <div class="hero">
        <div class="badge">ABOUT</div>
        <div class="display-title">Abdullah Rashid</div>
        <p class="display-sub" style="margin-top:14px;">
            3rd Semester · Bachelor of Software Engineering<br>
            University of Europe for Applied Sciences, Potsdam · SS26
        </p>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns([1,1], gap="large")

    with a1:
        st.markdown("""
        <div class="eyebrow">CONTACT</div>
        <div class="sec-title">Get in Touch</div>
        <div class="gcard" style="margin-top:12px;">
            <div style="display:flex; flex-direction:column; gap:18px;">
                <div style="display:flex; align-items:center; gap:14px;">
                    <div style="width:40px; height:40px; min-width:40px; background:rgba(59,130,246,0.1);
                                border:1px solid rgba(59,130,246,0.2); border-radius:10px;
                                display:flex; align-items:center; justify-content:center; font-size:1.1rem;">📧</div>
                    <div>
                        <div style="font-size:0.68rem; color:#6e7681; margin-bottom:3px; font-family:'JetBrains Mono',monospace; letter-spacing:0.08em;">EMAIL</div>
                        <a href="mailto:abdullahrashid130704@outlook.com"
                           style="color:#3b82f6; text-decoration:none; font-size:0.86rem;">
                            abdullahrashid130704@outlook.com
                        </a>
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:14px;">
                    <div style="width:40px; height:40px; min-width:40px; background:rgba(59,130,246,0.1);
                                border:1px solid rgba(59,130,246,0.2); border-radius:10px;
                                display:flex; align-items:center; justify-content:center; font-size:1.1rem;">📞</div>
                    <div>
                        <div style="font-size:0.68rem; color:#6e7681; margin-bottom:3px; font-family:'JetBrains Mono',monospace; letter-spacing:0.08em;">PHONE</div>
                        <div style="color:#f0f6fc; font-size:0.86rem;">+49 15 510 337 507</div>
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:14px;">
                    <div style="width:40px; height:40px; min-width:40px; background:rgba(59,130,246,0.1);
                                border:1px solid rgba(59,130,246,0.2); border-radius:10px;
                                display:flex; align-items:center; justify-content:center; font-size:1.1rem;">🔗</div>
                    <div>
                        <div style="font-size:0.68rem; color:#6e7681; margin-bottom:3px; font-family:'JetBrains Mono',monospace; letter-spacing:0.08em;">LINKEDIN</div>
                        <a href="https://linkedin.com/in/abdullahr2004" target="_blank"
                           style="color:#3b82f6; text-decoration:none; font-size:0.86rem;">
                            linkedin.com/in/abdullahr2004
                        </a>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="eyebrow">PROJECT LINKS</div>
        <div class="sec-title">Resources</div>
        <div class="gcard" style="margin-top:12px;">
            <div style="display:flex; flex-direction:column; gap:10px;">
        """, unsafe_allow_html=True)

        links = [
            ("💻","GitHub Repository","Source code, notebook, all figures",
             "https://github.com/abdullah1307-git/mvtec-anomaly-classification"),
            ("📓","Kaggle Notebook","Full executed training notebook",
             "https://www.kaggle.com/code/abdullah130704/notebook8b312b2661"),
            ("📊","MVTec AD Dataset","15 categories · 5,354 images",
             "https://www.kaggle.com/datasets/ipythonx/mvtec-ad"),
            ("🤗","Hugging Face Model","ConvNeXtTiny weights · 112 MB",
             "https://huggingface.co/abdullah130704/mvtec-anomaly-model"),
        ]
        for icon, title, sub, url in links:
            st.markdown(f"""
            <a href="{url}" target="_blank" style="text-decoration:none; display:block;">
                <div style="display:flex; align-items:center; gap:14px; padding:12px 14px;
                            background:rgba(255,255,255,0.03); border:1px solid #1e2d3d;
                            border-radius:10px; transition:border-color 0.2s; margin-bottom:8px;">
                    <div style="font-size:1.3rem;">{icon}</div>
                    <div>
                        <div style="color:#f0f6fc; font-size:0.86rem; font-weight:500;">{title}</div>
                        <div style="color:#6e7681; font-size:0.74rem; margin-top:2px;">{sub}</div>
                    </div>
                    <div style="margin-left:auto; color:#484f58; font-size:0.8rem;">↗</div>
                </div>
            </a>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    # Course info
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">ACADEMIC CONTEXT</div>
    <div class="sec-title">Course Information</div>
    <div class="gcard" style="margin-top:12px;">
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:20px;">
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem;
                            color:#484f58; letter-spacing:0.1em; margin-bottom:6px;">COURSE</div>
                <div style="color:#f0f6fc; font-size:0.85rem; font-weight:500;">Machine Learning & Smart Systems</div>
            </div>
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem;
                            color:#484f58; letter-spacing:0.1em; margin-bottom:6px;">UNIVERSITY</div>
                <div style="color:#f0f6fc; font-size:0.85rem; font-weight:500;">UE for Applied Sciences, Potsdam</div>
            </div>
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem;
                            color:#484f58; letter-spacing:0.1em; margin-bottom:6px;">SEMESTER</div>
                <div style="color:#f0f6fc; font-size:0.85rem; font-weight:500;">Summer Semester 2026 · 3rd</div>
            </div>
            <div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem;
                            color:#484f58; letter-spacing:0.1em; margin-bottom:6px;">PROGRAMME</div>
                <div style="color:#f0f6fc; font-size:0.85rem; font-weight:500;">Bachelor of Software Engineering</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    footer()
