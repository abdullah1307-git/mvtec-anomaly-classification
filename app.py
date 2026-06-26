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

# ---- Page Configuration ----
st.set_page_config(
    page_title="AnomalyVision",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0d1424 100%);
    }

    [data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid #2d3748;
    }

    [data-testid="stSidebar"] * { color: #f1f5f9 !important; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .hero-container {
        background: linear-gradient(135deg, #1a2235 0%, #0f172a 100%);
        border: 1px solid #2d3748;
        border-radius: 16px;
        padding: 40px;
        margin-bottom: 30px;
        text-align: center;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-top: 12px;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #3b82f6;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 16px;
        letter-spacing: 0.05em;
    }

    .card {
        background: #1a2235;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }

    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 16px;
    }

    .result-normal {
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }

    .result-anomalous {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }

    .result-label { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; margin: 0; }
    .result-confidence { font-size: 1rem; color: #94a3b8; margin-top: 8px; }

    .metric-card {
        background: #1a2235;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    .metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; color: #3b82f6; }
    .metric-label { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }

    .sample-img-card {
        background: #1a2235;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        cursor: pointer;
        transition: border-color 0.2s;
    }

    .sample-img-card:hover { border-color: #3b82f6; }

    .sample-normal { border-top: 3px solid #10b981; }
    .sample-anomaly { border-top: 3px solid #ef4444; }

    .footer {
        background: #111827;
        border-top: 1px solid #2d3748;
        border-radius: 12px;
        padding: 30px;
        margin-top: 40px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .footer-left { color: #94a3b8; font-size: 0.85rem; }
    .footer-right { text-align: right; }
    .footer-name { font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 600; color: #f1f5f9; }
    .footer-info { font-size: 0.82rem; color: #94a3b8; margin-top: 4px; line-height: 1.6; }
    .footer-link { color: #3b82f6; text-decoration: none; }

    .info-box {
        background: rgba(59,130,246,0.08);
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 10px;
        padding: 16px;
        color: #94a3b8;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    .section-header { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700; color: #f1f5f9; margin-bottom: 8px; }
    .section-sub { color: #94a3b8; font-size: 0.9rem; margin-bottom: 24px; }
    .category-pill { display: inline-block; background: rgba(6,182,212,0.1); border: 1px solid rgba(6,182,212,0.25); color: #06b6d4; padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; margin: 3px; }

    h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }
    p, li { color: #94a3b8; }
    .stMarkdown { color: #94a3b8; }
    .stProgress > div > div { background: linear-gradient(90deg, #3b82f6, #06b6d4) !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================
HF_BASE = "https://huggingface.co/abdullah130704/mvtec-anomaly-model/resolve/main"
MODEL_URL = f"{HF_BASE}/best_model_weights.pth"
SAMPLE_BASE = f"{HF_BASE}/sample_images"

CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ============================================================
# MODEL DEFINITION
# ============================================================
class ConvNeXtTinyModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNeXtTinyModel, self).__init__()
        self.backbone = convnext_tiny(weights=None)
        num_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
@st.cache_resource
def load_model():
    model_path = Path("best_model_weights.pth")
    if not model_path.exists():
        with st.spinner("Downloading model weights... (first time only, ~112MB)"):
            response = requests.get(MODEL_URL, stream=True)
            if response.status_code == 200:
                with open(model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error("Could not download model. Please try again later.")
                return None
    device = torch.device("cpu")
    model = ConvNeXtTinyModel(num_classes=2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(pil_image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform(pil_image.convert("RGB")).unsqueeze(0)


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
    std  = torch.tensor(IMAGENET_STD).view(3,1,1)
    img  = tensor.cpu() * std + mean
    img  = img.permute(1,2,0).numpy()
    return np.clip(img, 0, 1)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.saved_gradients = None
        self.saved_activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.saved_activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.saved_gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=1):
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, target_class].backward()
        weights = self.saved_gradients.mean(dim=[2,3], keepdim=True)
        cam = (weights * self.saved_activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cv2.resize(cam, (IMG_SIZE, IMG_SIZE))


def create_gradcam_figure(original_pil, heatmap):
    img_np = np.array(original_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = np.clip(0.4 * colored + 0.6 * img_np, 0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#1a2235')
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', color='#f1f5f9', fontsize=11, pad=10)
    axes[0].axis('off')
    axes[1].imshow(overlay)
    axes[1].set_title('Grad-CAM Heatmap\n(Red = Model Focus)', color='#f1f5f9', fontsize=11, pad=10)
    axes[1].axis('off')
    plt.tight_layout(pad=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a2235', edgecolor='none')
    buf.seek(0)
    plt.close()
    return buf


def predict(model, image_tensor, original_pil):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[predicted_class].item()
    prob_normal    = probabilities[0].item()
    prob_anomalous = probabilities[1].item()
    target_layer = list(model.backbone.features.children())[-1]
    gradcam = GradCAM(model, target_layer)
    image_tensor_grad = image_tensor.clone().requires_grad_(True)
    heatmap = gradcam.generate(image_tensor_grad, target_class=predicted_class)
    gradcam_fig = create_gradcam_figure(original_pil, heatmap)
    return predicted_class, confidence, prob_normal, prob_anomalous, gradcam_fig


@st.cache_data
def load_sample_image_from_hf(category, filename):
    url = f"{SAMPLE_BASE}/{category}/{filename}"
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    return None


def get_sample_files_for_category(category):
    sample_files = {
        'bottle':     ['normal_01.png', 'anomaly_01_broken_large.png', 'anomaly_02_broken_small.png'],
        'cable':      ['normal_01.png', 'anomaly_01_bent_wire.png',    'anomaly_02_cable_swap.png'],
        'capsule':    ['normal_01.png', 'anomaly_01_crack.png',        'anomaly_02_faulty_imprint.png'],
        'carpet':     ['normal_01.png', 'anomaly_01_color.png',        'anomaly_02_cut.png'],
        'grid':       ['normal_01.png', 'anomaly_01_bent.png',         'anomaly_02_broken.png'],
        'hazelnut':   ['normal_01.png', 'anomaly_01_crack.png',        'anomaly_02_cut.png'],
        'leather':    ['normal_01.png', 'anomaly_01_color.png',        'anomaly_02_cut.png'],
        'metal_nut':  ['normal_01.png', 'anomaly_01_bent.png',         'anomaly_02_color.png'],
        'pill':       ['normal_01.png', 'anomaly_01_color.png',        'anomaly_02_combined.png'],
        'screw':      ['normal_01.png', 'anomaly_01_manipulated_front.png', 'anomaly_02_scratch_head.png'],
        'tile':       ['normal_01.png', 'anomaly_01_crack.png',        'anomaly_02_glue_strip.png'],
        'toothbrush': ['normal_01.png', 'anomaly_01_defective.png',    None],
        'transistor': ['normal_01.png', 'anomaly_01_bent_lead.png',    'anomaly_02_cut_lead.png'],
        'wood':       ['normal_01.png', 'anomaly_01_color.png',        'anomaly_02_combined.png'],
        'zipper':     ['normal_01.png', 'anomaly_01_broken_teeth.png', 'anomaly_02_combined.png'],
    }
    return sample_files.get(category, ['normal_01.png', None, None])


def render_footer():
    st.markdown("""
    <div class='footer'>
        <div class='footer-left'>
            <strong style='color: #f1f5f9;'>AnomalyVision</strong><br>
            Machine Learning & Smart Systems — Final Project<br>
            University of Europe for Applied Sciences, Potsdam
        </div>
        <div class='footer-right'>
            <div class='footer-name'>Abdullah Rashid</div>
            <div class='footer-info'>
                📧 <a href='mailto:abdullahrashid130704@outlook.com' class='footer-link'>abdullahrashid130704@outlook.com</a><br>
                📞 +49 15 510 337 507<br>
                🔗 <a href='https://linkedin.com/in/abdullahr2004' class='footer-link' target='_blank'>linkedin.com/in/abdullahr2004</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0;'>
        <div style='font-family: Space Grotesk; font-size: 1.4rem; font-weight: 700;
                    background: linear-gradient(135deg, #3b82f6, #06b6d4);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text;'>
            🔍 AnomalyVision
        </div>
        <div style='font-size: 0.75rem; color: #64748b; margin-top: 4px;'>
            Industrial Defect Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #2d3748;'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Live Demo", "📊 About the Project", "📈 Model Results"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color: #2d3748;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size: 0.78rem; color: #64748b; line-height: 1.8;'>
        <div style='font-weight: 600; color: #94a3b8; margin-bottom: 8px;'>MODEL INFO</div>
        <div>🏆 Best Model: ConvNeXtTiny</div>
        <div>✅ Accuracy: 90.66%</div>
        <div>📊 ROC-AUC: 0.9544</div>
        <div>⚡ Inference: 4.4ms</div>
        <div>📦 Categories: 15</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 1: LIVE DEMO
# ============================================================
if page == "🏠 Live Demo":

    st.markdown("""
    <div class='hero-container'>
        <div class='hero-badge'>POWERED BY CONVNEXTTINY + GRAD-CAM</div>
        <h1 class='hero-title'>AnomalyVision</h1>
        <p class='hero-subtitle'>
            Upload an industrial product image and get an instant AI-powered defect detection result.<br>
            Our model classifies images as <strong style='color:#10b981'>Normal</strong> or
            <strong style='color:#ef4444'>Anomalous</strong> with explainable heatmaps.
        </p>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("Model could not be loaded. Please refresh the page.")
        st.stop()

    # ---- Input Section ----
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<div class='card-title'>UPLOAD YOUR OWN IMAGE</div>", unsafe_allow_html=True)

        category = st.selectbox(
            "Select Product Category",
            [c.replace('_', ' ').title() for c in CATEGORIES],
            help="Select the type of industrial product in your image"
        )

        uploaded_file = st.file_uploader(
            "Drop your image here",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )

        st.markdown("""
        <div class='info-box' style='margin-top: 16px;'>
            <strong style='color: #3b82f6;'>How it works:</strong><br>
            1. Select the product category<br>
            2. Upload your own image OR use a sample below<br>
            3. Click Analyze to get the AI prediction<br>
            4. View Grad-CAM heatmap showing model focus areas
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card-title'>IMAGE PREVIEW</div>", unsafe_allow_html=True)
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
        else:
            st.markdown("""
            <div style='background: #1a2235; border: 2px dashed #2d3748; border-radius: 12px;
                        padding: 60px 20px; text-align: center; color: #475569;'>
                <div style='font-size: 3rem;'>🖼️</div>
                <div style='margin-top: 12px; font-size: 0.9rem;'>
                    Upload your image or select a sample below
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ---- Sample Images Section ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>OR TRY A SAMPLE IMAGE</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box' style='margin-bottom: 16px;'>
        <strong style='color: #3b82f6;'>No image? No problem!</strong>
        Select a category below to browse sample images.
        Green border = Normal | Red border = Anomalous.
        Click any image to load it for analysis.
    </div>
    """, unsafe_allow_html=True)

    selected_sample_cat = st.selectbox(
        "Browse samples by category",
        [c.replace('_', ' ').title() for c in CATEGORIES],
        key="sample_cat_selector"
    )

    sample_cat_key = selected_sample_cat.lower().replace(' ', '_')
    sample_files = get_sample_files_for_category(sample_cat_key)
    sample_labels = ['Normal', 'Anomaly Type 1', 'Anomaly Type 2']
    sample_colors = ['#10b981', '#ef4444', '#ef4444']

    sample_cols = st.columns(3)
    selected_sample_image = None
    selected_sample_label = None

    for i, (filename, label, color) in enumerate(zip(sample_files, sample_labels, sample_colors)):
        with sample_cols[i]:
            if filename is not None:
                sample_img = load_sample_image_from_hf(sample_cat_key, filename)
                if sample_img is not None:
                    st.image(sample_img, use_column_width=True)
                    st.markdown(f"""
                    <div style='text-align:center; color: {color}; font-size: 0.82rem;
                                font-weight: 600; margin-top: 4px;'>{label}</div>
                    """, unsafe_allow_html=True)
                    if st.button(f"Use this image", key=f"sample_btn_{i}"):
                        selected_sample_image = sample_img
                        selected_sample_label = label
            else:
                st.markdown("""
                <div style='background: #1a2235; border: 1px dashed #2d3748; border-radius: 8px;
                            padding: 40px 10px; text-align: center; color: #475569; font-size: 0.8rem;'>
                    No second anomaly type for this category
                </div>
                """, unsafe_allow_html=True)

    # ---- Determine which image to analyze ----
    analyze_image = None
    analyze_label = None

    if selected_sample_image is not None:
        analyze_image = selected_sample_image
        analyze_label = f"Sample: {selected_sample_cat} ({selected_sample_label})"
        st.success(f"Sample image loaded: {selected_sample_cat} - {selected_sample_label}")

    elif uploaded_file is not None:
        analyze_image = Image.open(uploaded_file).convert("RGB")
        analyze_label = uploaded_file.name

    # ---- Analyze Button ----
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col2:
        analyze_btn = st.button(
            "🔍 Analyze Image",
            use_container_width=True,
            disabled=(analyze_image is None)
        )

    # ---- Results ----
    if analyze_btn and analyze_image is not None:
        with st.spinner("Running AI analysis..."):
            image_tensor = preprocess_image(analyze_image)
            pred_class, confidence, prob_normal, prob_anom, gradcam_fig = predict(
                model, image_tensor, analyze_image
            )

        st.markdown("<hr style='border-color: #2d3748; margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Analysis Results</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-sub'>Image: {analyze_label} | Model: ConvNeXtTiny</div>",
                    unsafe_allow_html=True)

        res_col1, res_col2, res_col3 = st.columns([1, 1, 1])

        with res_col1:
            if pred_class == 0:
                st.markdown("""
                <div class='result-normal'>
                    <div style='font-size:2.5rem;'>✅</div>
                    <p class='result-label' style='color:#10b981;'>NORMAL</p>
                    <p class='result-confidence'>No defect detected</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='result-anomalous'>
                    <div style='font-size:2.5rem;'>⚠️</div>
                    <p class='result-label' style='color:#ef4444;'>ANOMALOUS</p>
                    <p class='result-confidence'>Defect detected</p>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("<div class='card-title'>CONFIDENCE SCORES</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='margin-bottom:6px;'>
                <div style='display:flex; justify-content:space-between;'>
                    <span style='color:#10b981; font-size:0.88rem; font-weight:600;'>Normal</span>
                    <span style='color:#f1f5f9; font-size:0.88rem;'>{prob_normal*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(prob_normal)

            st.markdown(f"""
            <div style='margin-top:16px; margin-bottom:6px;'>
                <div style='display:flex; justify-content:space-between;'>
                    <span style='color:#ef4444; font-size:0.88rem; font-weight:600;'>Anomalous</span>
                    <span style='color:#f1f5f9; font-size:0.88rem;'>{prob_anom*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(prob_anom)

            st.markdown(f"""
            <div class='metric-card' style='margin-top:20px;'>
                <div class='metric-value'>{confidence*100:.1f}%</div>
                <div class='metric-label'>Overall Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col3:
            st.markdown("<div class='card-title'>MODEL STATS</div>", unsafe_allow_html=True)
            st.markdown("""
            <div class='metric-card' style='margin-bottom:12px;'>
                <div class='metric-value' style='font-size:1.1rem;'>ConvNeXtTiny</div>
                <div class='metric-label'>Model Used</div>
            </div>
            <div class='metric-card' style='margin-bottom:12px;'>
                <div class='metric-value'>90.66%</div>
                <div class='metric-label'>Model Accuracy</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>0.9544</div>
                <div class='metric-label'>ROC-AUC Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>GRAD-CAM EXPLANATION</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box' style='margin-bottom:16px;'>
            <strong style='color:#3b82f6;'>What is Grad-CAM?</strong>
            It shows <em>which regions of the image the model focused on</em> when making its decision.
            🔴 Red/Yellow = High attention | 🔵 Blue = Low attention
        </div>
        """, unsafe_allow_html=True)
        st.image(gradcam_fig, use_column_width=True)

    render_footer()


# ============================================================
# PAGE 2: ABOUT THE PROJECT
# ============================================================
elif page == "📊 About the Project":

    st.markdown("""
    <div class='hero-container'>
        <div class='hero-badge'>RESEARCH PROJECT</div>
        <h1 class='hero-title'>About the Project</h1>
        <p class='hero-subtitle'>
            Explainable Multi-Product Industrial Anomaly Classification<br>
            Using CNN Transfer Learning and Cross-Category Evaluation
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Project Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <p style='color:#94a3b8; line-height:1.8; font-size:0.95rem;'>
        This project addresses the real-world problem of automated industrial quality control.
        In manufacturing, detecting product defects manually is slow, expensive, and error-prone.
        This system uses deep learning to automatically classify industrial product images as
        <strong style='color:#10b981;'>Normal</strong> or
        <strong style='color:#ef4444;'>Anomalous (defective)</strong>,
        covering 15 different product categories from the MVTec AD benchmark dataset.
        </p>
        <p style='color:#94a3b8; line-height:1.8; font-size:0.95rem; margin-top:12px;'>
        Beyond just making predictions, the system provides
        <strong style='color:#3b82f6;'>visual explanations</strong> using Grad-CAM and SHAP,
        showing exactly which parts of the image led to each decision.
        This makes the system trustworthy and deployable in real industrial settings.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Dataset</div>", unsafe_allow_html=True)
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>DATASET DETAILS</div>
            <table style='width:100%; color:#94a3b8; font-size:0.88rem; border-collapse:collapse;'>
                <tr style='border-bottom:1px solid #2d3748;'>
                    <td style='padding:10px 0; color:#64748b;'>Name</td>
                    <td style='padding:10px 0; color:#f1f5f9; font-weight:500;'>MVTec AD</td>
                </tr>
                <tr style='border-bottom:1px solid #2d3748;'>
                    <td style='padding:10px 0; color:#64748b;'>Categories</td>
                    <td style='padding:10px 0; color:#f1f5f9; font-weight:500;'>15 product types</td>
                </tr>
                <tr style='border-bottom:1px solid #2d3748;'>
                    <td style='padding:10px 0; color:#64748b;'>Total Images</td>
                    <td style='padding:10px 0; color:#f1f5f9; font-weight:500;'>5,000+</td>
                </tr>
                <tr style='border-bottom:1px solid #2d3748;'>
                    <td style='padding:10px 0; color:#64748b;'>Task</td>
                    <td style='padding:10px 0; color:#f1f5f9; font-weight:500;'>Binary Classification</td>
                </tr>
                <tr>
                    <td style='padding:10px 0; color:#64748b;'>Split</td>
                    <td style='padding:10px 0; color:#f1f5f9; font-weight:500;'>60% / 20% / 20%</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with d_col2:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>15 PRODUCT CATEGORIES</div>
            <div style='margin-top:8px;'>
                <span class='category-pill'>Bottle</span>
                <span class='category-pill'>Cable</span>
                <span class='category-pill'>Capsule</span>
                <span class='category-pill'>Carpet</span>
                <span class='category-pill'>Grid</span>
                <span class='category-pill'>Hazelnut</span>
                <span class='category-pill'>Leather</span>
                <span class='category-pill'>Metal Nut</span>
                <span class='category-pill'>Pill</span>
                <span class='category-pill'>Screw</span>
                <span class='category-pill'>Tile</span>
                <span class='category-pill'>Toothbrush</span>
                <span class='category-pill'>Transistor</span>
                <span class='category-pill'>Wood</span>
                <span class='category-pill'>Zipper</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Models Compared</div>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3 = st.columns(3)

    with m_col1:
        st.markdown("""
        <div class='card' style='border-color:#4b5563;'>
            <div style='font-size:1.8rem; margin-bottom:12px;'>🏗️</div>
            <div style='font-family:Space Grotesk; font-size:1rem; font-weight:600; color:#f1f5f9; margin-bottom:8px;'>Custom CNN</div>
            <div style='color:#64748b; font-size:0.78rem; margin-bottom:12px;'>BASELINE MODEL</div>
            <div style='color:#94a3b8; font-size:0.85rem; line-height:1.6;'>Built from scratch with 4 convolutional blocks. No pretrained weights. Used as our performance baseline.</div>
            <div style='margin-top:16px; padding-top:16px; border-top:1px solid #2d3748;'>
                <div style='color:#94a3b8; font-size:0.82rem;'>422,530 parameters</div>
                <div style='color:#f1f5f9; font-size:1.1rem; font-weight:600; margin-top:4px;'>63.77% accuracy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown("""
        <div class='card' style='border-color:#3b82f6;'>
            <div style='font-size:1.8rem; margin-bottom:12px;'>⚡</div>
            <div style='font-family:Space Grotesk; font-size:1rem; font-weight:600; color:#f1f5f9; margin-bottom:8px;'>EfficientNetV2S</div>
            <div style='color:#3b82f6; font-size:0.78rem; margin-bottom:12px;'>TRANSFER LEARNING</div>
            <div style='color:#94a3b8; font-size:0.85rem; line-height:1.6;'>Pretrained on ImageNet. Fine-tuned using two-stage training strategy.</div>
            <div style='margin-top:16px; padding-top:16px; border-top:1px solid #2d3748;'>
                <div style='color:#94a3b8; font-size:0.82rem;'>15,221,034 parameters</div>
                <div style='color:#3b82f6; font-size:1.1rem; font-weight:600; margin-top:4px;'>86.93% accuracy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown("""
        <div class='card' style='border-color:#10b981;'>
            <div style='font-size:1.8rem; margin-bottom:12px;'>🏆</div>
            <div style='font-family:Space Grotesk; font-size:1rem; font-weight:600; color:#f1f5f9; margin-bottom:8px;'>ConvNeXtTiny</div>
            <div style='color:#10b981; font-size:0.78rem; margin-bottom:12px;'>BEST MODEL</div>
            <div style='color:#94a3b8; font-size:0.85rem; line-height:1.6;'>Pretrained on ImageNet. Achieved best results across all metrics. Used for live inference.</div>
            <div style='margin-top:16px; padding-top:16px; border-top:1px solid #2d3748;'>
                <div style='color:#94a3b8; font-size:0.82rem;'>15,670,018 parameters</div>
                <div style='color:#10b981; font-size:1.1rem; font-weight:600; margin-top:4px;'>90.66% accuracy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><div class='section-header'>Methodology Pipeline</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <div style='display:flex; gap:12px; flex-wrap:wrap; margin-top:8px;'>
            <div style='background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.3); border-radius:8px; padding:14px 20px; flex:1; min-width:120px; text-align:center;'>
                <div style='color:#3b82f6; font-weight:700; font-size:1.2rem;'>1</div>
                <div style='color:#f1f5f9; font-size:0.85rem; margin-top:4px; font-weight:500;'>Data Loading</div>
                <div style='color:#64748b; font-size:0.75rem; margin-top:4px;'>15 categories</div>
            </div>
            <div style='background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.3); border-radius:8px; padding:14px 20px; flex:1; min-width:120px; text-align:center;'>
                <div style='color:#3b82f6; font-weight:700; font-size:1.2rem;'>2</div>
                <div style='color:#f1f5f9; font-size:0.85rem; margin-top:4px; font-weight:500;'>Preprocessing</div>
                <div style='color:#64748b; font-size:0.75rem; margin-top:4px;'>Resize, normalize</div>
            </div>
            <div style='background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.3); border-radius:8px; padding:14px 20px; flex:1; min-width:120px; text-align:center;'>
                <div style='color:#3b82f6; font-weight:700; font-size:1.2rem;'>3</div>
                <div style='color:#f1f5f9; font-size:0.85rem; margin-top:4px; font-weight:500;'>Training</div>
                <div style='color:#64748b; font-size:0.75rem; margin-top:4px;'>3 CNN models</div>
            </div>
            <div style='background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.3); border-radius:8px; padding:14px 20px; flex:1; min-width:120px; text-align:center;'>
                <div style='color:#3b82f6; font-weight:700; font-size:1.2rem;'>4</div>
                <div style='color:#f1f5f9; font-size:0.85rem; margin-top:4px; font-weight:500;'>Evaluation</div>
                <div style='color:#64748b; font-size:0.75rem; margin-top:4px;'>F1, AUC, CM</div>
            </div>
            <div style='background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.3); border-radius:8px; padding:14px 20px; flex:1; min-width:120px; text-align:center;'>
                <div style='color:#3b82f6; font-weight:700; font-size:1.2rem;'>5</div>
                <div style='color:#f1f5f9; font-size:0.85rem; margin-top:4px; font-weight:500;'>Explainability</div>
                <div style='color:#64748b; font-size:0.75rem; margin-top:4px;'>Grad-CAM + SHAP</div>
            </div>
            <div style='background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); border-radius:8px; padding:14px 20px; flex:1; min-width:120px; text-align:center;'>
                <div style='color:#10b981; font-weight:700; font-size:1.2rem;'>6</div>
                <div style='color:#f1f5f9; font-size:0.85rem; margin-top:4px; font-weight:500;'>Deployment</div>
                <div style='color:#64748b; font-size:0.75rem; margin-top:4px;'>Streamlit app</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_footer()


# ============================================================
# PAGE 3: MODEL RESULTS
# ============================================================
elif page == "📈 Model Results":

    st.markdown("""
    <div class='hero-container'>
        <div class='hero-badge'>EXPERIMENTAL RESULTS</div>
        <h1 class='hero-title'>Model Results</h1>
        <p class='hero-subtitle'>
            Complete performance comparison of all 3 CNN models<br>
            evaluated on the MVTec AD test set
        </p>
    </div>
    """, unsafe_allow_html=True)

    k_col1, k_col2, k_col3, k_col4 = st.columns(4)
    with k_col1:
        st.markdown("""<div class='metric-card'><div class='metric-value'>90.66%</div><div class='metric-label'>Best Accuracy</div></div>""", unsafe_allow_html=True)
    with k_col2:
        st.markdown("""<div class='metric-card'><div class='metric-value'>0.9544</div><div class='metric-label'>Best ROC-AUC</div></div>""", unsafe_allow_html=True)
    with k_col3:
        st.markdown("""<div class='metric-card'><div class='metric-value'>0.8706</div><div class='metric-label'>Best Macro F1</div></div>""", unsafe_allow_html=True)
    with k_col4:
        st.markdown("""<div class='metric-card'><div class='metric-value'>0.790ms</div><div class='metric-label'>Fastest Inference</div></div>""", unsafe_allow_html=True)

    st.markdown("<br><div class='section-header'>Full Comparison Table</div>", unsafe_allow_html=True)

    import pandas as pd
    results_data = {
        "Model":           ["Custom CNN", "EfficientNetV2S", "ConvNeXtTiny ⭐"],
        "Accuracy":        ["0.6377",     "0.8693",          "0.9066"],
        "Macro Precision": ["0.5409",     "0.8318",          "0.8697"],
        "Macro Recall":    ["0.5481",     "0.7949",          "0.8636"],
        "Macro F1":        ["0.5499",     "0.8142",          "0.8706"],
        "Weighted F1":     ["0.6531",     "0.8674",          "0.9044"],
        "ROC-AUC":         ["0.5775",     "0.9025",          "0.9544"],
        "PR-AUC":          ["0.2821",     "0.8159",          "0.8916"],
        "Inference (ms)":  ["0.790",      "3.154",           "4.401"],
        "Parameters":      ["422,530",    "15,221,034",      "15,670,018"],
        "Size (MB)":       ["1.63",       "79.10",           "106.95"],
    }
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.markdown("<br><div class='section-header'>Research Questions</div>", unsafe_allow_html=True)
    rq_data = [
        ("RQ1", "Which architecture achieves the strongest performance?",
         "ConvNeXtTiny with 90.66% accuracy and 0.9544 ROC-AUC"),
        ("RQ2", "How much does performance vary between categories?",
         "Range of 0.30 — best: Carpet (100%), worst: Toothbrush (70%)"),
        ("RQ3", "Do Grad-CAM and SHAP overlap with anomalous regions?",
         "Yes — correct predictions show focus on actual defect areas"),
        ("RQ4", "Which categories have highest false-negative rates?",
         "Toothbrush and Screw showed highest confusion rates"),
        ("RQ5", "Can a compact model give practical trade-off?",
         "Custom CNN at 0.790ms is fastest but sacrifices accuracy"),
    ]

    for rq, question, answer in rq_data:
        st.markdown(f"""
        <div class='card' style='margin-bottom:12px;'>
            <div style='display:flex; gap:16px; align-items:flex-start;'>
                <div style='background:rgba(59,130,246,0.15); border:1px solid rgba(59,130,246,0.3);
                            color:#3b82f6; padding:6px 12px; border-radius:8px; font-weight:700;
                            font-size:0.85rem; white-space:nowrap;'>{rq}</div>
                <div>
                    <div style='color:#f1f5f9; font-size:0.9rem; font-weight:500;'>{question}</div>
                    <div style='color:#10b981; font-size:0.85rem; margin-top:6px;'>→ {answer}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_footer()
