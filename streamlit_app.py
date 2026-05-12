"""
FairVision - Age Group Classification Demo
Streamlit Web Application
IJSE | Certified AI & ML Engineer | 2025/2026
"""

import os
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="FairVision - Age Classification",
    page_icon="👁️",
    layout="centered"
)

# ==========================================
# MODEL URLS (optional - only if not in same folder)
# ==========================================
MODEL_URLS = {
    "baseline_model.pth": os.getenv("URL_BASELINE", ""),
    "model_weighted.pth":  os.getenv("URL_WEIGHTED", ""),
    "model_balanced.pth":  os.getenv("URL_BALANCED", ""),
}


@st.cache_resource(show_spinner=False)
def download_model(filename: str) -> bool:
    """Download model from URL if not already present locally."""
    if os.path.exists(filename):
        return True
    url = MODEL_URLS.get(filename, "")
    if not url:
        return False
    try:
        import requests
        with st.spinner(f"Downloading {filename}..."):
            r = requests.get(url, timeout=180, stream=True)
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Download failed for {filename}: {e}")
        return False


# ==========================================
# CNN MODEL — matches FairFaceCNN in notebook exactly
# 4 blocks + GAP + FC(256->512->9)
# ==========================================
class FairVisionCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(FairVisionCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


AGE_NAMES = ['0-2', '3-9', '10-19', '20-29', '30-39',
             '40-49', '50-59', '60-69', '70+']


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load model weights. Returns (model, is_loaded, message)."""
    model = FairVisionCNN(num_classes=9)
    available = download_model(model_path)

    if not available:
        return model, False, "Model file not found. Running in demo mode with random predictions."

    try:
        state = torch.load(model_path, map_location="cpu")
        # Handle checkpoint format saved with extra keys
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
        return model, True, "Model loaded successfully."
    except Exception as e:
        return model, False, f"Could not load weights ({e}). Running in demo mode."


# ==========================================
# IMAGE TRANSFORM (matches notebook eval_transform)
# ==========================================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================================
# UI HEADER
# ==========================================
st.title("FairVision")
st.subheader("CNN-Based Age Group Classification System")

st.markdown("""
**About this system:**
FairVision is a CNN-based face analytics prototype trained from scratch on the
[FairFace dataset](https://huggingface.co/datasets/HuggingFaceM4/FairFace).
It predicts the **age group** of a person from a face image across **9 categories**.

This demo was developed as part of the IJSE Certified AI & ML Engineer programme.
The system includes three models: a baseline CNN and two bias-mitigated versions
using class-weighted loss and balanced mini-batch sampling.
""")

st.divider()

# ==========================================
# MODEL SELECTOR
# ==========================================
st.subheader("Model Selection")

model_choice = st.selectbox(
    "Choose which trained model to use for inference:",
    ["Baseline Model", "Weighted Loss Model", "Balanced Sampling Model"],
)

model_files = {
    "Baseline Model":          "baseline_model.pth",
    "Weighted Loss Model":     "model_weighted.pth",
    "Balanced Sampling Model": "model_balanced.pth",
}

model_descriptions = {
    "Baseline Model": (
        "**Baseline** — Standard CNN trained with CrossEntropyLoss and no fairness intervention. "
        "Achieves the highest overall accuracy but shows the largest demographic performance gap."
    ),
    "Weighted Loss Model": (
        "**Weighted Loss** — CNN trained with inverse-frequency class weights to reduce age-group "
        "imbalance. Improves minority class performance at a small overall accuracy cost."
    ),
    "Balanced Sampling Model": (
        "**Balanced Sampling** — CNN trained with WeightedRandomSampler ensuring equal age-group "
        "representation per mini-batch. Best worst-group accuracy; most equitable across demographics."
    ),
}

selected_file = model_files[model_choice]
st.info(model_descriptions[model_choice])

with st.spinner("Loading model..."):
    model, model_loaded, load_msg = load_model(selected_file)

if model_loaded:
    st.success(f"✅ {load_msg}")
else:
    st.warning(f"⚠️ {load_msg}")

st.divider()

# ==========================================
# IMAGE UPLOAD & INFERENCE
# ==========================================
st.subheader("Upload a Face Image")
st.caption("Upload a clear, frontal face image (JPG or PNG). The model accepts images similar to FairFace.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("Prediction Results")

        with st.spinner("Running inference..."):
            if model_loaded:
                img_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    logits = model(img_tensor)
                    probs = torch.softmax(logits, dim=1).squeeze().numpy()
            else:
                # Demo mode: random plausible distribution
                raw = np.random.dirichlet(np.ones(9) * 0.5)
                probs = raw / raw.sum()

        top3_idx = np.argsort(probs)[::-1][:3]
        bar_colors = ["#1a9850", "#4575b4", "#d73027"]

        st.markdown("**Top 3 Predicted Age Groups:**")
        for rank, idx in enumerate(top3_idx):
            label = AGE_NAMES[idx]
            prob = probs[idx]
            color = bar_colors[rank]
            st.markdown(f"""
            <div style="margin-bottom:14px;">
                <span style="font-weight:bold;">#{rank+1} &nbsp; Age {label}</span>
                <span style="float:right; font-weight:bold;">{prob*100:.1f}%</span>
                <div style="background:#e0e0e0; border-radius:6px; height:20px;
                            margin-top:6px; clear:both; overflow:hidden;">
                    <div style="background:{color}; width:{max(prob*100, 2):.1f}%;
                                height:20px; border-radius:6px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.success(f"**Predicted Age Group: {AGE_NAMES[top3_idx[0]]}**")

        if not model_loaded:
            st.caption("Demo mode active — predictions are illustrative only (model not loaded).")

    st.divider()

    # Full probability bar chart
    st.subheader("Full Probability Distribution")
    prob_df = pd.DataFrame({"Age Group": AGE_NAMES, "Probability": probs})
    st.bar_chart(prob_df.set_index("Age Group")["Probability"])

    with st.expander("View raw probabilities table"):
        prob_df["Probability (%)"] = (prob_df["Probability"] * 100).round(2)
        st.dataframe(
            prob_df[["Age Group", "Probability (%)"]].set_index("Age Group"),
            use_container_width=True
        )

st.divider()

# ==========================================
# MODEL ARCHITECTURE INFO
# ==========================================
with st.expander("Model Architecture Details"):
    st.markdown("""
    **FairVisionCNN** — Custom 4-block CNN trained entirely from scratch using PyTorch.

    | Component | Configuration |
    |-----------|--------------|
    | Block 1 | Conv(3→32)×2, BatchNorm, ReLU, MaxPool, Dropout2d(0.25) |
    | Block 2 | Conv(32→64)×2, BatchNorm, ReLU, MaxPool, Dropout2d(0.25) |
    | Block 3 | Conv(64→128)×2, BatchNorm, ReLU, MaxPool, Dropout2d(0.25) |
    | Block 4 | Conv(128→256), BatchNorm, ReLU, MaxPool, Dropout2d(0.25) |
    | GAP | AdaptiveAvgPool2d(1) |
    | Classifier | FC(256→512), ReLU, Dropout(0.5), FC(512→9) |
    | Input size | 3 × 64 × 64 RGB |
    | Output | 9 age-group logits |

    **Training setup:** AdamW (lr=0.001, weight_decay=1e-4), ReduceLROnPlateau,
    batch size 64. No pretrained weights used.

    **Dataset:** FairFace 0.25 config — 86,744 training / 10,954 test samples.
    """)

# ==========================================
# LIMITATIONS
# ==========================================
st.subheader("Limitations & Responsible Use")
st.markdown("""
- This system was developed for **educational purposes only**.
- Trained solely on FairFace — may not generalise to different lighting, poses, or camera conditions.
- Despite bias mitigation, **performance gaps across demographic groups remain**.
- **Do not** use for real-world security, surveillance, hiring, or identity verification.
- Age estimation from face images raises **privacy and ethical concerns** — always obtain informed consent.
- Accuracy is lowest for **extreme age groups (0–2, 70+)** due to class imbalance in training data.
""")

st.caption("FairVision | IJSE Certified AI & ML Engineer | 2025/2026")
