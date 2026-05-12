# FairVision — Streamlit Deployment Guide

## Files in this package

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main Streamlit application |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | App theme & settings |
| `baseline_model.pth` | *(Add after training)* Baseline CNN weights |
| `model_weighted.pth` | *(Add after training)* Weighted-loss model weights |
| `model_balanced.pth` | *(Add after training)* Balanced-sampling model weights |

---

## Step 1 — Train the models

Run the Jupyter Notebook (`FairVision_Assignment_Fixed.ipynb`) end-to-end.
After training, three `.pth` files will appear in the notebook directory:
- `baseline_model.pth`
- `model_weighted.pth`
- `model_balanced.pth`

Copy all three into this deployment folder alongside `streamlit_app.py`.

---

## Step 2 — Test locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## Step 3 — Deploy to Streamlit Cloud (Recommended — Free)

### 3a. Push to GitHub

1. Create a **public** GitHub repository (e.g. `fairvision-demo`).
2. Add all files from this folder, including the `.pth` files.
3. Push to the `main` branch.

> **Note on file size:** PyTorch `.pth` files for this model are typically
> 3–15 MB each, which is well within GitHub's 100 MB per-file limit.
> If they are larger, use the HuggingFace Hub option below.

### 3b. Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Select your repository, branch (`main`), and main file (`streamlit_app.py`).
4. Click **Deploy**.
5. Wait 2–3 minutes for the build to complete.
6. Copy the public URL (e.g. `https://yourname-fairvision-demo.streamlit.app`).

---

## Alternative — Host models on HuggingFace Hub (if .pth files are large)

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Create a new **public model repository**.
3. Upload the three `.pth` files.
4. Get the raw download URL for each file:
   ```
   https://huggingface.co/<username>/<repo>/resolve/main/baseline_model.pth
   ```
5. In Streamlit Cloud → **Settings → Secrets**, add:
   ```toml
   URL_BASELINE = "https://huggingface.co/..."
   URL_WEIGHTED  = "https://huggingface.co/..."
   URL_BALANCED  = "https://huggingface.co/..."
   ```
   The app will automatically download models on first run.

---

## Submission checklist

- [ ] App is publicly accessible at the URL
- [ ] All three models load without errors
- [ ] Image upload and top-3 prediction works
- [ ] URL included in the technical report (Section 9)
- [ ] App remains live until grading is complete
