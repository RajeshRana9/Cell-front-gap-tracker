"""
Streamlit app: Cell front gap tracker

How it works (brief):
- Upload a set of time-series microscopy images (day0, day1, day2...).
- For each image we detect the two opposing "fronts" by simple edge detection + column-wise scanning:
    * Convert to grayscale -> blur -> threshold -> edge detection (Canny)
    * For each image column we find the first edge from top (y_top) and first edge from bottom (y_bottom)
    * Column gap = y_bottom - y_top (when both exist); image gap summary = mean/median/std of column gaps
- The app reports mean/median/std gaps (in pixels and in calibrated units when you provide microns-per-pixel)
- Exports a CSV with per-image statistics and the per-column gap arrays if requested.

Notes / limitations (near-accuracy):
- This is a pragmatic, fast method. If your images are noisy, or fronts are not continuous, consider preprocessing (contrast, background subtraction) or use more advanced segmentation (deep learning).
- If fronts are roughly horizontal and opposing (one from top, one from bottom), this approach works well. For arbitrary orientations you should rotate or use distance transform on segmented masks.

Run: `pip install streamlit opencv-python-headless numpy pandas matplotlib scikit-image` 
       `streamlit run streamlit_cell_gap_app.py`

"""

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Cell Front Gap Tracker", layout="wide")

st.title("🧫 Cell-front gap tracker — Streamlit")
st.markdown(
"""
Upload a series of images (time series). The app estimates the column-wise gap between two opposing fronts
and summarizes how the gap changes across time. Provide a calibration (microns per pixel) to convert results into physical units.
"""
)

# Sidebar controls
st.sidebar.header("Settings")
px_to_micron = st.sidebar.number_input("Microns per pixel (calibration)", min_value=0.0, value=1.0, step=0.01, format="%.4f")
use_server_folder = st.sidebar.checkbox("Load from server folder (path) instead of uploading files", value=False)
server_path = None
if use_server_folder:
    server_path = st.sidebar.text_input("Server folder path (absolute)", value="/mnt/data/images")

smoothing = st.sidebar.slider("Gaussian blur sigma", 0.0, 5.0, 1.0, 0.5)
canny_low = st.sidebar.slider("Canny low threshold", 1, 200, 50)
canny_high = st.sidebar.slider("Canny high threshold", 1, 300, 150)
min_valid_columns = st.sidebar.slider("Minimum valid columns per image (to accept stats)", 1, 50, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("If your images have timestamps in filenames, the app will sort them alphabetically (recommended: include day number).")

# File uploader
st.header("Upload images")
if not use_server_folder:
    uploaded = st.file_uploader("Upload multiple images (PNG, JPG, TIFF). Preferred: one file per timepoint", type=["png","jpg","jpeg","tif","tiff"], accept_multiple_files=True)
else:
    uploaded = []

# Helper functions

def read_image_file(fobj):
    # Accepts file-like or path string
    if isinstance(fobj, str):
        img = cv2.imread(fobj, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Unable to read image from path: {fobj}")
        # convert BGR to RGB
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        img = Image.open(fobj).convert("RGB")
        arr = np.array(img)
        return arr


def preprocess_for_edges(img_rgb, sigma=1.0):
    # img_rgb: HxWx3
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if sigma > 0:
        k = max(3, int(round(sigma * 4 + 1)))
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), sigma)
    # Adaptive threshold or Otsu
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological closing to connect thin gaps
    kernel = np.ones((3,3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.Canny(th, canny_low, canny_high)
    return edges, gray


def compute_column_gaps(edges):
    h, w = edges.shape
    gaps = np.full(w, np.nan)
    for x in range(w):
        col = edges[:, x]
        ys = np.where(col > 0)[0]
        if ys.size == 0:
            continue
        # first edge from top
        y_top = ys.min()
        # first edge from bottom
        y_bottom = ys.max()
        if y_bottom >= y_top:
            gaps[x] = y_bottom - y_top
    return gaps


def overlay_result(img_rgb, edges, gaps):
    out = img_rgb.copy()
    h, w = edges.shape
    # draw edges (red) and lines showing top/bottom median
    # Mark columns where gaps exist
    for x in range(w):
        if np.isfinite(gaps[x]):
            ytop = np.where(edges[:, x] > 0)[0].min()
            ybot = np.where(edges[:, x] > 0)[0].max()
            # draw small vertical line
            cv2.line(out, (x, int(ytop)), (x, int(ybot)), (255, 0, 0), 1)
    # convert to PIL-compatible RGB
    return out


# load images list
image_items = []  # list of tuples (name, ndarray)
if use_server_folder and server_path:
    if os.path.isdir(server_path):
        files = [os.path.join(server_path, f) for f in os.listdir(server_path) if f.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff"))]
        files.sort()
        for p in files:
            try:
                image_items.append((os.path.basename(p), read_image_file(p)))
            except Exception as e:
                st.warning(f"Failed to read {p}: {e}")
    else:
        st.error("Server path not found or is not a directory")

if uploaded:
    # sort by filename to preserve time order if user named files appropriately
    uploaded_sorted = sorted(uploaded, key=lambda f: f.name)
    for f in uploaded_sorted:
        try:
            image_items.append((f.name, read_image_file(f)))
        except Exception as e:
            st.warning(f"Failed reading {f.name}: {e}")

if not image_items:
    st.info("No images loaded yet. Upload multiple images or enable server folder.")
    st.stop()

st.success(f"Loaded {len(image_items)} images.")

# Process images
results = []
col1, col2 = st.columns([1,1])
with col1:
    st.header("Per-image results")
with col2:
    st.header("Summary across times")

for name, img in image_items:
    edges, gray = preprocess_for_edges(img, sigma=smoothing)
    gaps = compute_column_gaps(edges)
    valid_count = np.sum(np.isfinite(gaps))
    if valid_count < min_valid_columns:
        mean_gap = np.nan
        median_gap = np.nan
        std_gap = np.nan
    else:
        mean_gap = float(np.nanmean(gaps))
        median_gap = float(np.nanmedian(gaps))
        std_gap = float(np.nanstd(gaps))

    # store per-image result
    results.append({
        "filename": name,
        "mean_gap_px": mean_gap,
        "median_gap_px": median_gap,
        "std_gap_px": std_gap,
        "valid_columns": int(valid_count),
        "gaps_array": gaps.tolist(),
    })

    # show small preview
    with col1:
        st.subheader(name)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.imshow(img)
        ax.axis('off')
        # overlay edges as alpha
        ax.imshow(edges, cmap='Reds', alpha=0.35)
        ax.set_title(f"Mean gap: {mean_gap:.2f} px  (valid cols: {valid_count})" if not np.isnan(mean_gap) else "Insufficient data")
        st.pyplot(fig)

# Build summary dataframe
df = pd.DataFrame([{k:v for k,v in r.items() if k!='gaps_array'} for r in results])
# Add calibrated units if provided
if px_to_micron and px_to_micron > 0:
    df['mean_gap_micron'] = df['mean_gap_px'] * px_to_micron
    df['median_gap_micron'] = df['median_gap_px'] * px_to_micron
    df['std_gap_micron'] = df['std_gap_px'] * px_to_micron

with col2:
    st.dataframe(df.style.format({
        'mean_gap_px': '{:.2f}',
        'median_gap_px': '{:.2f}',
        'std_gap_px': '{:.2f}',
    }))

    # Plot trend
    fig2, ax2 = plt.subplots(figsize=(6,3))
    x = list(range(len(df)))
    ax2.plot(x, df['mean_gap_px'], marker='o', label='Mean gap (px)')
    if 'mean_gap_micron' in df:
        ax2.plot(x, df['mean_gap_micron'], marker='x', linestyle='--', label=f'Mean gap (microns, ×{px_to_micron})')
    ax2.set_xlabel('Timepoint (sorted filenames)')
    ax2.set_ylabel('Gap')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['filename'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

# Allow download of CSV and full JSON
st.markdown('---')
st.header('Export results')
if st.button('Download CSV (summary)'):
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button('Download CSV', data=csv_bytes, file_name=f'gap_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

# Option to download full data with per-column gaps
if st.checkbox('Include per-column gaps arrays in export (JSON)'):
    import json
    full = results
    json_bytes = json.dumps(full, default=str).encode('utf-8')
    st.download_button('Download full JSON', data=json_bytes, file_name=f'gap_full_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

st.markdown('---')
st.write('Tips:')
st.write('- If the fronts are not roughly horizontal, rotate the images first or run a segmentation method that identifies the two fronts as masks.')
st.write('- If edges are noisy, increase Gaussian sigma or do background subtraction before uploading.')

st.caption('This app gives a fast, pragmatic (near-accuracy) estimate of gaps between two opposing fronts by column-wise scanning of edges. For publication-grade accuracy, follow-up segmentation and manual QC are recommended.')
