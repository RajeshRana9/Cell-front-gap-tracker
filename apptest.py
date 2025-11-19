"""
Streamlit app: Cell front gap tracker (with overlay preview)

Features added in this version:
- Toggleable overlay that draws the detected top/bottom front lines on each image.
- Choice of overlay mode: 'edges' (Canny heat), 'lines' (column-wise top/bottom), or 'filled-mask' (simple interpolation between fronts)
- Per-image overlay preview with ability to zoom by changing figure size.
- Quick recommended presets for your example images.

Run: `pip install streamlit opencv-python-headless numpy pandas matplotlib scikit-image pillow`
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

st.title("🧫 Cell-front gap tracker — Streamlit (Overlay enabled)")
st.markdown(
"""
Upload a series of images (time series). The app estimates the column-wise gap between two opposing fronts
and shows an overlay so you can visually QC what is being measured.
"""
)

# Sidebar controls
st.sidebar.header("Settings")
px_to_micron = st.sidebar.number_input("Microns per pixel (calibration)", min_value=0.0, value=1.0, step=0.01, format="%.4f")
use_server_folder = st.sidebar.checkbox("Load from server folder (path) instead of uploading files", value=False)
server_path = None
if use_server_folder:
    server_path = st.sidebar.text_input("Server folder path (absolute)", value="/mnt/data/images")

smoothing = st.sidebar.slider("Gaussian blur sigma", 0.0, 5.0, 1.0, 0.1)
canny_low = st.sidebar.slider("Canny low threshold", 1, 300, 25)
canny_high = st.sidebar.slider("Canny high threshold", 1, 400, 120)
min_valid_columns = st.sidebar.slider("Minimum valid columns per image (to accept stats)", 1, 200, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("Overlay options")
show_overlay = st.sidebar.checkbox("Show overlay on previews", value=True)
overlay_mode = st.sidebar.selectbox("Overlay mode", ["edges","lines","filled-mask"]) 
preview_figsize = st.sidebar.slider("Preview figure height (inch)", 2.0, 10.0, 3.0, 0.5)

# Quick presets for your A1 images
if st.sidebar.button('Apply recommended preset for your A1 images'):
    smoothing = 1.2
    # update widget values by rerunning using st.experimental_rerun would be required; instead show a message
    st.sidebar.success('Recommended values (smoothing=1.2, low=25, high=120, min_cols=10) — adjust sliders manually if needed.')

st.sidebar.markdown("---")
st.sidebar.caption("Tip: use the overlay to confirm edges follow the true cell fronts. Use 'lines' mode to see column-wise measured lines.")

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
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if sigma > 0:
        k = max(3, int(round(sigma * 4 + 1)))
        if k % 2 == 0:
            k += 1
        gray = cv2.GaussianBlur(gray, (k, k), sigma)
    # Adaptive threshold using Otsu to help remove background gradient
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.Canny(th, canny_low, canny_high)
    return edges, gray, th


def compute_column_gaps(edges):
    h, w = edges.shape
    tops = np.full(w, np.nan)
    bots = np.full(w, np.nan)
    gaps = np.full(w, np.nan)
    for x in range(w):
        col = edges[:, x]
        ys = np.where(col > 0)[0]
        if ys.size == 0:
            continue
        tops[x] = ys.min()
        bots[x] = ys.max()
        if bots[x] >= tops[x]:
            gaps[x] = bots[x] - tops[x]
    return gaps, tops, bots


def draw_overlay(img_rgb, edges, tops, bots, mode='lines'):
    out = img_rgb.copy()
    h, w = edges.shape
    # Convert to BGR for cv2 drawing convenience
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    if mode == 'edges':
        # draw edges as semi-transparent red overlay
        red = np.zeros_like(out_bgr)
        red[:, :, 2] = edges
        overlay = cv2.addWeighted(out_bgr, 1.0, red, 0.45, 0)
        out_bgr = overlay
    elif mode == 'lines':
        # draw column-wise top (green) and bottom (blue)
        for x in range(w):
            if not np.isnan(tops[x]):
                ytop = int(tops[x])
                ybot = int(bots[x])
                cv2.circle(out_bgr, (x, ytop), 1, (0,255,0), -1)  # green dot for top
                cv2.circle(out_bgr, (x, ybot), 1, (255,0,0), -1)  # blue dot for bottom
        # optionally draw median lines
        valid_tops = np.where(np.isfinite(tops))[0]
        valid_bots = np.where(np.isfinite(bots))[0]
        if valid_tops.size>0:
            ytop_med = int(np.nanmedian(tops))
            cv2.line(out_bgr, (0, ytop_med), (w-1, ytop_med), (0,255,0), 1)
        if valid_bots.size>0:
            ybot_med = int(np.nanmedian(bots))
            cv2.line(out_bgr, (0, ybot_med), (w-1, ybot_med), (255,0,0), 1)
    elif mode == 'filled-mask':
        # build a mask interpolating between tops and bots
        mask = np.zeros((h,w), dtype=np.uint8)
        for x in range(w):
            if np.isfinite(tops[x]) and np.isfinite(bots[x]):
                y1 = int(tops[x]); y2 = int(bots[x])
                mask[y1:y2+1, x] = 255
        colored = np.zeros_like(out_bgr)
        colored[:, :, 1] = mask  # green channel mask
        out_bgr = cv2.addWeighted(out_bgr, 1.0, colored, 0.35, 0)

    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    return out_rgb

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
    st.header("Per-image results & overlays")
with col2:
    st.header("Summary across times")

for idx, (name, img) in enumerate(image_items):
    edges, gray, th = preprocess_for_edges(img, sigma=smoothing)
    gaps, tops, bots = compute_column_gaps(edges)
    valid_count = int(np.sum(np.isfinite(gaps)))
    if valid_count < min_valid_columns:
        mean_gap = np.nan
        median_gap = np.nan
        std_gap = np.nan
    else:
        mean_gap = float(np.nanmean(gaps))
        median_gap = float(np.nanmedian(gaps))
        std_gap = float(np.nanstd(gaps))

    results.append({
        "filename": name,
        "mean_gap_px": mean_gap,
        "median_gap_px": median_gap,
        "std_gap_px": std_gap,
        "valid_columns": int(valid_count),
        "gaps_array": gaps.tolist(),
    })

    # show small preview with overlay if requested
    with col1:
        st.subheader(f"{idx+1}. {name}")
        fig, ax = plt.subplots(figsize=(8, preview_figsize))
        if show_overlay:
            overlay_img = draw_overlay(img, edges, tops, bots, mode=overlay_mode)
            ax.imshow(overlay_img)
        else:
            ax.imshow(img)
            ax.imshow(edges, cmap='Reds', alpha=0.25)
        ax.axis('off')
        title_text = f"Mean gap: {mean_gap:.2f} px  (valid cols: {valid_count})" if not np.isnan(mean_gap) else "Insufficient data"
        ax.set_title(title_text)
        st.pyplot(fig)

# Build summary dataframe
df = pd.DataFrame([{k:v for k,v in r.items() if k!='gaps_array'} for r in results])
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
