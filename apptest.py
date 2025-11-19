"""
Streamlit — Wound Healing (closure) measurement app

Saves:
 - summary CSV with mean/median/std gap (px and microns)
 - wound area (px^2 and microns^2)
 - percent closure relative to first timepoint

How it measures (robust for wound-healing setups):
 - Convert to grayscale -> blur -> threshold (Otsu or adaptive)
 - Morphological clean-up -> binary mask of cells (True where cells are)
 - For each column: find topmost cell row (from top), bottommost cell row (from bottom)
 - Column gap = bottom_row - top_row - 1 (only if both exist and bottom>top)
 - Wound area computed between these boundaries where mask==False
"""

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.morphology import remove_small_objects, closing, square
from skimage.measure import label

st.set_page_config(page_title="Wound Healing Closure App", layout="wide")
st.title("🩺 Wound healing closure measurement (robust segmentation)")

st.markdown("""
Upload your time-series wound images (e.g., Day0, Day1, Day2...).  
This app segments **cell areas**, computes column-wise gap (true wound width), wound area, and percent closure compared to Day 0.
""")

# Sidebar - parameters
st.sidebar.header("Preprocessing & segmentation")
px_to_micron = st.sidebar.number_input("Microns per pixel (calibration)", min_value=0.0, value=1.0, step=0.01, format="%.4f")
use_server_folder = st.sidebar.checkbox("Load from server folder (path)", value=False)
server_path = None
if use_server_folder:
    server_path = st.sidebar.text_input("Server folder path (absolute)", value="/mnt/data")

# segmentation choices
seg_method = st.sidebar.selectbox("Segmentation method", ["Otsu (global)", "Adaptive (local)"])
smoothing = st.sidebar.slider("Gaussian blur sigma", 0.0, 5.0, 1.2, 0.1)
adaptive_block = st.sidebar.slider("Adaptive block size (odd)", 3, 201, 101, 2)
morph_kernel = st.sidebar.slider("Morph closing kernel size (px)", 1, 51, 5, 2)
min_object_size = st.sidebar.number_input("Remove small objects smaller than (px)", min_value=0, value=200, step=50)

st.sidebar.markdown("---")
st.sidebar.subheader("Measurement & QC")
min_valid_cols = st.sidebar.number_input("Minimum valid columns to accept measurement", min_value=1, value=10)
show_overlay = st.sidebar.checkbox("Show overlay previews", value=True)
overlay_mode = st.sidebar.selectbox("Overlay mode", ["mask+medians", "gap-filled", "top-bottom-lines"])
preview_fig_h = st.sidebar.slider("Preview height (inch)", 3.0, 10.0, 4.0, 0.5)
st.sidebar.markdown("---")
st.sidebar.markdown("Quick presets for A1 images: smoothing=1.2, Otsu, block=101, kernel=5, min_obj=200")
if st.sidebar.button("Apply A1 preset"):
    smoothing = 1.2
    seg_method = "Otsu (global)"
    adaptive_block = 101
    morph_kernel = 5
    min_object_size = 200
    st.sidebar.success("Preset applied: adjust sliders if needed (widgets do not auto-update).")

# ROI cropping (same for all images) - useful to keep consistent area
st.sidebar.subheader("ROI (apply same box to all images)")
use_roi = st.sidebar.checkbox("Crop to ROI", value=False)
roi_top = st.sidebar.number_input("ROI top (px)", min_value=0, value=0, step=1)
roi_left = st.sidebar.number_input("ROI left (px)", min_value=0, value=0, step=1)
roi_height = st.sidebar.number_input("ROI height (px)", min_value=0, value=0, step=1)
roi_width = st.sidebar.number_input("ROI width (px)", min_value=0, value=0, step=1)
st.sidebar.caption("If ROI height/width = 0, full image is used. Make sure all images share same size.")

# File upload
st.header("Upload images (or use server folder)")
if not use_server_folder:
    uploaded = st.file_uploader("Upload multiple images (png/jpg/tif). Name them so sorting = time order.", accept_multiple_files=True, type=["png","jpg","jpeg","tif","tiff"])
else:
    uploaded = []

image_items = []
if use_server_folder and server_path:
    if os.path.isdir(server_path):
        files = [os.path.join(server_path, f) for f in os.listdir(server_path) if f.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff"))]
        files.sort()
        for p in files:
            try:
                img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if len(img.shape)==3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_items.append((os.path.basename(p), img))
            except Exception as e:
                st.warning(f"Failed reading {p}: {e}")
    else:
        st.error("Server path invalid.")
if uploaded:
    uploaded_sorted = sorted(uploaded, key=lambda f: f.name)
    for f in uploaded_sorted:
        try:
            pil = Image.open(f).convert("RGB")
            arr = np.array(pil)
            image_items.append((f.name, arr))
        except Exception as e:
            st.warning(f"Failed reading {f.name}: {e}")

if not image_items:
    st.info("No images loaded. Upload images or enable server folder.")
    st.stop()

st.success(f"Loaded {len(image_items)} images.")

# helper functions
def crop_image(img, top, left, h, w):
    if h<=0 or w<=0:
        return img
    H, W = img.shape[:2]
    top = int(max(0, min(top, H-1)))
    left = int(max(0, min(left, W-1)))
    bottom = int(max(top+1, min(top+h, H)))
    right = int(max(left+1, min(left+w, W)))
    return img[top:bottom, left:right].copy()

def segment_cells(img_rgb, method="Otsu (global)", sigma=1.2, adaptive_block=101, morph_k=5, min_size=200):
    """Return binary mask True for cell pixels."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if sigma>0:
        k = max(3, int(round(sigma*4+1)))
        if k%2==0: k+=1
        gray = cv2.GaussianBlur(gray, (k,k), sigma)
    if method.startswith("Otsu"):
        # global threshold by Otsu
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # decide whether foreground is cells or background — we expect cells to be darker/brighter? Check which yields more objects
        mask = th>0   # True where bright
        # If cells appear bright? but in your A1 they are darker at edges; try invert if necessary
        # We'll test both and pick the mask with larger connected components area (heuristic)
        mask_inv = ~mask
        lbl1 = label(mask)
        lbl2 = label(mask_inv)
        area1 = np.sum(mask)
        area2 = np.sum(mask_inv)
        # Prefer smaller foreground area as cells usually cover less area than empty wound? We'll decide heuristically:
        # If mask area is > 0.8 of image, invert.
        H,W = gray.shape
        if area1 > 0.9*H*W:
            mask = ~mask
    else:
        # adaptive/local threshold (gaussian)
        # adaptiveMean or adaptiveGaussian
        # OpenCV adaptiveThreshold expects uint8
        block = int(adaptive_block)
        if block%2==0:
            block += 1
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, blockSize=block, C=2)
        mask = th>0
        # ensure we have sensible foreground choice
        H,W = gray.shape
        if np.sum(mask) > 0.9*H*W:
            mask = ~mask

    # morphological closing to fill small holes and connect edges
    kernel = np.ones((morph_k, morph_k), np.uint8)
    mask_uint8 = (mask.astype(np.uint8)*255)
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
    bw = mask_closed>0

    # remove small objects - using skimage
    if min_size>0:
        try:
            bw = remove_small_objects(bw, min_size=min_size)
        except Exception:
            # fallback: simple connected component filter with cv2
            lbl = label(bw)
            props = np.bincount(lbl.ravel())
            keep = np.zeros_like(bw)
            for i,count in enumerate(props):
                if i==0: continue
                if count>=min_size:
                    keep[lbl==i] = True
            bw = keep
    # final fill small holes by closing again
    if morph_k>0:
        bw = closing(bw, square(morph_k))

    return bw.astype(bool), gray

def compute_gaps_from_mask(mask):
    """Given boolean mask True=cell, compute column-wise top & bottom cell rows and gap/area.
       Returns:
         gaps_px: array(float, w) (nan where invalid)
         tops, bots arrays int (nan where invalid)
         wound_area_px: number of background pixels that lie between top and bottom for each column summed
    """
    h,w = mask.shape
    tops = np.full(w, np.nan)
    bots = np.full(w, np.nan)
    gaps = np.full(w, np.nan)
    wound_area = 0
    for x in range(w):
        col = mask[:, x]
        cell_rows = np.where(col)[0]
        if cell_rows.size == 0:
            # no cells in this column
            continue
        # top-most cell pixel (closest to top)
        y_top = cell_rows.min()
        y_bot = cell_rows.max()
        tops[x] = y_top
        bots[x] = y_bot
        if y_bot > y_top:
            # gap = empty rows between them
            gaps[x] = y_bot - y_top - 1
            # count background pixels between them where mask == False
            if y_bot - y_top - 1 > 0:
                wound_area += np.sum(~mask[y_top+1:y_bot, x])
        else:
            gaps[x] = 0.0
    return gaps, tops, bots, wound_area

# Process each image
results = []
first_mean = None

col1, col2 = st.columns([1,1])
with col1:
    st.header("Per-image previews & QC")
with col2:
    st.header("Numeric summary & trend")

for idx, (name, img) in enumerate(image_items):
    # apply ROI crop if requested
    proc_img = img.copy()
    if use_roi and roi_height>0 and roi_width>0:
        proc_img = crop_image(proc_img, int(roi_top), int(roi_left), int(roi_height), int(roi_width))

    mask, gray = segment_cells(proc_img, method=seg_method, sigma=float(smoothing),
                                adaptive_block=int(adaptive_block), morph_k=int(morph_kernel),
                                min_size=int(min_object_size))
    gaps_px, tops, bots, wound_area_px = compute_gaps_from_mask(mask)
    valid_cols = int(np.sum(np.isfinite(gaps_px)))
    if valid_cols < int(min_valid_cols):
        mean_gap = np.nan
        median_gap = np.nan
        std_gap = np.nan
    else:
        mean_gap = float(np.nanmean(gaps_px))
        median_gap = float(np.nanmedian(gaps_px))
        std_gap = float(np.nanstd(gaps_px))

    if first_mean is None:
        first_mean = mean_gap  # baseline

    # percent closure relative to first image (when first_mean valid)
    if (first_mean is None) or np.isnan(first_mean) or np.isnan(mean_gap):
        pct_closed = np.nan
    else:
        # closure = (initial - current)/initial
        pct_closed = float((first_mean - mean_gap) / first_mean * 100.0)

    # convert to microns
    mean_gap_micron = mean_gap * px_to_micron if (not np.isnan(mean_gap)) else np.nan
    median_gap_micron = median_gap * px_to_micron if (not np.isnan(median_gap)) else np.nan
    std_gap_micron = std_gap * px_to_micron if (not np.isnan(std_gap)) else np.nan
    wound_area_micron2 = wound_area_px * (px_to_micron**2)

    results.append({
        "filename": name,
        "mean_gap_px": mean_gap,
        "median_gap_px": median_gap,
        "std_gap_px": std_gap,
        "valid_columns": valid_cols,
        "wound_area_px2": int(wound_area_px),
        "mean_gap_micron": mean_gap_micron,
        "median_gap_micron": median_gap_micron,
        "std_gap_micron": std_gap_micron,
        "wound_area_micron2": wound_area_micron2,
        "percent_closed": pct_closed,
    })

    # visualization / overlay
    with col1:
        st.subheader(f"{idx+1}. {name}")
        fig, ax = plt.subplots(figsize=(8, preview_fig_h))
        # base image
        ax.imshow(proc_img)
        ax.axis('off')

        # overlay mask (semi transparent)
        if show_overlay:
            # colored mask overlay
            mask_rgb = np.zeros_like(proc_img, dtype=np.uint8)
            mask_rgb[...,1] = (mask.astype(np.uint8) * 180)  # green-ish overlay
            ax.imshow(mask_rgb, alpha=0.35)

            # median top and bottom lines if valid
            if np.sum(np.isfinite(tops))>0:
                med_top = int(np.nanmedian(tops))
                ax.axhline(med_top, color='lime', linestyle='--', linewidth=1.2, label='median top')
            if np.sum(np.isfinite(bots))>0:
                med_bot = int(np.nanmedian(bots))
                ax.axhline(med_bot, color='cyan', linestyle='--', linewidth=1.2, label='median bottom')

            if overlay_mode == "top-bottom-lines":
                # draw per-column top/bottom (sparse: every 5 px to reduce clutter)
                h,w = mask.shape
                for x in range(0, w, max(1,w//200)):
                    if not np.isnan(tops[x]):
                        ax.plot([x],[tops[x]], marker='o', markersize=2, color='g')
                    if not np.isnan(bots[x]):
                        ax.plot([x],[bots[x]], marker='o', markersize=2, color='b')
            elif overlay_mode == "gap-filled":
                # fill area between top and bottom where gap exists
                h,w = mask.shape
                for x in range(w):
                    if np.isfinite(tops[x]) and np.isfinite(bots[x]) and bots[x] > tops[x]+1:
                        y1 = int(tops[x]+1)
                        y2 = int(bots[x]-1)
                        ax.fill_between([x-0.4, x+0.4], [y1,y1], [y2,y2], color='magenta', alpha=0.25)

            # label info
            txt = f"Mean gap: {mean_gap:.1f} px ({mean_gap_micron:.2f} μm) | Wound area: {wound_area_px} px² ({wound_area_micron2:.1f} μm²) | % closed: {pct_closed if not np.isnan(pct_closed) else 'N/A'}"
            ax.set_title(txt)
        else:
            ax.set_title("Overlay disabled")
        st.pyplot(fig)

# Summary table and trend plots
df = pd.DataFrame(results)
# ensure ordering
df = df.reset_index(drop=True)

with col2:
    st.dataframe(df.style.format({
        'mean_gap_px': '{:.2f}',
        'median_gap_px': '{:.2f}',
        'std_gap_px': '{:.2f}',
        'mean_gap_micron': '{:.2f}',
        'wound_area_px2': '{:d}',
        'wound_area_micron2': '{:.1f}',
        'percent_closed': '{:.2f}'
    }))

    # mean gap trend plot
    fig2, ax2 = plt.subplots(figsize=(6,3))
    x = list(range(len(df)))
    ax2.plot(x, df['mean_gap_px'], marker='o', label='Mean gap (px)')
    if px_to_micron>0:
        ax2.plot(x, df['mean_gap_micron'], marker='x', linestyle='--', label=f'Mean gap (μm)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['filename'], rotation=45, ha='right')
    ax2.set_ylabel("Gap")
    ax2.set_xlabel("Timepoint (sorted filenames)")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    # percent closure plot
    fig3, ax3 = plt.subplots(figsize=(6,3))
    ax3.plot(x, df['percent_closed'], marker='s', color='tab:orange')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df['filename'], rotation=45, ha='right')
    ax3.set_ylabel("% closed (relative to first timepoint)")
    ax3.set_ylim(-10, 110)
    ax3.grid(True)
    st.pyplot(fig3)

# Export
st.markdown("---")
st.header("Export results")
if st.button("Download CSV summary"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name=f"wound_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

if st.checkbox("Include per-image masks and meta in JSON export (large)"):
    import json, base64, io
    payload = []
    for (name,img), row in zip(image_items, results):
        # encode mask as compressed PNG to keep export smallish
        proc_img = img.copy()
        if use_roi and roi_height>0 and roi_width>0:
            proc_img = crop_image(proc_img, int(roi_top), int(roi_left), int(roi_height), int(roi_width))
        mask, _ = segment_cells(proc_img, method=seg_method, sigma=float(smoothing),
                                adaptive_block=int(adaptive_block), morph_k=int(morph_kernel),
                                min_size=int(min_object_size))
        # encode mask to PNG
        mask_u8 = (mask.astype(np.uint8)*255)
        ok, buf = cv2.imencode('.png', mask_u8)
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        payload.append({
            "filename": name,
            "mask_png_base64": b64,
            "meta": row
        })
    json_bytes = json.dumps(payload).encode('utf-8')
    st.download_button("Download full JSON", data=json_bytes, file_name=f"wound_detail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

st.caption("Note: this segmentation-based approach measures the empty wound region between cell masks. If images vary in exposure or magnification between timepoints, align/normalize them first. For publication figures, visually QC overlays for multiple fields, and compute mean ± SEM across biological replicates.")
