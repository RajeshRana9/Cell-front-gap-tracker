# Cell-Front Gap Measurement App (Streamlit)

This Streamlit application measures the gap between two cell fronts from time-series microscopy images (e.g., Day 0 → Day 2).  
It detects edges, computes column-wise distances, and plots how the gap changes over time.

---

## 🚀 Features
- Upload or load images from a server folder
- Edge detection (Gaussian blur + Otsu + Canny)
- Column-wise gap estimation between top and bottom fronts
- Pixel → micron conversion
- Trend plot over time
- Export CSV or JSON results

---

## 📦 Installation

```bash
git clone <your_repo_url>
cd <repo-folder>
pip install -r requirements.txt
