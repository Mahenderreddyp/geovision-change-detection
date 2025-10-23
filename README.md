# 🌍 GeoVision: Satellite Change Detection using Siamese U-Net

A complete deep learning pipeline for **temporal change detection** from **Sentinel-2 satellite imagery** — built to automatically detect **urban growth, vegetation loss, and construction activity** across multiple regions.

This project demonstrates:
- 🛰 **Sentinel-2 data acquisition** from STAC API (low-cloud, Level-2A)
- 🧭 **Preprocessing** (alignment, NDVI differencing, SCL masking)
- 🧩 **Weak-label dataset generation**
- 🧠 **Training a Siamese U-Net model** for pixel-level change detection
- ⚙️ **GPU-accelerated inference** on full GeoTIFFs
- 🌐 **Multi-AOI (Area of Interest)** inference and visualization
- ☁ **Integration-ready** for Vertex AI, Google Cloud Storage, or Esri ArcGIS workflows.

---

## 🛰️ Dataset: Sentinel-2 (ESA Copernicus)

**Source:** [Sentinel-2 Level-2A Collection via Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a)

- Spatial resolution: **10 m/pixel**
- Bands used: `B02 (Blue)`, `B03 (Green)`, `B04 (Red)`
- Time ranges:
  - **Chicago:** June 2022 → June 2023  
  - **San Jose:** July 2022 → July 2023  
  - **Houston:** August 2022 → August 2023
- Cloud coverage filter: `< 20 %`
- Weak labels generated via **NDVI change + Otsu thresholding**, masked by the **SCL (Scene Classification Layer)**.

---

## 🧩 Pipeline Overview

| Stage | Description |
|:------|:-------------|
| **1️⃣ Data Search & Download** | Query Sentinel-2 L2A STAC catalog for given AOI & dates |
| **2️⃣ Raster Alignment** | Reproject both timestamps to a common CRS (auto UTM) |
| **3️⃣ Weak Label Generation** | NDVI difference + Otsu threshold → binary mask |
| **4️⃣ Patch Extraction** | 256×256 patches with 50 % stride |
| **5️⃣ Siamese U-Net Training** | 3+3 RGB input → 1-channel change mask output |
| **6️⃣ GPU Inference (Full Scene)** | Memory-safe tiling, GPU-optimized |
| **7️⃣ Multi-AOI Visualization** | Automated previews and Folium interactive overlays |

---

## 🧠 Model Architecture: Siamese U-Net

**Base:** U-Net encoder–decoder  
**Input:** 6 channels (3 bands × 2 timestamps)  
**Output:** 1 binary change mask  

t0 (RGB) ─┐                   ┌────────────┐
           ├─ Siamese U-Net ─►    ΔFeature │
t1 (RGB) ─┘                   └────────────┘
                 │
                 ▼
           Sigmoid → Binary Change Map

**Loss:** BCEWithLogitsLoss  
**Optimizer:** Adam (LR = 1e-3)  
**Metrics:** IoU (Intersection over Union)

---

## ⚙️ Training Summary

| Metric | Value |
|:--|:--|
| Training patches | 38 (256×256) |
| Epochs | 200 |
| Best IoU | **0.556** |
| Framework | PyTorch 2.1 + CUDA 12 |
| Hardware | Vertex AI (L4 GPU) |
| Checkpoint | `runs/best_model.pt` |

**Loss curve:** BCE steadily decreased from 0.57 → 0.05  
**IoU:** improved from 0.10 → **0.55+**, showing consistent generalization.

---

## 🧪 Sample Predictions

### 🏙 Chicago (Urban Change)
![Chicago](examples/outputs/Chicago/preview.png)

### 🌴 San Jose (Suburban)
![SanJose](examples/outputs/SanJose/preview.png)

### 🏗 Houston (Construction Growth)
![Houston](examples/outputs/Houston/preview.png)

Each red overlay indicates detected change regions between the two timestamps.

---

## 🧰 Project Structure

geovision-change-detection/
│
├─ src/geovision/
│   ├─ model.py       # Siamese U-Net
│   ├─ data.py        # Patch dataset
│   ├─ train.py       # Training loop + IoU metric
│   ├─ infer.py       # GPU-safe full-scene inference
│   └─ utils.py       # Normalization utilities
│
├─ scripts/
│   ├─ train.py           # Train Siamese U-Net
│   ├─ infer_full.py      # Inference on single AOI
│   └─ infer_multi_aoi.py # Multi-AOI batch inference
│
├─ examples/
│   └─ outputs/           # Previews for README
│
├─ runs/                  # Saved model checkpoints
├─ processed/             # Patch-level .npy datasets
├─ interim/               # Aligned rasters
├─ raw/                   # Downloaded Sentinel pairs
├─ requirements.txt
├─ Makefile
└─ README.md


⸻

---

## 🚀 Quickstart

### 1️⃣ Clone and setup
```bash
git clone https://github.com/mahenderreddyp/geovision-change-detection.git
cd geovision-change-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2️⃣ Train the model

```bash 
make train
```

Produces:
```bash
runs/best_model.pt
```
3️⃣ Run inference
```bash
make infer
```
Outputs:
```bash
outputs/change_mask_inferred.tif
outputs/preview.png
```
4️⃣ Batch inference for multiple AOIs
```bash
make infer-multi
```
Outputs under:
```bash
outputs/<AOI>/change_mask.tif
outputs/<AOI>/preview.png
```

⸻

## 📈 Results Visualization

| AOI | Visualization | Description |
|:----|:--------------|:-------------|
| **Chicago** | ![Chicago](examples/outputs/Chicago/preview.png) | Detected new building footprints & road expansion |
| **San Jose** | ![SanJose](examples/outputs/SanJose/preview.png) | Suburban vegetation & infrastructure change |
| **Houston** | ![Houston](examples/outputs/Houston/preview.png) | Rapid construction zone development |


⸻

## 🧭 Integrations
	•	Vertex AI Notebooks: ready for GPU/TPU training
	•	GCS Storage Mount: for large GeoTIFF ingestion
	•	Esri ArcGIS Pro / ArcGIS Online: generated masks can be imported as raster layers for spatial analysis.

⸻

## 🧰 Environment Variables

| Variable | Default | Description |
|:----------|:----------|:-------------|
| `RAW_PATH` | `raw` | Sentinel-2 download directory |
| `INTERIM_PATH` | `interim` | Aligned rasters |
| `PROC_PATH` | `processed` | Patch dataset |
| `RUNS_PATH` | `runs` | Model checkpoints |
| `OUT_PATH` | `outputs` | Inference outputs |


⸻

📜 Example Command (Custom AOI)
```bash
python scripts/infer_full.py \
  --t0_path interim/MyCity/t0_aligned.tif \
  --t1_path interim/MyCity/t1_aligned.tif \
  --ckpt runs/best_model.pt \
  --out_path outputs/MyCity/change_mask.tif
```

## 🧪 Performance Tips

- ✅ Use **GPU batch inference** with `patch=512`  
- ✅ Keep **NumPy < 2.0** for PyTorch compatibility  
- ✅ **Downsample large rasters** for safe visualization  
- ✅ Run **multiple AOIs sequentially** (not parallel)  

---

## 🏗️ Future Work

- 🌎 Integrate **12-band Sentinel-2 MSI** (B02–B12)  
- 🧮 Add **attention modules (CBAM)** for better feature fusion  
- ⚙️ Migrate training to **PyTorch Lightning**  
- ☁ Deploy inference API on **Vertex AI Prediction or Cloud Run**  
- 🗺 Export masks as **GeoJSON / shapefile** via `rasterio.features.shapes`  

---

## 🧑‍💻 Author

**Mahender Reddy Pokala**  

---

## 📄 License

**MIT License** — feel free to fork and modify, but please cite this work if used in research.

```bibtex
@software{pokala2025_geovision,
  author = {Pokala, Mahender Reddy},
  title = {GeoVision: Satellite Change Detection using Siamese U-Net},
  year = {2025},
  url = {https://github.com/mahenderreddyp/geovision-change-detection}
}