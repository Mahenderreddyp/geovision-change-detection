# GeoVision: Satellite Change Detection System

A production-ready deep learning pipeline for automated temporal change detection from Sentinel-2 satellite imagery. This system identifies urban growth, vegetation loss, and infrastructure development across multiple geographic regions using a Siamese U-Net architecture.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

GeoVision provides an end-to-end solution for detecting changes in satellite imagery over time. The system processes Sentinel-2 Level-2A data through a complete pipeline from acquisition to visualization, producing pixel-level change detection masks suitable for GIS integration and spatial analysis.

**Key Features:**
- Automated Sentinel-2 data acquisition via STAC API with cloud filtering
- Robust preprocessing pipeline with geospatial alignment and NDVI-based weak labeling
- Trained Siamese U-Net model for accurate change detection (IoU: 0.556)
- GPU-optimized inference for full-scene processing
- Multi-region support with automated visualization generation
- Cloud-ready integration with Vertex AI and Google Cloud Storage

## Model Performance

Our trained Siamese U-Net model demonstrates strong generalization across diverse geographic regions:

| Metric | Value |
|--------|-------|
| **Best IoU** | **0.856** |
| **Training Loss** | 0.001 (final) |
| **Model Size** | ~28 MB |
| **Inference Resolution** | 10 m/pixel |
| **Training Patches** | 38 (256×256) |
| **Training Epochs** | 200 |

The model checkpoint is included at `runs/best_model.pt` and ready for immediate deployment.

## Data Specifications

**Satellite Data Source:** Sentinel-2 Level-2A (ESA Copernicus Program)  
**Provider:** [Microsoft Planetary Computer STAC Catalog](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a)

**Technical Details:**
- Spatial Resolution: 10 m/pixel
- Spectral Bands: B02 (Blue), B03 (Green), B04 (Red)
- Cloud Coverage Filter: < 20%
- Scene Classification: SCL layer for quality masking

**Validated Regions:**
- Chicago, IL: June 2022 → June 2023
- San Jose, CA: July 2022 → July 2023
- Houston, TX: August 2022 → August 2023

## Architecture

The system employs a Siamese U-Net architecture optimized for temporal change detection:

```
Input: 6 channels (RGB × 2 timestamps)
  ↓
Siamese U-Net Encoder-Decoder
  ↓
Feature Difference Layer
  ↓
Sigmoid Activation
  ↓
Output: Binary Change Mask (1 channel)
```

**Training Configuration:**
- Loss Function: Binary Cross-Entropy with Logits
- Optimizer: Adam (learning rate: 1e-3)
- Hardware: NVIDIA L4 GPU (Vertex AI)
- Framework: PyTorch 2.1 with CUDA 12

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training/inference)
- 16 GB RAM minimum

### Setup

```bash
# Clone the repository
git clone https://github.com/mahenderreddyp/geovision-change-detection.git
cd geovision-change-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Using the Pre-trained Model

The repository includes a trained model ready for inference:

```bash
# Run inference on a single region
make infer

# Run batch inference on multiple regions
make infer-multi
```

### Training from Scratch

```bash
# Train the model with default configuration
make train
```

### Custom Inference

```bash
python scripts/infer_full.py \
  --t0_path interim/YourCity/t0_aligned.tif \
  --t1_path interim/YourCity/t1_aligned.tif \
  --ckpt runs/best_model.pt \
  --out_path outputs/YourCity/change_mask.tif
```

## Project Structure

```
geovision-change-detection/
├── src/geovision/
│   ├── model.py              # Siamese U-Net implementation
│   ├── data.py               # Dataset and patch extraction
│   ├── train.py              # Training loop with metrics
│   ├── infer.py              # GPU-optimized inference
│   └── utils.py              # Preprocessing utilities
├── scripts/
│   ├── train.py              # Training script
│   ├── infer_full.py         # Single AOI inference
│   └── infer_multi_aoi.py    # Multi-region batch processing
├── geovision_bucket/
│   ├── raw/                  # Downloaded Sentinel-2 data
│   ├── interim/              # Aligned and preprocessed rasters
│   ├── processed/            # Training patches (X.npy, y.npy)
│   ├── runs/                 # Model checkpoints
│   └── outputs/              # Inference results and visualizations
├── requirements.txt
├── Makefile
└── README.md
```

## Pipeline Stages

| Stage | Description | Output |
|-------|-------------|--------|
| **Data Acquisition** | Query STAC catalog for low-cloud Sentinel-2 scenes | Raw GeoTIFFs |
| **Preprocessing** | Reproject to common CRS, align timestamps | Aligned rasters |
| **Weak Labeling** | NDVI differencing with Otsu thresholding | Binary masks |
| **Patch Generation** | Extract 256×256 patches with 50% overlap | NumPy arrays |
| **Training** | Siamese U-Net with BCE loss | Model checkpoint |
| **Inference** | GPU-accelerated tiled prediction | Change masks |
| **Visualization** | Generate previews and interactive maps | PNG/HTML outputs |

## Results

### Chicago: Urban Development

Detected new building footprints, road expansion, and infrastructure development in metropolitan areas.

![Chicago Results](examples/outputs/viz_prediction.png)

## Integration Capabilities

**Cloud Platforms:**
- Google Cloud Platform (Vertex AI, Cloud Storage)
- AWS (S3, SageMaker compatible)
- Azure (Blob Storage, Machine Learning)

**GIS Software:**
- Esri ArcGIS Pro / ArcGIS Online
- QGIS (native GeoTIFF support)
- Google Earth Engine (export compatible)

**API Deployment:**
- Vertex AI Prediction endpoints
- Cloud Run containerized services
- FastAPI REST API wrapper (customizable)

## Configuration

Environment variables can be configured via `.env` file:

```bash
RAW_PATH=raw                    # Sentinel-2 download directory
INTERIM_PATH=interim            # Aligned raster storage
PROC_PATH=processed             # Training patch location
RUNS_PATH=runs                  # Model checkpoint directory
OUT_PATH=outputs                # Inference output location
```

## Performance Optimization

**Inference Best Practices:**
- Use GPU with batch size 8-16 for optimal throughput
- Enable memory-efficient tiling for large rasters (>10000×10000 pixels)
- Apply spatial downsampling for visualization (recommended: 25% for preview generation)

**Training Recommendations:**
- Maintain NumPy < 2.0 for PyTorch compatibility
- Use mixed precision training (torch.cuda.amp) for faster convergence
- Monitor GPU memory usage with batch size adjustments

## Roadmap

**Planned Enhancements:**
- [ ] Full 12-band Sentinel-2 MSI integration (B02–B12)
- [ ] Attention mechanisms (CBAM, SE-blocks) for improved feature fusion
- [ ] PyTorch Lightning migration for distributed training
- [ ] REST API deployment with FastAPI and Docker
- [ ] Vector export (GeoJSON/Shapefile) via rasterio.features
- [ ] Time-series analysis for multi-temporal change tracking
- [ ] Pre-trained weights for transfer learning

## Citation

If you use this work in your research, please cite:

```bibtex
@software{pokala2025_geovision,
  author = {Pokala, Mahender Reddy},
  title = {GeoVision: Satellite Change Detection using Siamese U-Net},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/mahenderreddyp/geovision-change-detection}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

**Mahender Reddy Pokala**  
[GitHub](https://github.com/mahenderreddyp) | [LinkedIn](https://www.linkedin.com/in/mahenderreddyp)

## Acknowledgments

- ESA Copernicus Programme for Sentinel-2 data
- Microsoft Planetary Computer for STAC API access
- PyTorch and rasterio communities for excellent geospatial tools
