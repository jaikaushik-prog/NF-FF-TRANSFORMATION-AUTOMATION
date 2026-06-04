# Near-Field Antenna Measurement & ML-Based Anomaly Detection

> **Automated anomaly detection, phase drift calibration, and correction for near-field antenna radiation pattern measurements using a PyTorch 1D Convolutional Autoencoder + LightGBM gradient-boosted regressor, paired with MATLAB Near-Field to Far-Field (NF-FF) transformation scripts — all wrapped in an interactive Streamlit dashboard.**

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution Architecture](#solution-architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [GUI Dashboard](#gui-dashboard)
  - [Launching the Dashboard](#launching-the-dashboard)
  - [Dashboard Features](#dashboard-features)
  - [Sidebar Controls](#sidebar-controls)
  - [Visualisation Tabs](#visualisation-tabs)
- [Usage](#usage)
  - [Python: End-to-End Pipeline (CLI)](#python-end-to-end-pipeline-cli)
  - [Python: Anomaly Detection Pipeline](#python-anomaly-detection-pipeline)
  - [Python: LightGBM Phase Drift Calibration](#python-lightgbm-phase-drift-calibration)
  - [Python: NF-FF Transformation (Planar & Circular)](#python-nf-ff-transformation-planar--circular)
  - [MATLAB: NF-FF Transformation (Planar)](#matlab-nf-ff-transformation-planar)
  - [MATLAB: NF-FF Transformation (Circular)](#matlab-nf-ff-transformation-circular)
- [How It Works](#how-it-works)
  - [1. Data Ingestion](#1-data-ingestion)
  - [2. Hybrid Training Approach](#2-hybrid-training-approach)
  - [3. Autoencoder Architecture](#3-autoencoder-architecture)
  - [4. Adaptive Anomaly Detection](#4-adaptive-anomaly-detection)
  - [5. Smart Interpolation](#5-smart-interpolation)
  - [6. LightGBM Phase Drift Calibration](#6-lightgbm-phase-drift-calibration)
  - [7. Pipeline Orchestration](#7-pipeline-orchestration)
- [Results](#results)
- [Data Format](#data-format)
- [Configuration & CLI Options](#configuration--cli-options)
- [Workflow Integration](#workflow-integration)
- [Theoretical Background](#theoretical-background)
- [Troubleshooting](#troubleshooting)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

This project provides a complete pipeline for **near-field antenna measurement processing** at 10 GHz, combining:

1. **Anomaly Detection** — A PyTorch 1D Convolutional Autoencoder that automatically detects and corrects measurement anomalies (stepper-motor jitter, cable reflections, RF noise spikes) in near-field sweep data.

2. **Phase Drift Calibration** — A LightGBM gradient-boosted regressor that predicts and corrects thermal phase drift caused by VNA/cable temperature fluctuations during long chamber scans.

3. **Electromagnetic Theory** — Python and MATLAB scripts that perform Near-Field to Far-Field (NF-FF) transformations using both **planar** (2D FFT / Plane-Wave Spectrum) and **circular** (Cylindrical Mode Expansion) methods, with evanescent mode filtering and Spatial Nyquist safeguards.

4. **Interactive GUI Dashboard** — A Streamlit-based lab benchtop interface (`app.py`) with interactive Plotly visualisations including 3D radiation pattern surfaces, U-V heatmaps, polar plots, cylindrical-mode spectra, and real-time pipeline progress tracking.

5. **Unified Pipeline Orchestrator** — A backend integration layer (`main.py`) that chains all processing stages (ingestion → anomaly detection → phase calibration → NF-FF transform) into a single callable pipeline with strict shape validation and progress callbacks.

6. **Smart Data Ingestion** — An auto-detecting parser (`ingestion.py`) that handles CST planar exports, simple VNA CSVs, and extended measurement CSVs with temperature/time metadata — all normalised into a standardised format.

The ML pipeline replaces the brittle manual `sort → unique → interpolate` approach with an intelligent system that learns the physics of smooth radiation patterns and flags only the corrupted data points.

---

## Problem Statement

In near-field antenna measurement chambers, several hardware issues corrupt raw data:

| Issue | Cause | Effect on Data |
|-------|-------|----------------|
| **Motor jitter** | Stepper motor skipping/repeating steps | Missing or duplicated angular positions; sharp amplitude spikes |
| **Cable reflections** | RF cable flexing during rotation | Sudden phase discontinuities and amplitude spikes |
| **Thermal drift** | Temperature changes in the anechoic chamber | Slow baseline magnitude/phase drift |
| **VNA noise** | Vector Network Analyzer thermal noise floor | Random low-level noise across all points |

### The Old Way (Manual)

The traditional MATLAB approach (`circular_nfff_legacy.m`, lines 19-53):
1. Shift angles by 180 degrees to avoid wrap-around issues
2. Sort all data by angle
3. Remove duplicate angles via `unique()`
4. Linearly interpolate to fill gaps
5. Pad to a full 360-degree canvas

**Problems:** This blindly interpolates *everything* — both missing points (correct) and anomalous spikes (incorrect). It cannot distinguish between a legitimate measurement gap and a corrupted data point.

### The New Way (Autoencoder)

The autoencoder learns what a *clean* radiation pattern looks like from real CST simulation physics. When it encounters a motor-jitter spike, it *cannot reconstruct it*, causing the reconstruction error to spike at exactly that coordinate. Only the flagged points are interpolated, preserving all valid measurement data.

---

## Solution Architecture

```
                    ┌─────────────────────────────┐
                    │   CST Simulation Data        │
                    │   (Simulated_NF_Data.txt)    │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Data Ingestion (auto-      │
                    │   detect format: CST / CSV   │
                    │   / Extended VNA)             │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Data Augmentation          │
                    │   • Gaussian noise           │
                    │   • Amplitude scaling        │
                    │   • Angular jitter           │
                    │   • Baseline drift           │
                    │   • Normalisation [-1, 1]    │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   1D Conv Autoencoder        │
                    │   Encoder: 360→180→90→45     │
                    │   Decoder: 45→90→180→360     │
                    │   (500 epochs, MSE loss)     │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Anomaly Detection          │
                    │   Adaptive threshold:        │
                    │   mean + 3*std               │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Smart Interpolation        │
                    │   Fix ONLY flagged points    │
                    │   using clean neighbours     │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   LightGBM Phase Drift       │
                    │   Calibration                │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   NF-FF Transformation       │
                    │   Planar (PWS) or Circular   │
                    │   (CME + Hankel)             │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │   Streamlit Dashboard        │
                    │   Interactive Plotly plots   │
                    │   3D surfaces, polar, UV     │
                    └─────────────────────────────┘
```

---

## Project Structure

```
NF-FF-TRANSFORM-ALGORITHM-main/
├── app.py                      ← Streamlit GUI dashboard
├── main.py                     ← Pipeline orchestrator (backend)
├── requirements.txt            ← Python package dependencies
├── README.md
├── src/
│   ├── python/
│   │   ├── ingestion.py        ← Auto-detecting VNA/CST data parser
│   │   ├── nf_autoencoder.py   ← PyTorch anomaly detection pipeline
│   │   ├── nf_calibration.py   ← LightGBM phase drift calibration
│   │   ├── nf_ff_planar.py     ← Planar NF-FF (2D FFT / PWS)
│   │   └── nf_ff_circular.py   ← Circular NF-FF (CME + Hankel)
│   └── matlab/
│       ├── planar_nfff_legacy.m
│       ├── circular_nfff_legacy.m
│       └── untitled3.m         ← Additional MATLAB analysis script
├── data/
│   ├── raw/
│   │   └── Simulated_NF_Data.txt
│   └── processed/
│       ├── cleaned_sweep.csv
│       └── calibrated_sweep.csv
├── models/
│   ├── nf_autoencoder.pth
│   └── lightgbm_calibration_model.txt
├── docs/
│   ├── Instead of manually sorting arrays.txt
│   └── A System Architecture Breakdown of a DSP-Based Modern Wireline Transceiver.docx
└── results/
    ├── nf_anomaly_results.png
    └── calibration_diagnostic.png
```

### File Descriptions

| File | Language | Description |
|------|----------|-------------|
| `app.py` | Python | **Streamlit GUI dashboard** — interactive lab benchtop interface with Plotly 3D radiation surfaces, U-V heatmaps, polar plots, anomaly diagnostics, per-point error charts, pipeline progress tracking, metric cards, and stage status badges. Launch with `streamlit run app.py`. |
| `main.py` | Python | **Pipeline orchestrator** — chains all processing modules (ingestion → autoencoder → LightGBM → NF-FF transform) into a single `MetrologyPipeline` class with `PipelineConfig` dataclass, `PipelineStage` enum, strict inter-stage shape validation, and progress callbacks for the Streamlit front-end. |
| `ingestion.py` | Python | **Data ingestion module** — auto-detects and parses three input formats: CST planar exports (whitespace-separated, 9 columns), simple VNA CSVs (`angle_deg, mag_dB, phase_deg`), and extended CSVs with temperature/time metadata. Outputs a standardised `IngestedData` dataclass. |
| `nf_autoencoder.py` | Python | End-to-end pipeline: loads CST data, augments, trains autoencoder, detects anomalies with adaptive threshold, smart-interpolates corrupted points, saves cleaned output and diagnostic plots |
| `nf_calibration.py` | Python | LightGBM phase drift calibration: synthetic drift data generation, 5-fold cross-validated training, inference on raw sweep files, 2-panel diagnostic plotting |
| `nf_ff_planar.py` | Python | Planar NF-FF transformation using 2D FFT and Plane-Wave Spectrum method, replacing `planar_nfff_legacy.m`. Generates U-V space far-field plots. |
| `nf_ff_circular.py` | Python | 1D circular NF-FF transformation with Cylindrical Mode Expansion, evanescent mode filtering, and Spatial Nyquist check, replacing `circular_nfff_legacy.m`. |
| `requirements.txt` | — | Python package dependencies (PyTorch, NumPy, Pandas, Matplotlib, LightGBM, scikit-learn, SciPy, Streamlit, Plotly) |
| `nf_autoencoder.pth` | Binary | Saved PyTorch model state dictionary (trained weights) |
| `lightgbm_calibration_model.txt` | Text | Saved LightGBM booster model for phase drift prediction |
| `cleaned_sweep.csv` | CSV | Anomaly-corrected output: `angle_deg`, `mag_dB`, `phase_deg` — ready for MATLAB |
| `calibrated_sweep.csv` | CSV | Phase-drift-corrected output with `calibrated_phase_deg` column |
| `nf_anomaly_results.png` | Image | Four-panel diagnostic: training loss, anomaly flags, reconstruction error, before/after |
| `calibration_diagnostic.png` | Image | Two-panel diagnostic: thermal drift landscape, phase before/after calibration |
| `Simulated_NF_Data.txt` | Data | CST planar near-field export at 10 GHz — 11x9 spatial grid (x, y, z, Ex, Ey, Ez complex components) |
| `planar_nfff_legacy.m` | MATLAB | Planar NF-FF transformation using 2D FFT and Plane-Wave Spectrum method |
| `circular_nfff_legacy.m` | MATLAB | 1D circular NF-FF transformation with Cylindrical Mode Expansion, evanescent mode filtering, Spatial Nyquist check, and proper FFT scaling |
| `untitled3.m` | MATLAB | Additional MATLAB analysis / prototyping script |

---

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| **Python** | >= 3.10 | ML pipeline runtime |
| **PyTorch** | >= 2.0.0 | Autoencoder model |
| **LightGBM** | >= 4.0.0 | Phase drift calibration |
| **scikit-learn** | >= 1.3.0 | Cross-validation & metrics |
| **NumPy** | >= 1.24.0 | Numerical computation |
| **Pandas** | >= 2.0.0 | CSV I/O & data manipulation |
| **Matplotlib** | >= 3.7.0 | Diagnostic plots (CLI mode) |
| **SciPy** | >= 1.11.0 | Hankel functions for circular NF-FF |
| **Streamlit** | >= 1.30.0 | Interactive GUI dashboard |
| **Plotly** | >= 5.18.0 | Interactive 3D/2D visualisations in dashboard |
| **MATLAB** | R2020a+ | Legacy NF-FF transformation scripts (optional) |

---

## Installation

```bash
# 1. Clone or navigate to the project
cd NF-FF-TRANSFORM-ALGORITHM-main/

# 2. Install Python dependencies
pip install -r requirements.txt
```

> **Note:** PyTorch will install the CPU-only version by default. For GPU acceleration (CUDA), install PyTorch separately following [pytorch.org](https://pytorch.org/get-started/locally/).

---

## GUI Dashboard

The project includes a full-featured **Streamlit-based lab benchtop interface** (`app.py`) that wraps the entire pipeline into an interactive web dashboard with real-time progress tracking and rich Plotly visualisations.

### Launching the Dashboard

```bash
streamlit run app.py
```

This opens the dashboard in your default browser at `http://localhost:8501`.

### Dashboard Features

| Feature | Description |
|---------|-------------|
| **One-click pipeline** | The 🚀 **Run Full Pipeline** button executes all four stages (Ingestion → Anomaly Detection → Phase Calibration → NF-FF Transform) with a live progress bar |
| **File upload** | Drag-and-drop VNA `.csv` or CST `.txt` files directly into the browser — auto-detected and parsed |
| **Demo mode** | Checkbox to process the included `Simulated_NF_Data.txt` without needing external data |
| **Metric cards** | At-a-glance KPIs: detected format, input point count, anomalies fixed, phase drift corrected, peak gain |
| **Stage timing** | Per-stage execution time breakdown (ingestion, anomaly, calibration, transform) |
| **Dark theme** | Premium dark UI with glassmorphism metric cards, gradient headers, Inter font, and indigo accent colours |

### Sidebar Controls

The sidebar provides full configuration without touching the command line:

| Control | Default | Description |
|---------|---------|-------------|
| **Frequency (GHz)** | 10.0 | Operating frequency for wavelength calculation |
| **Step Size (mm)** | 10.0 | Spatial sampling step for planar grids |
| **Probe Radius (m)** | 0.5 | Measurement radius for circular CME transforms |
| **FFT Size** | 256 | Zero-padded FFT size (64 / 128 / 256 / 512 / 1024) |

### Visualisation Tabs

The dashboard automatically selects the appropriate visualisation set based on the detected data geometry:

#### Planar Data (CST 2D Grid)

| Tab | Content |
|-----|---------|
| 🌐 **3D Surface** | Interactive 3D far-field radiation pattern with Plasma colourmap, evanescent-region masking, visible-region circle overlay, and orbit/zoom controls |
| 🗺️ **U-V Heatmap** | 2D U-V direction-cosine heatmap with dashed visible-region boundary |
| 📈 **Theta Cuts** | Principal-plane cuts at φ=0° and φ=90° with normalised amplitude |

#### Circular Data (1D Angular Sweep)

| Tab | Content |
|-----|---------|
| 🎯 **Polar Plot** | Interactive polar overlay of near-field and far-field patterns |
| 📈 **Cartesian Overlay** | NF vs FF amplitude comparison on a Cartesian axis |
| 🔬 **Mode Spectrum** | Cylindrical-mode spectrum before and after Hankel compensation, with propagating-mode region highlighted |

#### Anomaly Diagnostics (Always Shown)

| Panel | Content |
|-------|---------|
| **Left** | Before vs after repair — raw sweep (grey), autoencoder reconstruction (blue dashed), cleaned (green), anomaly markers (red ✕) |
| **Right** | Per-point reconstruction error with filled area chart and adaptive threshold (μ+3σ) horizontal line |

#### Detailed Stage Status

An expandable section at the bottom shows pass/skip badges for each pipeline stage with execution times and a full JSON metrics dump.

---

## Usage

### Python: End-to-End Pipeline (CLI)

The `main.py` orchestrator runs all four stages in sequence without the GUI:

```bash
python main.py
```

This will:
1. Ingest the CST demo data (`data/raw/Simulated_NF_Data.txt`)
2. Run autoencoder anomaly detection (loads pre-trained weights or trains from scratch)
3. Run LightGBM phase drift calibration (synthetic demo if no temperature metadata)
4. Compute the NF-FF transformation (planar PWS for 2D grids, circular CME for 1D sweeps)
5. Print all pipeline metrics and per-stage execution times

You can also import and use the pipeline programmatically:

```python
from main import MetrologyPipeline, PipelineConfig

config = PipelineConfig(freq_ghz=10.0, n_fft=512)
pipe = MetrologyPipeline(config)
results = pipe.run_full_sweep("data/raw/Simulated_NF_Data.txt")

print(results["metrics"])          # Dict of all KPIs
print(results["stages_completed"]) # ['ingestion', 'anomaly', 'calibration', 'transform']
print(results["timings"])          # Per-stage wall-clock times
```

#### PipelineConfig Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `freq_ghz` | 10.0 | Operating frequency |
| `step_mm` | 10.0 | Spatial sampling step |
| `n_fft` | 256 | FFT zero-padding size |
| `n_sweep_points` | 360 | Number of angular points for 1D sweep resampling |
| `r_probe_m` | 0.5 | Probe radius for circular CME |
| `autoencoder_weights` | `models/nf_autoencoder.pth` | Path to saved PyTorch weights |
| `calibration_model` | `models/lightgbm_calibration_model.txt` | Path to saved LightGBM model |
| `cst_base_file` | `data/raw/Simulated_NF_Data.txt` | CST base file for autoencoder training |
| `device` | `cpu` | PyTorch device (`cpu` or `cuda`) |
| `autoencoder_epochs` | 500 | Training epochs if weights don't exist |
| `autoencoder_batch` | 64 | Training batch size |

---

### Python: Anomaly Detection Pipeline

#### Full Pipeline (Train + Demo)

```bash
python src/python/nf_autoencoder.py
```

This will:
1. Load the CST simulation data (`Simulated_NF_Data.txt`)
2. Train the autoencoder for 500 epochs (~20 seconds on CPU)
3. Save model weights to `nf_autoencoder.pth`
4. Inject 8 synthetic anomalies into the CST base pattern
5. Run anomaly detection with adaptive threshold
6. Smart-interpolate only the flagged points
7. Save `cleaned_sweep.csv` and `nf_anomaly_results.png`

#### Inference on Real Measurement Data

```bash
python src/python/nf_autoencoder.py --infer your_raw_sweep.csv
```

The CSV must have 3 columns: `angle_deg`, `mag_dB`, `phase_deg`.

#### Custom CST Base File

```bash
python src/python/nf_autoencoder.py --cst YourCSTExport.txt
```

#### Suppress Plot Window

```bash
python src/python/nf_autoencoder.py --no-plot
```

#### Custom Training Parameters

```bash
python src/python/nf_autoencoder.py --epochs 1000 --batch-size 128
```

---

### Python: LightGBM Phase Drift Calibration

#### Train on Synthetic Data

```bash
python src/python/nf_calibration.py --mode train
```

This will:
1. Generate 10,000 physics-informed synthetic measurements (4-hour scan, 22-26 C drift)
2. Train a LightGBM regressor with 5-fold cross-validation
3. Print MAE, RMSE, and R2 metrics per fold
4. Save the model to `lightgbm_calibration_model.txt`
5. Run a demo calibration on the synthetic data
6. Save `calibrated_sweep.csv` and `calibration_diagnostic.png`

#### Calibrate Real Measurement Data

```bash
python src/python/nf_calibration.py --mode calibrate --input raw_sweep.csv
```

The CSV must contain columns: `motor_angle_deg`, `ambient_temp_c`, `elapsed_time_min`, `raw_phase_deg`.

#### Custom Training Size

```bash
python src/python/nf_calibration.py --mode train --samples 20000
```

---


### Python: NF-FF Transformation (Planar & Circular)

The Python equivalents of the legacy MATLAB scripts provide identical mathematical operations using optimized NumPy and SciPy operations.

#### Planar NF-FF (Plane-Wave Spectrum)

```bash
python src/python/nf_ff_planar.py --input Simulated_NF_Data.txt
```
This script computes the 2D FFT, filters evanescent modes (U^2 + V^2 > 1), and generates a 4-panel diagnostic plot with near-field amplitude and far-field radiation patterns in U-V space.

#### Circular NF-FF (Cylindrical Mode Expansion)

```bash
python src/python/nf_ff_circular.py --input cleaned_sweep.csv
```
This script computes the 1D FFT, applies CME compensation using Hankel functions, filters evanescent modes (|n| > k * R_probe), and plots the 1D polar pattern.

---

### MATLAB: NF-FF Transformation (Planar)

Open `src/matlab/planar_nfff_legacy.m` in MATLAB. This script:

1. Loads `Simulated_NF_Data.txt` (CST planar export)
2. Extracts complex Ex and Ey field components
3. Reshapes into an 11x9 2D spatial grid
4. Plots near-field amplitude |Ex| in dB
5. Performs **2D FFT** (zero-padded to 256x256) to compute the Plane-Wave Spectrum
6. Maps to U-V angular coordinates (direction cosines)
7. Filters evanescent waves (U^2 + V^2 > 1)
8. Plots the 2D far-field radiation pattern

**Key Parameters:**
- Frequency: 10 GHz (lambda = 30 mm)
- Spatial sampling: dx = dy = 10 mm
- FFT size: 256x256

---

### MATLAB: NF-FF Transformation (Circular)

Open `src/matlab/circular_nfff_legacy.m` in MATLAB. This script now reads the **Python-cleaned output** directly:

1. **Spatial Nyquist Check** — validates that the angular step size is fine enough for the configured probe distance
2. Loads `cleaned_sweep.csv` (output from the Python autoencoder pipeline)
3. Converts magnitude (dB) and phase (degrees) to complex electric field
4. Performs **1D FFT** (scaled by 1/N for absolute magnitude preservation) for Cylindrical Mode Extraction
5. Applies **Cylindrical Mode Expansion (CME)** compensation using Hankel functions of the 2nd kind
6. **Evanescent mode filtering** — zeroes out non-propagating modes where |n| > k*R to prevent numerical noise amplification
7. Reconstructs far-field via inverse FFT
8. Plots near-field and far-field polar patterns

**Key Parameters:**
- Frequency: 10 GHz
- Probe distance (R_probe): 0.5 m
- Max propagating mode: floor(k * R_probe) = 104
- Dynamic range: -40 dB

> **Note:** The old manual stepper-motor fix (sort/unique/interpolate) has been removed. The Python autoencoder pipeline now handles anomaly detection and interpolation before the data reaches MATLAB.

---

## How It Works

### 1. Data Ingestion

The `VNADataIngester` class (`src/python/ingestion.py`) automatically detects and parses three input formats:

| Format | Detection Trigger | Columns | Output |
|--------|------------------|---------|--------|
| **CST Planar** | Header contains `ExRe` or `x [mm]` | 9 cols: x, y, z, ExRe, ExIm, EyRe, EyIm, EzRe, EzIm | 2D complex field grids + 1D centre-row profile |
| **Simple CSV** | Header contains `angle` but no temp/time | 3 cols: `angle_deg`, `mag_dB`, `phase_deg` | 1D angular sweep arrays |
| **Extended CSV** | Header contains `angle` + `temp` or `elapsed` | 5 cols: angle, mag, phase, temperature, elapsed_time | 1D sweep + thermal metadata for LightGBM |

The ingester uses flexible column-name aliasing (e.g., `angle_deg`, `angle`, `motor_angle_deg`, `theta`, `phi` all map to the angle column), making it robust to different lab export formats.

All three formats are normalised into an `IngestedData` dataclass containing:
- Standardised Pandas DataFrame
- 1D arrays: `angles`, `magnitudes`, `phases`
- 2D arrays (planar only): `Ex_2D`, `Ey_2D`, `x_mm`, `y_mm`
- Metadata flags: `has_temperature`, `has_elapsed_time`

### 2. Hybrid Training Approach

Instead of training on purely synthetic cosine patterns (academically weak) or requiring thousands of CST simulations (computationally prohibitive), this project uses a **hybrid approach**:

1. **Load ONE real CST simulation** — captures the true antenna physics: main lobe shape, side-lobe structure, null positions
2. **Extract a 1D angular profile** — takes the center row (y = 0) of |Ey|, maps spatial x-position to observation angle via theta = arctan(x/z)
3. **Generate thousands of augmented copies** — each perturbed with realistic measurement imperfections:

| Augmentation | Physical Basis | Range |
|-------------|----------------|-------|
| Gaussian noise | VNA thermal noise | sigma = 0.1 - 1.5 dB |
| Amplitude scaling | Cable loss / gain drift | x0.95 - x1.05 |
| Angular jitter | Small motor offset | +/- 2 sample points |
| Baseline drift | Temperature shift | +/- 0.5 dB linear ramp |

4. **Normalise to [-1, 1]** — maps the dB range [-50, 0] for stable autoencoder convergence

This gives the model 32,000 effectively unique training samples (500 epochs x 64 batch size) while preserving the real electromagnetic physics.

### 3. Autoencoder Architecture

```
Input (1 x 360)
    │
    ▼
┌─────────────────────────────────────┐
│  ENCODER                            │
│  Conv1d(1→16, k=7, s=2)  → 180     │
│  BatchNorm + LeakyReLU(0.2)        │
│  Conv1d(16→32, k=5, s=2) → 90      │
│  BatchNorm + LeakyReLU(0.2)        │
│  Conv1d(32→16, k=5, s=2) → 45      │
│  BatchNorm + LeakyReLU(0.2)        │
└─────────────┬───────────────────────┘
              │ Latent Space (16 x 45)
┌─────────────▼───────────────────────┐
│  DECODER                            │
│  ConvT1d(16→32, k=5, s=2) → 90     │
│  BatchNorm + LeakyReLU(0.2)        │
│  ConvT1d(32→16, k=5, s=2) → 180    │
│  BatchNorm + LeakyReLU(0.2)        │
│  ConvT1d(16→1, k=7, s=2)  → 360    │
└─────────────────────────────────────┘
    │
    ▼
Output (1 x 360)
```

The **bottleneck** (16 channels x 45 spatial points = 720 values) forces the network to learn only the smooth, physically meaningful structure. Sharp, localised motor-jitter spikes cannot be efficiently encoded — they get smoothed out in the reconstruction, causing the error to spike at exactly those coordinates.

**Training details:**
- Optimiser: Adam (lr = 1e-3)
- Scheduler: Cosine annealing
- Loss: Mean Squared Error (MSE)
- Epochs: 500
- Device: CPU (~20s) or CUDA GPU (~5s)

### 4. Adaptive Anomaly Detection

Rather than a brittle fixed threshold (e.g., 5 dB), the system computes an **adaptive threshold** from the reconstruction error distribution:

```
threshold = mean(errors) + 3 * std(errors)
```

This is equivalent to flagging any point whose reconstruction error falls outside the 99.7th percentile of the error distribution, making it robust across:
- Different antennas with varying pattern shapes
- Different SNR levels in various anechoic chambers
- Different frequency bands

### 5. Smart Interpolation

Unlike the old approach that interpolates *all* data uniformly, the smart interpolation:

1. Identifies **clean points** (those NOT flagged as anomalies)
2. Uses clean points as **anchors**
3. Linearly interpolates **only at the flagged coordinates**
4. Returns corrected magnitude and phase arrays

This preserves all valid measurement data and fixes only what is actually corrupted.

### 6. LightGBM Phase Drift Calibration

While the autoencoder handles sharp, localised anomalies (motor jitter, cable reflections), it cannot correct **slow, continuous thermal drift** that accumulates over long scans. The LightGBM calibration module addresses this complementary problem.

#### Drift Model

The thermal phase drift follows a non-linear model:

```
drift(T, t) = 0.5 * (T - T_ref)^1.5 + 0.02 * t
```

where `T` is the instantaneous chamber temperature (deg C), `T_ref` is the starting temperature, and `t` is elapsed time (minutes). This captures:
- **Cable dielectric heating** — the dominant non-linear T^1.5 term from thermal expansion of PTFE dielectric in coaxial cables
- **VNA LO drift** — a slow, monotonic frequency drift in the local oscillator, linear in time

#### Training Features

| Feature | Physical Basis |
|---------|---------------|
| `motor_angle_deg` | Angle-dependent cable flex and connector stress |
| `ambient_temp_c` | Primary driver of dielectric and LO drift |
| `elapsed_time_min` | Monotonic VNA aging / warm-up drift |
| `raw_phase_deg` | Allows the model to learn residual correlations |

#### Pipeline

1. `SyntheticDriftDataGenerator` creates physics-informed training data (simulating a 4-hour scan from 22 C to 26 C)
2. `CalibrationPipeline` trains an `LGBMRegressor` (800 estimators, depth 7, cosine LR) with 5-fold cross-validation
3. `calibrate_sweep()` predicts the drift at each point, subtracts it, and wraps to [-180, +180) degrees

#### Results

| Metric | Value |
|--------|-------|
| 5-Fold Mean MAE | 0.0037 deg |
| 5-Fold Mean RMSE | 0.0056 deg |
| 5-Fold Mean R2 | 0.999995 |
| Phase error reduction | 71.2% |

### 7. Pipeline Orchestration

The `MetrologyPipeline` class (`main.py`) chains all four processing stages with:

- **Strict shape validation** between stages (e.g., ensuring the anomaly detector's output length matches the calibrator's expected input)
- **Automatic format routing** — planar data goes to the PWS 2D-FFT transform; 1D angular sweeps go to the circular CME+Hankel transform
- **Progress callbacks** — the `progress_cb(stage, fraction, message)` function is called at key points, driving the Streamlit progress bar
- **Graceful fallbacks** — if no pre-trained weights exist, the autoencoder trains on the fly; if no temperature metadata exists, the calibrator runs a synthetic demo
- **Unified results dict** — all outputs (metrics, nearfield data, anomaly results, calibration results, farfield arrays, stage list, timings) are returned in a single dictionary

```
Pipeline Stages:
  1. INGESTION    →  VNADataIngester.parse()
  2. ANOMALY      →  NearFieldAutoencoder + detect_anomalies() + smart_interpolate()
  3. CALIBRATION  →  CalibrationPipeline + calibrate_sweep()
  4. TRANSFORM    →  PlanarNFFFTransformer or CircularNFFFTransformer
```

---

## Results

### Autoencoder Anomaly Detection

| Metric | Value |
|--------|-------|
| Final training loss (MSE) | 0.0017 |
| Adaptive threshold | 5.84 dB |
| **True positives** | **8/8 (100%)** |
| **False positives** | **0** |
| **Missed anomalies** | **0** |

The pipeline generates a 4-panel diagnostic plot (`nf_anomaly_results.png`):

| Panel | Content |
|-------|---------| 
| **Top-left** | Training loss curve (log scale) — shows convergence from ~2.1 to ~0.002 |
| **Top-right** | Raw sweep with detected anomalies (red dots) and ground-truth (green circles) |
| **Bottom-left** | Per-point reconstruction error with adaptive threshold line |
| **Bottom-right** | Before vs after repair — raw (grey), autoencoder reconstruction (blue dashed), cleaned (green) |

### LightGBM Phase Drift Calibration

| Metric | Value |
|--------|-------|
| 5-Fold Mean MAE | 0.0037 deg |
| 5-Fold Mean RMSE | 0.0056 deg |
| 5-Fold Mean R2 | 0.999995 |
| Phase error BEFORE | MAE = 4.150 deg |
| Phase error AFTER | MAE = 1.197 deg |
| **Improvement** | **71.2% reduction** |

The pipeline generates a 2-panel diagnostic plot (`calibration_diagnostic.png`):

| Panel | Content |
|-------|---------| 
| **Left** | Thermal drift landscape — colour-mapped scatter of drift vs temperature and elapsed time |
| **Right** | Phase comparison — true (green), corrupted (red), LightGBM-calibrated (blue dashed) |

---

## Data Format

### CST Planar Export (`Simulated_NF_Data.txt`)

Whitespace-separated, 2 header lines:

```
           x [mm]           y [mm]           z [mm]       ExRe [V/m]       ExIm [V/m]       EyRe [V/m]       EyIm [V/m]       EzRe [V/m]       EzIm [V/m]
---------------------------------------------------------------------------------------------------------------------------------------------------------
              -50              -40              115        6.6136875       -2.8198285        18.573277       -8.7962875        -2.506053       -28.628908
```

- **Grid:** 11 x-points (-50 to 50 mm, step 10) x 9 y-points (-40 to 40 mm, step 10) = 99 rows
- **Measurement plane:** z = 115 mm
- **Fields:** Complex Ex, Ey, Ez components (Real + Imaginary)

### Simple CSV (for `--infer`)

```csv
angle_deg,mag_dB,phase_deg
0,-42.3,12.5
1,-41.8,13.1
...
359,-43.1,11.9
```

### Extended VNA CSV (with temperature/time metadata)

```csv
motor_angle_deg,raw_mag_dB,raw_phase_deg,ambient_temp_c,elapsed_time_min
0,-42.3,12.5,22.1,0.0
1,-41.8,13.1,22.2,0.1
...
359,-43.1,11.9,25.8,239.4
```

> **Auto-detection:** The `VNADataIngester` class automatically detects which of the three formats above is provided, so you never need to specify the format manually.

### Cleaned Output (`cleaned_sweep.csv`)

Same format as the simple CSV above — ready to load into MATLAB for NF-FF transformation.

---

## Configuration & CLI Options

### nf_autoencoder.py

```
usage: nf_autoencoder.py [-h] [--cst CST] [--infer INFER] [--weights WEIGHTS]
                         [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--no-plot]

NF Autoencoder anomaly detector  (CST hybrid approach)

options:
  -h, --help            show this help message and exit
  --cst CST             Path to CST simulation file  (default: Simulated_NF_Data.txt)
  --infer INFER         CSV with columns: angle, mag_dB, phase_deg
  --weights WEIGHTS     Model weights path  (default: nf_autoencoder.pth)
  --epochs EPOCHS       Training epochs (default: 500)
  --batch-size BATCH_SIZE
                        Training batch size (default: 64)
  --no-plot             Suppress matplotlib window
```

### nf_calibration.py

```
usage: nf_calibration.py [-h] [--mode {train,calibrate}] [--input INPUT]
                         [--model MODEL] [--samples SAMPLES] [--output OUTPUT]
                         [--no-plot]

LightGBM Phase Drift Calibration for Near-Field Antenna Measurements (10 GHz)

options:
  -h, --help            show this help message and exit
  --mode {train,calibrate}
                        'train' = synthetic training + demo;
                        'calibrate' = inference on external file
  --input INPUT         Path to raw measurement CSV (required for calibrate)
  --model MODEL         Path to LightGBM model file (default: lightgbm_calibration_model.txt)
  --samples SAMPLES     Number of synthetic training samples (default: 10000)
  --output OUTPUT       Output CSV path for calibrated data
  --no-plot             Suppress matplotlib window
```

---


### nf_ff_planar.py & nf_ff_circular.py

```
usage: nf_ff_planar.py [-h] [--input INPUT] [--freq FREQ] [--n-fft N_FFT] [--no-plot] [--output OUTPUT]
usage: nf_ff_circular.py [-h] [--input INPUT] [--freq FREQ] [--r-probe R_PROBE] [--no-plot] [--output OUTPUT]
```

---

## Workflow Integration

The recommended end-to-end workflow for processing real antenna measurements:

### Option A: GUI Dashboard (Recommended)

```
Step 1:  Launch the dashboard
             streamlit run app.py

Step 2:  Upload your raw VNA/CST measurement file
             (or enable demo mode for Simulated_NF_Data.txt)

Step 3:  Adjust parameters in the sidebar
             (frequency, step size, probe radius, FFT size)

Step 4:  Click "🚀 Run Full Pipeline"
             → Watch the progress bar through all 4 stages

Step 5:  Explore interactive visualisations
             → 3D radiation surface / polar plots
             → Anomaly detection diagnostics
             → Stage-by-stage timing and metrics
```

### Option B: CLI Pipeline

```
Step 1:  Measure in the anechoic chamber
             |  (record angle, mag, phase, temperature, elapsed time)
             v
Step 2:  Export raw sweep as CSV
             |
     --------+--------
     |                |
     v                v
Step 3a: Anomaly      Step 3b: Phase Drift
  Detection            Calibration
  nf_autoencoder.py    nf_calibration.py
  --infer raw.csv      --mode calibrate
     |                    --input raw.csv
     v                |
  cleaned_sweep.csv   v
     |             calibrated_sweep.csv
     +--------+--------+
              |
              v
Step 4:  Review diagnostic plots
             |  (nf_anomaly_results.png + calibration_diagnostic.png)
             v
Step 5:  Load cleaned data
             |
             v
Step 6:  Run Python NF-FF scripts (or legacy MATLAB scripts)
             |  python src/python/nf_ff_circular.py --input calibrated_sweep.csv
             |
             v
Step 7:  Far-field radiation pattern
```

### Option C: Programmatic Pipeline

```python
from main import MetrologyPipeline, PipelineConfig

pipe = MetrologyPipeline(PipelineConfig(freq_ghz=10.0))
results = pipe.run_full_sweep("your_measurement.csv")

# Access all results
ff_pattern = results["farfield"]
anomalies  = results["anomaly"]
metrics    = results["metrics"]
```

> **Tip:** For best results, run the anomaly detection first (fixes sharp spikes), then run the calibration on the cleaned output (corrects slow drift). The two modules address complementary types of measurement corruption. The GUI dashboard and `main.py` pipeline handle this ordering automatically.

---

## Theoretical Background

### Near-Field to Far-Field Transformation

Antenna far-field measurements require a minimum distance of `2*D^2/lambda` (Fraunhofer distance), which can be impractically large for electrically large apertures. Near-field measurements are taken on a surface close to the antenna and mathematically transformed to obtain the far-field pattern.

### Planar Scanning (planar_nfff_legacy.m)

Uses the **Plane-Wave Spectrum** (PWS) method:
1. Sample the tangential E-field on a planar surface
2. Compute the 2D spatial Fourier Transform (FFT)
3. Each spectral component represents a plane wave propagating at angle (theta, phi)
4. Map spectral frequencies to direction cosines: U = sin(theta)*cos(phi), V = sin(theta)*sin(phi)
5. Filter evanescent waves (U^2 + V^2 > 1)

### Cylindrical Scanning (circular_nfff_legacy.m)

Uses **Cylindrical Mode Expansion** (CME):
1. Sample the E-field on a cylindrical surface (rotating the AUT)
2. Decompose into cylindrical modes via 1D FFT (scaled by 1/N for absolute magnitude)
3. Compensate for the near-field probe distance using Hankel functions of the 2nd kind
4. **Filter evanescent modes** — zero out modes where |n| > k*R_probe to prevent numerical noise amplification from near-zero Hankel denominators
5. Project modes to infinity: multiply by `(1i)^n`
6. Reconstruct far-field via inverse FFT

#### Spatial Nyquist Criterion

The script includes a runtime check to ensure the angular sampling rate satisfies:

```
delta_phi <= lambda / (2 * R_probe)    (in radians)
```

For 1-degree steps (N=360), the maximum resolvable mode is n=180. At 10 GHz with R_probe=0.5 m, the physical cutoff is k*R = 104.7, so 1-degree steps are safely above the Nyquist limit. However, if R_probe exceeds ~0.86 m, a finer angular resolution would be required.

### Autoencoder Anomaly Detection

An autoencoder trained on clean data learns to reconstruct only the smooth, physically plausible patterns. When presented with a corrupted input, the sharp anomalous features lie outside the learned manifold and produce high reconstruction error — providing a natural anomaly score at every measurement point.

### Gradient-Boosted Phase Calibration

LightGBM is used for phase drift correction because:
- **Speed** — trains in ~3 seconds on CPU (vs minutes for neural networks)
- **Interpretability** — feature importance shows that `ambient_temp_c` is the dominant predictor, matching the physics
- **Tabular data** — gradient-boosted trees consistently outperform neural networks on structured/tabular feature sets
- **Low latency** — inference takes <1 ms for 15,000 points

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `UnicodeEncodeError: 'charmap' codec` | Set environment variable: `$env:PYTHONIOENCODING = "utf-8"` before running |
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'lightgbm'` | Run `pip install lightgbm scikit-learn` |
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install streamlit plotly` |
| `ModuleNotFoundError: No module named 'plotly'` | Run `pip install plotly>=5.18.0` |
| Streamlit dashboard won't launch | Ensure `streamlit` is installed and run `streamlit run app.py` (not `python app.py`) |
| Dashboard shows "Pipeline Error" | Check the error message — usually a missing data file or incompatible input format |
| High training loss (> 100) | Data normalisation may be missing — ensure the augmentation pipeline normalises to [-1, 1] |
| 0 anomalies detected | Check that the threshold isn't too high; try lowering the sigma multiplier from 3.0 to 2.5 |
| MATLAB `readmatrix` error | Ensure `Simulated_NF_Data.txt` uses whitespace delimiters and has 2 header lines |
| MATLAB far-field has spikes | Evanescent mode filter not applied — update `circular_nfff_legacy.m` to zero modes where abs(n) > k*R_probe |
| MATLAB `Spatial Nyquist` warning | Probe is too far for 1-degree steps — either move probe closer or reduce angular step size |
| GPU out of memory | Use `--batch-size 32` or run on CPU (default) |

---

## Future Work

- [x] ~~**LightGBM calibration predictor**~~ — Completed! Predicts and corrects thermal phase drift using gradient-boosted regression
- [x] ~~**Combined pipeline**~~ — Completed! `main.py` chains autoencoder anomaly detection and LightGBM calibration into a single end-to-end `MetrologyPipeline` class
- [x] ~~**Interactive GUI**~~ — Completed! Streamlit dashboard (`app.py`) with Plotly 3D surfaces, polar plots, U-V heatmaps, and real-time pipeline progress tracking
- [x] ~~**Smart data ingestion**~~ — Completed! Auto-detecting parser (`ingestion.py`) handles CST, simple CSV, and extended VNA CSV formats
- [ ] **2D anomaly detection** — Extend the autoencoder to 2D convolutional for planar near-field grids
- [ ] **Real-time inference** — Integrate with the measurement controller for live anomaly flagging during acquisition
- [ ] **Transfer learning** — Fine-tune the pretrained model on a small set of real measurements from a specific antenna
- [ ] **ONNX export** — Convert the PyTorch model to ONNX for deployment in MATLAB via the Deep Learning Toolbox
- [ ] **Real drift data training** — Replace synthetic drift data with logged temperature/phase data from actual chamber scans

---

## License

This project is for academic and research use. Please cite appropriately if used in publications.

---

*Built for near-field antenna measurement research at 10 GHz.*
