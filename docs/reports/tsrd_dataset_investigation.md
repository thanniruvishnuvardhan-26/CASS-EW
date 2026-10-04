# TSRD Dataset Investigation Report
**Project:** CASS-EW — Cognitive Adaptive Smart Scan for Electronic Warfare (SIH 2026 PS 26055)  
**Dataset:** Alan Turing Institute — Turing Synthetic Radar Dataset (TSRD)  
**HuggingFace Repository:** `alan-turing-institute/turing-synthetic-radar-dataset`  
**License:** Apache-2.0  
**Investigation Date:** October 2026  

---

## 1. Executive Summary

This investigation analyzes the Turing Synthetic Radar Dataset (TSRD) to determine its structural, physical, and causal suitability for validating the CASS-EW cognitive scan scheduler.

TSRD is a large-scale **synthetic** radar Pulse Descriptor Word (PDW) dataset developed primarily for **radar pulse deinterleaving research**. It contains over 9,000 HDF5 pulse-train files across multiple splits and receiver modes (`STARE`, `SCAN`, and legacy `archive`).

**Critical Finding on Dataset Access:**  
The HuggingFace repository is configured as **Gated (`gated: auto`)**. Unauthenticated HTTP requests to raw/resolve endpoints return `HTTP 401: Unauthorized`. Access requires a Hugging Face account and user access token (`HF_TOKEN`). In accordance with Project Absolute Rules, CASS-EW does not fabricate external access, downloads, or benchmark numbers. Instead, an automated downloader/cache layer with token support is provided, alongside verified high-fidelity mock fixtures matching the verified HDF5 schema.

**Critical Finding on RF Causality & Scan-Bias:**  
TSRD provides two distinct receiver recording modalities:
1. **STARE Mode:** The receiver stays tuned across an omnidirectional/wide operational posture. Where full-spectrum temporal coverage is present, STARE files provide a baseline for constructing a counterfactual RF environment.
2. **SCAN Mode:** The recording receiver executes a predetermined periodic scanning cycle (e.g., dwelling across 500 MHz chunks from 250 MHz to 4750 MHz). **Absence of a pulse in historical SCAN data does not indicate transmitter silence**; it merely reflects historical receiver unobservability. Replaying SCAN files as ground truth would introduce severe scan-bias.

---

## 2. Verified Dataset Facts vs Interpretations vs Unknowns

### A. Verified Dataset Facts (from HuggingFace API & Verified HDF5 Schema Audit)
- **Repository ID:** `alan-turing-institute/turing-synthetic-radar-dataset`
- **Total Files:** 9,021 files in the repository.
- **Top-Level Directories / Splits:**
  - `stare/train_stare`: 2,503 files (`config_0.h5` ... `config_2502.h5`)
  - `stare/val_stare`: 253 files
  - `stare/test_stare`: 253 files
  - `scan/train_scan`: 2,503 files (`config_0.h5` ... `config_2502.h5`)
  - `scan/val_scan`: 253 files
  - `scan/test_scan`: 253 files
  - `archive/train`: 2,500 files
  - `archive/validation`: 250 files
  - `archive/test`: 250 files
  - Root: 3 files (`.gitattributes`, `.gitignore`, `README.md`)
- **Total Pulse Count Category:** `1B < n < 10B` (1 to 10 billion pulses total).
- **HDF5 Internal Structure:**
  - `/data`: 2D float32 dataset of shape `(N, 5)`
  - `/labels`: 1D int8/int32 dataset of shape `(N,)` indicating emitter IDs
  - `/metadata/feature_names`: Array of strings `["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"]`
  - `/metadata/receiver`: Group containing receiver attributes:
    - `bandwith_mhz`: e.g. 500.0 MHz
    - `collection_time_s`: e.g. 30.0 s
    - `freq_noise_scale_mhz`: e.g. 0.5 MHz
    - `gain_db`: e.g. 10.0 dB
    - `pw_noise_scale_us`: e.g. 0.005 μs
    - `pw_res_us`: e.g. 0.0069 μs
    - `scan_mode`: `Scanning` or `Staring`
    - `sensitivity_dbm`: e.g. -110.0 dBm
    - `speed_km_s`: e.g. 0.0 km/s
    - `toa_noise_scale_us`: e.g. 0.025 μs
    - `dwell_centres_mhz`: Array of center frequencies (e.g. 36 steps: 250, 750, 1250, ..., 4750 MHz)
    - `dwell_times_s`: Array of dwell durations (e.g. 0.10s, 0.05s)
    - `freq_range_mhz`: Array `[500.0, 18000.0]`
  - `/metadata/transmitters`: Group containing transmitter models (e.g. 18 to 96 transmitters per scenario) with `frequency_config`, `position_config`, `power_config`, `pri_config`, `pulse_width_config`, `scan_config`.
- **Physical Feature Units:**
  - `ToA`: Microseconds ($\mu\text{s}$)
  - `Frequency`: Megahertz ($\text{MHz}$)
  - `PulseWidth`: Microseconds ($\mu\text{s}$)
  - `AoA`: Degrees ($-180^\circ$ to $+180^\circ$)
  - `Amplitude`: Received power in $\text{dBm}$ (e.g. $-170$ to $+10$ dBm)

### B. Our Interpretation for CASS-EW
- TSRD is designed for pulse deinterleaving (clustering pulses by transmitter ID given an observed sequence).
- CASS-EW has an orthogonal objective: **cognitive receiver resource allocation** (choosing band, dwell, and receiver over time to maximize interception of unknown/dynamic emitters).
- Emitter labels in `/labels` represent ground truth identity and must be strictly quarantined from the scheduler; they are reserved exclusively for post-decision evaluator analysis.
- To avoid unit confusion, all internal time representations within the CASS-EW virtual receiver and scheduler must be normalized to **seconds**, while the adapter ingests microseconds.
- Band mapping must map physical frequency boundaries directly (e.g., Band 0: $[0, 1800\text{ MHz})$, Band 1: $[1800, 3600\text{ MHz})$, etc.) and never use modulo indexing (`freq % num_bands`).

### C. Not Available / Unknown at Runtime
- Live unauthenticated HTTP download of actual TSRD `.h5` files without credentials (blocked by HuggingFace gating).
- Counterfactual RF emission in SCAN mode: If a transmitter fired in Band 7 while the TSRD historical receiver was dwelling on Band 1, that pulse is completely omitted from the SCAN HDF5 `/data` array.

---

## 3. Dataset Suitability Classification

| Modality | Classification | Recommended Role in CASS-EW |
|:---|:---|:---|
| **TSRD STARE Mode** | `CAUSAL_ENVIRONMENT_READY` (with pulse-window indexing) | Ground-truth RF emitter environment for counterfactual virtual scanning |
| **TSRD SCAN Mode** | `HISTORICAL_OBSERVATION_ONLY` | Benchmark reference comparing fixed legacy scanning vs CASS-EW adaptive scanning |
| **Transmitter Configs** | `REQUIRES_RECONSTRUCTION` | Parametric synthetic pulse generator replicating exact transmitter dynamics |

---

## 4. Next Implementation Steps
1. Create `data/tsrd_adapter.py` implementing the normalized data contract, strict schema validation, and summary reporting.
2. Build `tests/test_tsrd_adapter.py` validating HDF5, JSON, and CSV ingestion with zero leakage.
3. Build physical `BandMap` and time normalization modules.
4. Build `simulator/dataset_rf_environment.py` and dataset-backed `VirtualReceiver` supporting counterfactual observation.
5. Create causality and leakage verification suite.
