# Phase 5 Implementation Report: Dataset -> CASS-EW Runtime Bridge

**Project:** CASS-EW — Cognitive Adaptive Smart Scan for Electronic Warfare (SIH 26055)  
**Phase:** Phase 5 — Dataset Runtime Bridge  
**Date:** October 3, 2026  
**Status:** COMPLETE (324/324 tests passing)

---

## 1. Objective
To build a clean, tested, reproducible runtime bridge connecting validated user-supplied RF/PDW datasets (CSV, JSON, HDF5) to the frozen CASS-EW production scheduler (`SpatialScheduler`) without modifying the frozen simulation/scheduler core. 

The end-to-end execution pipeline implemented:
```
USER DATASET (CSV / JSON / HDF5)
    ↓
FORMAT DETECTION (BaseDatasetAdapter.detect_format)
    ↓
FIELD MAPPING (Explicit / Inferred field dictionary)
    ↓
EXPLICIT UNIT NORMALIZATION (Canonical Units: s, MHz, µs, deg, dB)
    ↓
VALIDATION & BOUND CHECKING
    ↓
DatasetRFEnvironment (Causal windowed pulse emission)
    ↓
DatasetVirtualReceiver (Physical frequency filter & detection model)
    ↓
CASS-EW OBSERVATION CONTRACT (TSRDSchedulerBridge)
    ↓
EXISTING PRODUCTION SpatialScheduler (Frozen core intelligence)
    ↓
CASS-EW BAND DECISION (Next scan band target)
    ↓
DATASET-RUN METRICS (Strict separation of observation vs evaluation)
    ↓
REPRODUCIBLE RESULT REPORT (JSON artifact + CLI summary)
```

---

## 2. Architecture
The integration is structured in the dedicated dataset runtime layer:
- `data/dataset_runtime.py`:
  - `DatasetRunConfig`: Dataclass specifying dataset path, format, field mapping, explicit unit specs, dwell time, scan steps, seed, initial band, receiver configuration, band mapping bounds, and output paths.
  - `DatasetRunResult`: Dataclass capturing dataset metadata, run configuration, scan decision history, audited evaluation metrics, and operational warnings. Includes `.to_dict()` and `.save_json()`.
  - `DatasetRuntimeRunner`: Orchestrator class that ingests data via Phase 4 adapters, constructs `DatasetRFEnvironment` and `DatasetVirtualReceiver`, adapts to `TSRDSchedulerBridge`, drives the step loop causally, and computes post-run metrics with ground truth kept strictly isolated.
  - `run_dataset()`: Top-level functional entry point.
- `evaluation/dataset_runner.py`:
  - CLI entry point supporting `--dataset`, `--config`, `--steps`, `--dwell`, `--seed`, `--initial-band`, `--bands`, `--freq-min`, `--freq-max`, `--output`. Formats and displays run results cleanly.

---

## 3. Files Created
1. `data/dataset_runtime.py` - Core runtime bridge and execution orchestrator.
2. `evaluation/dataset_runner.py` - CLI tool for running arbitrary datasets through CASS-EW.
3. `tests/test_phase5_dataset_runtime.py` - Dedicated 19-test test suite for Phase 5.
4. `docs/reports/phase5_dataset_runtime_report.md` - This comprehensive architecture and audit report.

---

## 4. Files Modified
- `tests/user_fixtures.py` - Extended `generate_generic_hdf5` timestamp upper bound from 50,000 µs (0.05s) to 1,000,000 µs (1.0s) so multi-step dwells do not run out of synthetic pulses.

---

## 5. Runtime Flow
1. **Ingestion & Validation**: `load_dataset(path, config)` parses the raw file, applies explicit unit conversions to canonical units (`timestamp_s`, `frequency_mhz`, `pulse_width_us`, `aoa_deg`, `amplitude_db`), normalizes/sorts timestamps monotonically, and drops out-of-range physical values.
2. **Environment & Receiver Setup**:
   - `DatasetRFEnvironment` stores chronological `GroundTruthPDW` records and serves pulses strictly bounded within the current scan dwell `[t, t + dwell_s]`.
   - `DatasetVirtualReceiver` tunes to the current RF band using physical `BandMap` (no modulo arithmetic) and applies detection sensitivity / noise floor physics.
3. **Observation Adaptation**:
   - Pulses detected in the receiver are converted into `ObservablePDW` records (stripped of emitter IDs and scenario ground truth).
   - `TSRDSchedulerBridge` converts receiver outputs into the exact dictionary format expected by the frozen `SpatialScheduler.update()`.
4. **Cognitive Decision**:
   - `SpatialScheduler` processes the observation, updates its belief state, spatial map, and temporal trackers, and returns `next_band`.
   - The runner commands receiver band retuning (`receiver.tune_to_band(next_band)`), advances simulation time by `dwell_time_s + switching_time_s`, and records the scan step.
5. **Post-Run Audited Metric Computation**:
   - Computes receiver observation opportunities, hits, misses, false alarms, receiver Pd, Pfa, scan efficiency, emitter interception rate, and onset latency.

---

## 6. Dataset-to-Scheduler Boundary
The boundary between the dataset and the scheduler is strictly enforced:
- **Scheduler Visibility**:
  - Current tuned band index
  - Detection flag (`hit_flag`)
  - Detected pulse parameters: frequency (MHz), pulse width (µs), AoA (deg), amplitude (dB), timestamp (s)
  - Receiver ID
- **Prohibited & Inaccessible**:
  - Emitter labels / class names
  - Transmitter IDs
  - Future pulses in the dataset
  - Quiet scan ground truth vs undetected signal ground truth
  - Hidden scenario metadata

---

## 7. Ground-Truth Separation
Ground truth emitter labels and unobserved pulses are stored solely in the evaluator's memory. When ground truth is absent from a user dataset (e.g. blind interception recording):
- `has_ground_truth` is set to `False`.
- Emitter interception metrics and onset latencies return `None` (`null` in JSON).
- The system emits an explicit warning: `"Dataset does not contain ground truth emitter labels. Global emitter interception metrics are unavailable."`
- No fabricated zero or guessed metric is ever substituted.

---

## 8. Causal / Leakage Safeguards
To eliminate temporal and spatial information leakage:
1. **Windowed Access**: At step $t$, the receiver queries only pulses with $T_{\text{start}} \le \text{timestamp} < T_{\text{end}}$. Future pulses cannot be inspected.
2. **Adversarial Future Divergence Test**: We implemented an explicit anti-leakage regression test (`test_t_adversarial_future_pulse_divergence`). Two datasets with identical pulses up to $T = 0.05$s but wildly divergent future pulses after $T$ were executed through the bridge. The scheduler's decisions up to $T$ were verified to be bit-identical.

---

## 9. Band Mapping Behavior
- Frequencies are mapped using `BandMap` with contiguous physical boundaries.
- **Strictly No Modulo Mapping**: Modulo arithmetic (`freq % num_bands`) is prohibited.
- **Out-of-Coverage Policy**: Pulses outside `[min_freq, max_freq]` are never wrapped or silently clamped; the runner detects out-of-coverage pulses and logs explicit warnings.

---

## 10. Metric Definitions
All metrics strictly preserve the audited definitions:
- **Receiver Observation Opportunities**: Scans where signal energy was physically present in the tuned band.
- **Hit**: Signal physically present in tuned band AND receiver detector triggers.
- **Miss**: Signal physically present in tuned band AND receiver detector fails to trigger.
- **False Alarm**: No signal physically present in tuned band AND receiver detector triggers.
- **Receiver Pd**: $\text{Hits} / \text{Opportunities}$ (Returns `None` if Opportunities = 0).
- **Receiver Pfa**: $\text{False Alarms} / \text{Quiet Scans}$ (Returns `None` if Quiet Scans = 0).
- **Scan Efficiency**: $\text{Hits} / \text{Total Scans}$.
- **Global Emitter Interception Rate**: $\text{Unique Intercepted Emitters} / \text{Total Scenario Emitters}$ (Requires Ground Truth).
- **Intercept Latency**: $\text{First Detection Time} - \text{Ground-Truth Emitter Onset}$.

---

## 11. CLI Usage
Run any dataset via the module command:
```bash
python -m evaluation.dataset_runner \
    --dataset path/to/dataset.csv \
    --config path/to/dataset_config.json \
    --steps 100 \
    --dwell 0.005 \
    --seed 42 \
    --output results/run_summary.json
```

---

## 12. Example Configuration JSON
```json
{
  "field_mapping": {
    "timestamp_col": "timestamp_s",
    "freq_col": "frequency_mhz",
    "pw_col": "pulse_width_us",
    "aoa_col": "aoa_deg",
    "amp_col": "amplitude_db",
    "emitter_col": "emitter_label"
  },
  "unit_config": {
    "timestamp_unit": "s",
    "frequency_unit": "MHz",
    "pulse_width_unit": "us",
    "aoa_unit": "deg",
    "amplitude_unit": "dBm"
  }
}
```

---

## 13. Test Results
- **Previous Test Suite Count:** 305 tests passing.
- **New Phase 5 Tests:** 19 dedicated tests covering:
  - CSV runtime execution
  - JSON runtime execution
  - HDF5 runtime execution
  - Explicit field mapping
  - Explicit unit normalization
  - Scheduler receives no emitter labels
  - Scheduler cannot access future pulses
  - Physical band mapping
  - Out-of-coverage pulse handling
  - Deterministic same-seed execution
  - Stochastic seed sensitivity
  - Empty dataset handling
  - No-ground-truth dataset handling
  - Zero-denominator metric safety
  - Result JSON serialization
  - CLI execution via subprocess
  - Invalid dataset rejection
  - Frozen core file integrity
  - Adversarial future pulse divergence (Anti-leakage)
- **Total Test Suite:** 324 tests passing (`python -m unittest discover tests -v`), 0 failures, 0 errors.
- **TSRD Smoke Benchmark:** PASS (`python -m evaluation.tsrd_benchmark --steps 20`).

---

## 14. Frozen-Core Verification
The following core files and directories were verified to have suffered **0 modifications** in Phase 5:
- `algorithms/` (frozen production `SpatialScheduler` & baselines intact)
- `simulator/environment.py` (synthetic simulator core untouched)
- `simulator/receiver.py` (synthetic receiver core untouched)
- `config.py` (global constants untouched)
- `main.py` (main synthetic execution untouched)

---

## 15. Known Limitations
1. **Sensor Agnostic**: The runtime bridge assumes incoming pulses can be represented as canonical PDWs (ToA, FoA, PW, AoA, Amplitude). Non-PDW I/Q raw sample files must be pre-channelized into PDWs before passing to this bridge.
2. **Replay Mode**: The environment replays pulse arrival times deterministically as recorded; it does not simulate dynamic Doppler frequency drift or moving emitter trajectories unless reflected in the dataset timestamps and AoA values.

---

## 16. What Phase 5 Does NOT Claim
- **No Operational Real-World EW Claims**: Replaying recorded or synthetic datasets through CASS-EW validates the data pipeline and cognitive scanning behavior on structured PDW streams; it does not constitute full hardware-in-the-loop or real-world battlefield operational certification.
- **No Algorithm Superiority Inferences**: Benchmark metrics produced on an individual dataset reflect the dataset's specific density, emitter distribution, and frequency layout.
