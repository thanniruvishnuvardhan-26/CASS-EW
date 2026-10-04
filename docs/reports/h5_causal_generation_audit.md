# HDF5 Causal Generation Audit Report

## 1. Search for Dataset Generation Source
- **Result**: Checked `C:\Anti Gravity\Datasets` and `C:\Anti Gravity\CASS-EW\CASS-EW\dataset` recursively for `.py`, `.json`, `.yml`, and `.md` source generation files.
- **Findings**: The only relevant script is `dataset/analyze_dataset.py`, which is currently empty. No source code or documentation responsible for generating the datasets is present within the accessible filesystem.

## 2. Trace the Generation Chain
Based on the metadata schema and mathematical constraints, the generation chain is:
**TRANSMITTER PARAMETERS** (FACT - explicitly stored in `/metadata/transmitters`)
        ↓
**PULSE GENERATION** (INFERENCE - pulses must have been generated continuously per `pris_us`)
        ↓
**NOISE / DISTORTION** (FACT - explicitly configured via `toa_noise_scale_us`, `freq_noise_scale_mhz`, etc.)
        ↓
**RECEIVER** (FACT - geometric position `start_position_km` defined)
        ↓
**DWELL / SCAN** (FACT - explicit `dwell_centres_mhz` and `dwell_times_s` arrays define a historical scan pattern)
        ↓
**PDW DATASET** (FACT - resulting chronological `data` array)

## 3. Receiver Scan Evidence
The `/metadata/receiver` group strongly suggests a historical execution process rather than just configuration:
- `scan_mode`: "Scanning"
- `start_position_km`: The spatial origin of the receiver.
- `dwell_centres_mhz`: An exact sequence of 36 scan bands (250, 750, 1250, 1750, ... 17750).
- `dwell_times_s`: Exact continuous durations corresponding one-to-one with the centres (e.g., 0.1s, 0.1s, 0.05s).
- **Evidence of filtering**: This explicit sequence acts as a hard filter matrix over the continuous time domain.

## 4. Frequency Test
**Total PDWs**: 169,617  
**Test**: Do PDW frequencies strictly align with the nearest dwell center's bandwidth?
- Every PDW (`100.00%`) falls strictly within the `500.0 MHz` bandwidth of one of the 36 `dwell_centres_mhz`.
- **Max distance to nearest dwell center**: 250.00 MHz (exactly half the bandwidth).
- **Numerical Evidence**: The continuous environment was strictly band-pass filtered by the historical dwells.

## 5. Temporal Test
**Total Collection Time**: 30.0s  
**Test**: Do PDWs chronologically match the historical dwell timeline sequence?
- **Timeline Reconstruction**: 
  - Dwell 0 (250 MHz): `t=0.0s` to `0.1s`
  - Dwell 1 (750 MHz): `t=0.1s` to `0.2s`
  - Dwell 2 (1250 MHz): `t=0.2s` to `0.25s`
- **Actual PDW Data (First 20 records)**:
  - No PDWs from `t=0.0s` to `t=0.20s` (no active emitters in 250/750 MHz).
  - PDWs start exactly at `t=0.20014s` with frequency `1396.31 MHz`. This aligns perfectly with Dwell 2 (1250 MHz band).
  - At `t=0.30268s`, the frequency immediately jumps to `2701.24 MHz`, which perfectly aligns with Dwell 5 (2750 MHz band, active from `t=0.30s` to `0.35s`).
- **Numerical Evidence**: The PDW timestamps perfectly obey the reconstructed historical scan timeline.

## 6. Transmitter Pulse Count
**Test**: Do continuous transmitters show missing pulses?
- **Tx 0**: `PRI=2000.0us`. Expected over 30s: **15,000** pulses. Actual in dataset: **173**.
- **Tx 1**: `PRI=100.0us`. Expected over 30s: **300,000** pulses. Actual in dataset: **302**.
- **Tx 10**: `PRI=1500.0us`. Expected over 30s: **20,000** pulses. Actual in dataset: **132**.
- **Numerical Evidence**: >99% of generated pulses are missing. This is not arbitrary truncation; it is the mathematical result of the historical receiver only listening to specific frequencies for short fractional durations (e.g., 0.05s per scan cycle).

## 7. Label Validation
- Labels map directly to transmitter indices in `/metadata/transmitters`.
- `label = transmitter identity / ground-truth source` (FACT).
- `label ≠ detection` (FACT). A label only states which emitter created the pulse; it does not indicate whether a CASS-EW classification algorithm has recognized it.

## 8. Critical Counterfactual Question
**"If CASS-EW had selected a different frequency band at time t, can we determine from this dataset whether a transmitter would have produced an observable PDW during that alternative dwell?"**

**ONLY WITH ASSUMPTIONS**

**Explanation**: 
The stored dataset records are pre-filtered. If the historical dataset receiver was looking at 1250 MHz at `t=0.2s`, the dataset *only* contains PDWs for 1250 MHz at that moment. If CASS-EW's scheduler decides to look at 2750 MHz at `t=0.2s`, there are absolutely zero PDW records available in the dataset to show what was happening at 2750 MHz. The only way to answer the counterfactual is to *assume* the metadata (e.g., `pris_us`) is physically complete and use it to mathematically regenerate the missing pulses that the historical scanner ignored.

## 9. Final Classification

**C — PDWs GENERATED UNDER A FIXED HISTORICAL RECEIVER SCAN**

**FACTS**:
- 100% of PDW frequencies fall exactly within the historical receiver's active dwells.
- PDW timestamps perfectly track the historical scan timeline (e.g., jumping from 1250 MHz to 2750 MHz precisely at the historical scan boundaries).
- >99% of expected pulses are missing from the records due to this hard temporal/frequency filtering.

**INFERENCES**:
- The environment was generated continuously, but the `data` array represents *only the intercepted pulses* observed by the historical scanner.

**UNKNOWN**:
- How CASS-EW will be scored fairly if its scheduler deviates from the historical schedule, since the dataset lacks the records to reward correct alternative decisions.

**REPLAY IMPLICATION**:
- A naive direct replay of the `data` array into CASS-EW will break causality. The CASS-EW scheduler will issue commands, but the environment will only yield PDWs based on the *historical* receiver's commands. Causal replay will require synthesizing the unrecorded pulses from the transmitter metadata.
