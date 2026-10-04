# HDF5 Semantic Audit Report

## 1. Dataset Provenance
**FACT**: The dataset is entirely **SYNTHETIC** generated from an emitter simulation.
**Evidence**: 
- Receiver attributes explicitly define noise scaling parameters (`toa_noise_scale_us`, `freq_noise_scale_mhz`, `pw_noise_scale_us`), meaning noise was added to clean generated values.
- Transmitter metadata contains explicitly configured deterministic parameters (`freqs_mhz`, `pris_us`, `pws_us`).
- Abstract physical parameters (`speed_km_s`, `start_position_km`) are provided in mathematically perfect coordinate spaces.

## 2. Feature Semantics Table

| Field Name | Observed Range | Data Type | Continuous/Discrete | Explicit Unit in Metadata | Strong Evidence | Confidence | Conclusion |
|------------|---------------|-----------|---------------------|---------------------------|----------------|------------|------------|
| ToA | ~4k to 29.19M | float32 | Continuous | `toa_noise_scale_us` | Inter-pulse diffs align with `pris_us` | HIGH | microseconds (us) |
| Frequency | 4.6 to 11987 | float32 | Continuous | `freq_noise_scale_mhz` | Matches `freq_range_mhz` | HIGH | MHz |
| PulseWidth | 0.006 to 367 | float32 | Continuous | `pw_noise_scale_us` | Matches `pws_us` in Tx | HIGH | microseconds (us) |
| AoA | -179.99 to 179.98 | float32 | Continuous | `travel_angle_deg` | Value bounded by ±180 | HIGH | degrees |
| Amplitude | -170.6 to +10.8 | float32 | Continuous | `sensitivity_dbm` | Matches typical RF power | HIGH | dBm |

## 3. ToA Findings
- **Observed Range**: ~4447 to ~29,197,522.
- **Starting Value**: Varies per file (e.g., 200k, 180k, 4k). This indicates absolute time from the start of the simulation rather than relative interval time.
- **Final Value**: ~28M to 29M. Given `collection_time_s` = 30.0s, 29 million microseconds equals 29.0 seconds, exactly aligning with the collection time.
- **Inter-pulse Intervals**: Median diffs vary (20us to 61us). 
- **Conclusion**: ToA is measured in absolute **microseconds** from the simulation start. (CONFIDENCE: HIGH).

## 4. Frequency Findings
- **Observed Range**: ~4.6 to 11,987.
- **Receiver Frequency Range**: Explicitly stated in metadata `freq_range_mhz`: `[500.0, 18000.0]`.
- **Dwell Centers**: Defined in `dwell_centres_mhz` (e.g., 250, 750, ..., 17750).
- **Transmitter Frequencies**: Defined as `freqs_mhz`.
- **Conclusion**: Frequency values represent absolute **MHz**. (CONFIDENCE: HIGH).

## 5. PulseWidth Findings
- **Observed Range**: ~0.006 to 367.
- **Transmitter Metadata**: Transmitters have arrays of `pws_us`.
- **Receiver Metadata**: Features `pw_res_us` and `pw_noise_scale_us`.
- **Conclusion**: Pulse width is explicitly defined in **microseconds (us)**. (CONFIDENCE: HIGH).

## 6. AoA Findings
- **Observed Range**: Bounded cleanly between ~ -180.0 and ~ +180.0.
- **Receiver Metadata**: Contains `travel_angle_deg`.
- **Conclusion**: Angle of Arrival is explicitly in **degrees**, likely representing azimuth from the receiver's broadside or a relative coordinate system. (CONFIDENCE: HIGH).

## 7. Amplitude Findings
- **Observed Range**: -170.6 to +10.8.
- **Receiver Metadata**: Specifies `sensitivity_dbm` (e.g., -110.0) and `gain_db` (10.0).
- **Conclusion**: Amplitude is explicitly measured in **dBm**. (CONFIDENCE: HIGH).

## 8. Receiver Metadata Findings
The `/metadata/receiver` group fully documents the simulated receiving platform:
- `scan_mode`: Explicitly states `"Scanning"`.
- `bandwith_mhz`: 500.0 MHz.
- `collection_time_s`: 30.0 s.
- `dwell_centres_mhz`: Array of exactly 36 discrete center frequencies representing the scan pattern.
- `dwell_times_s`: Array matching the length of `dwell_centres` specifying the duration per dwell (0.05s or 0.1s).
- `start_position_km`: Initial geometric position `[X, Y]`.

## 9. Transmitter Metadata Findings
The `/metadata/transmitters` group details each true emitter:
- **Keys**: `frequency_config`, `position_config`, `power_config`, `pri_config`, `pulse_width_config`, `scan_config`.
- **Schema**: Values are arrays allowing complex multimode behaviors (e.g., `pris_us: [50.0, 100.0, 150.0]` for a staggering emitter). 
- **Count**: 18 to 96 transmitters depending on the configuration file.

## 10. Label Semantics
- **Unique Labels**: Range from 17 to 72 per file.
- **Mapping**: Labels are integer IDs (e.g., 0, 1, 2) that directly correspond to the transmitter group indices in `/metadata/transmitters`.
- **Meaning**: A label explicitly identifies the *source physical emitter* of the corresponding PDW row.
- **Assumption Check**: A label does *not* necessarily mean a CASS-EW classification detection; it is merely the ground truth identity of the pulse.

## 11. Detection/Opportunity Semantics
**Can a CASS-EW detection event be directly derived from a PDW row?**
**ONLY WITH EXPLICIT EVALUATION ASSUMPTIONS.**

**Explanation:** 
The PDW records represent discrete pulses that successfully hit the receiver. Given the number of records (~169k over 30s with 96 emitters) versus physical PRI rates (e.g., PRI=100us -> 10,000 pulses/sec/emitter -> 9 million pulses total), the dataset contains *only intercepted pulses*, heavily filtered by the receiver's hardcoded scanning schedule (`dwell_centres_mhz`). 
The environment is NOT a continuous pulse stream; it is pre-filtered by the dataset's native scanner.

## 12. Spatial Semantics
- **Observed**: `start_position_km` exists for both receiver and transmitters. AoA exists for PDWs.
- **Inference**: AoA is the geometric angle between the receiver and transmitter positions. 
- **Warning**: Do NOT assume AoA provides the full transmitter position. It is only a 1D angle, and range/position would require triangulation across multiple receivers or movement, which isn't guaranteed natively.

## 13. Cross-File Consistency
Analyzed `config_0.h5` through `config_4.h5`:
- **Schema**: 100% consistent.
- **Units**: 100% consistent (all metadata features share the same keys).
- **Receiver Metadata**: Identical scanning strategies (`dwell_centres`, `freq_range`) across these 5 files.
- **Transmitter Counts**: Variable (from 18 up to 96).
- **Conclusion**: The dataset structure is highly stable.

## 14. CASS-EW Mapping Readiness

A. **Can Frequency safely be mapped into CASS-EW bands now?**
**YES**. We definitively know Frequency is in MHz and the valid range aligns with CASS-EW's likely operational bands.

B. **Can ToA safely be used as replay time now?**
**NEEDS ASSUMPTION**. ToA is in microseconds. CASS-EW's scheduler clock units must be checked (seconds vs. ms vs. us) to apply an exact conversion factor (e.g., `ToA * 1e-6`).

C. **Can PulseWidth be passed as an observation feature?**
**YES**. It is numerically sound and conceptually clear.

D. **Can AoA be passed as an observation feature?**
**YES**. It is a continuous feature bounded properly in degrees.

E. **Can Amplitude be passed as an observation feature?**
**YES**. It is valid dBm.

F. **Can labels be used as hidden ground truth?**
**YES**. They perfectly identify the underlying emitter.

G. **Can labels be treated as detections?**
**NO**. Labels are physical ground truth sources, whereas CASS-EW "detections" must be evaluated against the scheduler's active tracking/classification state. 

H. **Can receiver metadata be mapped to CASS-EW receiver actions?**
**NO**. The receiver metadata (`dwell_centres_mhz`, `dwell_times_s`) represents the *historical* scan pattern used to generate the dataset. Mapping this to CASS-EW actions conflicts directly with CASS-EW's autonomous scheduler.

I. **Can a causal replay environment be built without modifying the frozen scheduler?**
**NEEDS ASSUMPTION**. Because the PDWs are *pre-intercepted* according to the dataset's historical scan pattern, allowing CASS-EW to "schedule" dynamically will result in CASS-EW asking for a band at time $T$, but the dataset only has PDWs if the *historical* receiver was also looking at that band at time $T$. We must clarify whether the replay layer yields PDWs based on CASS-EW's frequency choice AND historical frequency choice, or if we ignore the dataset's historical receiver and assume the PDWs are available regardless.

## 15. Summary of Certainty
- **FACT**: Feature units are (us, MHz, us, deg, dBm).
- **FACT**: Dataset is synthetic.
- **FACT**: ToA is chronological microseconds from $T=0$.
- **INFERENCE**: The dataset is pre-filtered by the receiver's historical scan schedule (due to low record counts relative to PRIs).
- **UNKNOWN**: How CASS-EW's dynamic scheduler is expected to causally interact with pre-filtered historical intercepts.

## 16. Test Results
- Ran `python -m unittest discover tests`.
- **177/177 tests passed**.
- The CASS-EW core remains fully intact.

## 17. Files Created
- `tools/h5_semantic_audit.py`
- `docs/reports/h5_semantic_audit.md`

## 18. Remaining Unknowns
The fundamental causal paradox: How does CASS-EW dynamically schedule its receiver if the HDF5 PDWs have already been culled by the historical simulation's receiver scan pattern?
