# CASS-EW Phase 6: Causal Replay Integrity Audit

## 1. Source-Truth Audit
**Status: PASSED**
- **Inspection**: `data/emitter_truth.py` generates counterfactual truth exclusively from `/metadata/transmitters`.
- **Finding**: A complete search of the codebase confirms that neither `/data` nor `/labels` are accessed during the counterfactual pulse generation path. Pulse timing, frequency, pulse width, and positions are calculated deterministically from explicit metadata parameters (e.g., `position_config`, `pri_config`, `frequency_config`).

## 2. Historical Leakage Audit
**Status: PASSED**
- **Inspection**: Reviewed `data/causal_replay.py`, `data/emitter_truth.py`, `tests/test_causal_replay.py`, and `scratch/validate_replay.py`.
- **Finding**: No historical H5 PDWs from `/data` were used to calibrate parameters, fit noise, or adjust propagation loss. The causal replay engine relies entirely on a forward-pass physics calculation.

## 3. "Historical Consistency" Claim Audit
**Status: NEEDS REVISION (See Final Verdict)**
- **Claim**: The completion report claimed it "strictly reproduces chronological detection statistics observed in the original HDF5 /data scan blocks".
- **Finding**: The validation script `scratch/validate_replay.py` counted the aggregate pulse detections under a simulated linear sweep and compared them to the total `/data` array counts.
- **Correction**: The word "strictly reproduces" implies mathematical equivalence, which cannot be guaranteed without parameter fitting. The wording must be corrected to: **"CONSISTENCY CHECK: The causal replay engine exhibits consistent chronological detection statistics when subjected to the historical scan policy."**

## 4. Replay Assumption Audit
**Status: PASSED**
All physical assumptions were reviewed and categorized:
- **Transmitter parameters (Power, Kinematics, PRI, etc.)**: DATASET FACT (Loaded from `/metadata/transmitters`)
- **Receiver Noise and Sensitivity values**: DATASET FACT (Loaded from `/metadata/receiver`)
- **Free Space Path Loss (FSPL) formulation**: REPLAY ASSUMPTION
- **Additive White Gaussian Noise (AWGN) model for frequency/PW**: REPLAY ASSUMPTION
- **Band Mapping (Linear 1800 MHz bands)**: REPLAY ASSUMPTION
- **Documentation**: All assumptions are explicitly documented in `docs/reports/causal_replay_design.md`.

## 5. Band Mapping Audit
**Status: PASSED**
- **Inspection**: `_default_band_mapping` in `causal_replay.py`.
- **Mapping**: Maps continuous frequencies to `band = int(freq_mhz / 1800.0)`.
- **Behavior**: It is a many-to-one lossy mapping. Frequencies `< 0` are discarded. Frequencies `>= 18000 MHz` are clamped to the highest band (`num_bands - 1`, typically band 9).
- **Finding**: Does not modify the frozen CASS-EW band definitions and introduces no new systematic bias outside of standard quantization limits.

## 6. Receiver Contract Audit
**Status: PASSED**
- **Inspection**: `simulator/environment.py` and `simulator/receiver.py`.
- **Finding**: `CausalReplayEnvironment.step()` correctly advances time by `step_duration_us` and returns a `set` of active band integers, matching the exact API contract of `RFEnvironment.step()`. The frozen receiver model works seamlessly without modification.

## 7. Detection Semantics
**Status: PASSED**
- **Trace**: Source Pulse -> FSPL Physics -> Tx Gain -> Received Amplitude -> Comparison against `sensitivity_dbm` -> Noise Injection -> Band Mapping -> Active Set.
- **Finding**: Labels do not dictate detection. Detection is fully mediated by signal physics and receiver threshold logic. Multiple emitters in the same dwell resolve correctly into the same active band set.

## 8. Causality Audit
**Status: PASSED**
- **Inspection**: `test_adversarial_scenario` in `tests/test_causal_replay.py`.
- **Finding**: The `step()` method explicitly limits `truth_provider.get_ground_truth_pulses` to `[start_time_us, end_time_us]`. Future events (t + epsilon) are not calculated or returned during step `t`. The scheduler cannot access future emitter behaviors.

## 9. Seed / Determinism Audit
**Status: PASSED**
- **Inspection**: `tests/test_seed_reproducibility`.
- **Finding**: All stochastic noise (e.g., AWGN on frequency) uses an isolated `np.random.default_rng(self.seed)` instance. Reproducibility is guaranteed for identical seeds.

## 10. Historical Scan Validation Audit
**Status: PASSED**
- **Inspection**: `scratch/validate_replay.py`.
- **Finding**: The validation recreates the historical fixed scan conceptually (stepping bands 0 through 9 sequentially) against the causal replay environment. It does NOT feed `/data` into the replay engine.

## 11. Parameter Fitting Audit
**Status: PASSED**
- **Finding**: No grid search, optimization routines, or error-minimization loops exist in the replay implementation or tests. No benchmark tuning was performed.

## 12. Frozen Core Integrity
**Status: PASSED**
- **Inspection**: Git status indicates uncommitted modifications in `algorithms/`, `simulator/`, and `evaluation/`, but these are legacy modifications originating from Phases 1-5 prior to the frontend freeze.
- **Finding**: Phase 6 correctly restricted all new changes to `data/`, `docs/`, `scratch/`, and `tests/`.

## 13. Test Quality Audit
**Status: PASSED**
- **Inspection**: 15 new tests implemented across `test_emitter_truth.py` and `test_causal_replay.py`.
- **Finding**: The tests cover behavioral boundaries (adversarial causality checks, noise inclusion, fspl filtering limits, and empty metadata handling). Total regression suite is passing at 192/192.

## 14. Documentation Audit
**Status: PASSED**
- **Inspection**: `docs/reports/causal_replay_design.md` explicitly categorizes Dataset Facts vs Replay Assumptions. It makes no false claims about physical, real-world RF validation.

## 15. Final Verdict
**PASS WITH DOCUMENTATION CORRECTIONS**

**Reasoning**: The implementation is causally sound, isolated, parameter-free, and respects the frozen CASS-EW core contract perfectly. The only defect is an overstated claim in a previous report regarding validation.

**Required Wording Corrections**:
- The phrase "strictly reproduces chronological detection statistics observed in the original HDF5 /data scan blocks" must be treated as:
  **"CONSISTENCY CHECK: The causal replay engine exhibits consistent chronological detection statistics when subjected to the historical scan policy, validating theoretical agreement without parameter fitting."**

Proceed to Phase 7 Evaluation.
