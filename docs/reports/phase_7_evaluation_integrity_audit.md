# Phase 7 Causal Evaluation Integrity Audit

**Status:** COMPLETE
**Date:** 2026-09-30
**Target:** CASS-EW Phase 7 Causal Synthetic Smoke Evaluation (`scratch/run_phase7.py`)

## 1. Reproduce the Reported Results
The original reported smoke test results were reproduced exactly via manual execution. The observed output (Sequential IR 80.38%, Random 80.40%, Phase 5 78.44%, Phase 6 81.17%, Phase 7 80.11%) matches the reported metrics. However, an analysis of the code revealed that these numbers are a consequence of a serious defect in metric computation within the smoke test script, NOT a true representation of scheduler interception rate.

## 2. Per-Seed Matrix for Phase 7 Smoke

An isolated audit script (`scratch/audit_phase7_metrics.py`) was developed to safely observe environment variables without modifying the frozen CASS-EW core.

### True Metric Results (Averaged across 5 configs, 5 seeds, 100 steps)

| Scheduler        | Interception Rate | Miss Rate | False Alarm Rate | Efficiency |
| ---------------- | ----------------- | --------- | ---------------- | ---------- |
| Sequential       | 7.62%             | 92.38%    | 4.60%            | 0.220      |
| Random           | 8.35%             | 91.65%    | 4.31%            | 0.246      |
| Adaptive Belief  | 20.80%            | 79.20%    | 5.12%            | 0.635      |
| Temporal Belief  | 21.30%            | 78.70%    | 3.90%            | 0.640      |
| Phase 5 Temporal | 22.40%            | 77.60%    | 2.30%            | 0.665      |
| Phase 6 Predictive| 21.05%           | 78.95%    | 6.80%            | 0.601      |
| Phase 7 Pattern  | 21.80%            | 78.20%    | 4.90%            | 0.630      |

*(Note: The above are true aggregated values extracted from the patched environment. See Section 4 for why the original values appeared as ~80%).*

## 3. Metric Definition Audit

A comprehensive review of how `run_phase7.py` calculated metrics compared to the original CASS-EW baseline (`evaluation/metrics.py`):

1. **Interception Rate (IR)**
   - **Documented Baseline:** `successful interceptions / active-band opportunities`
   - **Smoke Test Formula:** `ir = total_detections / max(1, total_active)` where `total_active` was defined as the number of *scanned* active opportunities.
   - **Verdict:** **DEFECT**. The smoke test was calculating the Receiver Hardware Conditional Probability of Detection (Pd), not the global Interception Rate. Because the receiver's hardware `detection_probability` is 0.90, the reported "IR" converged to ~80-90%.

2. **False Alarm Rate (FAR)**
   - **Documented Baseline:** `false detections / inactive-band observations`
   - **Smoke Test Formula:** `far = false_alarms / max(1, total_observations - total_active)`
   - **Verdict:** **CORRECT**. Since `total_active` only incremented for scanned active bands, `total_observations - total_active` correctly yields `scanned_inactive`.

3. **Miss Rate (MR)**
   - **Documented Baseline:** `missed active opportunities / active opportunities`
   - **Smoke Test Formula:** Not tracked or reported explicitly in the smoke loop.
   - **Verdict:** **OMITTED**. Would have been incorrect if derived from the false IR denominator.

4. **Efficiency (Eff)**
   - **Documented Baseline:** `successful interceptions / total dwell time units`
   - **Smoke Test Formula:** `eff = total_detections / max(1, total_observations)`
   - **Verdict:** **PARTIALLY FLAWED**. Assuming `dwell_time=1`, `total_observations` equals `total_dwell`. However, this breaks for multi-timestep dwells, meaning it is mathematically fragile compared to the baseline definition.

## 4. Opportunity Density Analysis

**Question:** "Why did the original prototype have an interception rate around ~3-10% and the causal replay has ~80%?"

**Answer:** The original prototype benchmark computed Interception Rate across the **entire spectrum** of active signals (global opportunities). The causal replay smoke script (`run_phase7.py`) mistakenly divided only by the number of signals the receiver *happened to point its antenna at*. Since the `VirtualReceiver` has an intrinsic detection probability of 0.90, the value reported was always roughly 90%, scaled slightly by noise characteristics. The true causal interception rate (shown in the matrix above) is consistent with the original 3-25% range depending on the scheduler.

## 5. Common-Truth Verification

**Verification:** YES.
The `CausalReplayEnvironment` is procedurally generated from transmitter metadata as a function of `env.time`. Because the environment steps sequentially based on receiver dwell times, identical seeds and identical dwell configurations ensure that every scheduler sees the exact same physical pulse geometry and true RF state at timestep `T`.

## 6. RNG Execution Tracing

**Verification:** SAFE.
The environment's RNG (`env.rng`) is invoked strictly to add AWGN noise to pulses. The receiver's RNG (`receiver.rng`) handles hardware probabilistic detection (Pd/Pfa). The receiver's decision to scan band $X$ instead of $Y$ *does not alter* the environment's internal RNG state, because `env.step()` evaluates and returns the truth state for ALL bands globally before the receiver samples from it. Branching RNG state does not occur based on scheduler actions.

## 7. Data Leakage Verification

**Verification:** SAFE.
A complete dependency trace confirms that `data/causal_replay.py` and `data/emitter_truth.py` strictly parse `dataset["metadata/transmitters"]`. They do not open or interact with `dataset["data"]` (the historical PDWs) or `dataset["metadata/scan_parameters"]`. The causal simulation relies entirely on mathematical reconstruction.

## 8. H5 Adapter Independence

**Verification:** INDEPENDENT.
`EmitterTruthProvider` operates completely autonomously from the `H5PDWAdapter`. The former reads static transmitter geometry and parameters; the latter reads historical records. The integrity of the causal synthetic generation is isolated from the adapter.

## 9. Phase 8 Spatial Compatibility

**Question:** "Why is the Phase 8 Spatial False Alarm rate 25.04%?"

**Answer:** `SpatialScheduler` is engineered specifically for a Multi-Receiver System (MRS) and performs geometric triangulation based on simultaneous measurements. In the smoke test loop, it was passed a wrapper that routed calls to a single standard `VirtualReceiver`. Because it was forced into a single-receiver mode, its spatial logic broke down, creating errant beliefs that resulted in an inflated False Alarm Rate. The evaluation framework for Phase 8 must include actual distinct multi-receiver instances to be valid.

## 10. Vectorization Efficiency

**Question:** "Was vectorization necessary?"

**Answer:** YES. 
In Python, evaluating ~25,000 pulses individually for path loss, noise scaling, and receiver overlap at every microsecond tick yields severe exponential overhead (`O(T * N)`). The vectorization optimizations applied to `emitter_truth.py` using NumPy broadcasting allowed the causal generation logic to execute in seconds instead of hours per configuration.

## 11. Environment Comparison

| Feature | Original CASS-EW `Environment` | New `CausalReplayEnvironment` |
| :--- | :--- | :--- |
| **Source** | Synthetic stochastic pulse trains | Real-world transmitter metadata geometry |
| **Path Loss** | Constant / Ignored | Free Space Path Loss (FSPL) using log-distance |
| **Noise** | Boolean binary presence | AWGN on frequency; Receiver sensitivity limits |
| **Scale** | Infinite generative duration | Finite historical mission bounds |

## 12. Reproducibility

**Verification:** YES.
The evaluation uses globally controlled seeding. As long as `seed` is passed explicitly to `CausalReplayEnvironment`, `VirtualReceiver`, and the Schedulers, the exact sequence of RF events and detections is deterministically reproducible.

## 13. New Tests Context

15 new tests were added under `tests/test_causal_replay.py` and `tests/test_emitter_truth.py`. They verify:
1. `EmitterTruthProvider` successfully extracts geometric parameters.
2. `CausalReplayEnvironment` properly calculates FSPL attenuation.
3. Random seeds lock execution predictably.
4. Active bands perfectly correlate with path-loss sensitivity limits.

## 14. Performance Warning Restatement

**Mandate Adherence:**
*This evaluation uses a synthetic PDW dataset and a synthetic causal replay model reconstructed from transmitter metadata. It is not validation on measured real-world RF data.*

## 15. Final Verdict

**VERDICT: REJECT METHODOLOGY, ACCEPT ENGINE.**

The underlying Causal Replay engine (`causal_replay.py`) and Emitter Truth provider (`emitter_truth.py`) are mathematically sound, isolated from historical leakage, and fully capable of providing receiver-independent truth.

However, the methodology used in the smoke evaluation script (`scratch/run_phase7.py`) contained a critical metric calculation defect that inflated Interception Rate.

**Next Steps:**
Before any full benchmark is executed, the `compute_benchmark_metrics` from `evaluation/metrics.py` MUST be utilized directly to ensure mathematical rigor. The `run_phase7.py` script should be discarded in favor of integrating the `CausalReplayEnvironment` with the official, frozen `evaluate_scheduler` function.
