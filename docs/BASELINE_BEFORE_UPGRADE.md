# CASS-EW Baseline Architecture & Metrics (Before Master Upgrade)

**Date**: October 2026  
**Scope**: Pre-Upgrade Verification of SIH Problem Statement 26055

---

## 1. Test Suite Verification
- **Total Tests**: 201
- **Passing**: 201
- **Failures / Errors**: 0
- **Execution Time**: ~1.11s (`python -m unittest discover tests`)
- **Status**: 100% Operational & Passing

---

## 2. Real Executable Architecture & Official Schedulers

### Official Production Pipeline
- **Command-line Interface**: `main.py`
  - `--mode synthetic`: Uses `SpatialScheduler` (multi-receiver, spatial + pattern + predictive + temporal + belief fusion).
  - `--mode replay`: Uses `PatternAwareScheduler` + `PDWReplayEngine`.
- **Benchmark Suite**: `evaluation/final_cass_ew_benchmark.py`
  - Evaluates single-receiver baselines (Random, Sequential, Phase 3–7) and multi-receiver systems (Phase 8 Spatial and Integrated Phase 10).
- **Web Application / Live Dashboard**: `web_app.py`
  - Default scheduler: `SpatialScheduler` with 3 receivers (`R1`, `R2`, `R3`) across 10 bands.

### Q-Learning / RL Mismatch Resolution
- In earlier experimental scripts (`evaluation/compare_algorithms.py`, `algorithms/rl_scheduler.py`, `training/train_rl.py`), a tabular Q-learning prototype (`RLScheduler`) was implemented with 3-bin belief discretization.
- **Truth**: The primary production system of CASS-EW does **NOT** use black-box Q-learning or neural RL. It uses an **interpretable, causal, multi-factor evidence-fusion cognitive scheduler** (`SpatialScheduler`).
- **Resolution**: `RLScheduler` is documented and retained strictly as an early experimental exploration/baseline. The official primary scheduler is the multi-evidence adaptive cognitive scheduler (`SpatialScheduler`).

---

## 3. Benchmark Metrics (Seeds 42, 43, 44, 45, 46)

From `evaluation/final_cass_ew_benchmark.py` (executed as module):

### Single-Receiver Baselines
| Algorithm | Interception Rate | Miss Rate | False Alarm Rate | Efficiency | Mean Intercept Time | Median Intercept Time | P90 Intercept Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random** | 6.09% | 93.91% | 10.75% | 0.0053 | N/A | N/A | N/A |
| **Sequential** | 14.63% | 85.37% | 9.72% | 0.0112 | N/A | N/A | N/A |
| **Phase 3 (AdaptiveBelief)** | 16.32% | 83.68% | 7.08% | 0.0093 | N/A | N/A | N/A |
| **Phase 4 (TemporalBelief)** | 16.44% | 83.56% | 7.40% | 0.0099 | N/A | N/A | N/A |
| **Phase 5 (MultiBandTemporal)** | 15.89% | 84.11% | 7.34% | 0.0103 | N/A | N/A | N/A |
| **Phase 6 (Predictive)** | 16.32% | 83.68% | 7.35% | 0.0104 | N/A | N/A | N/A |
| **Phase 7 (PatternAware)** | 15.38% | 84.62% | 7.35% | 0.0098 | N/A | N/A | N/A |

### Multi-Receiver Algorithms
| Algorithm | Interception Rate | Miss Rate | False Alarm Rate | Efficiency | Mean Intercept Time | Median Intercept Time | P90 Intercept Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Phase 8 (Spatial, w=0.3)** | 34.97% | 65.03% | 11.48% | 0.0616 | 1.00 | 1.00 | 1.00 |
| **Integrated Phase 10** | 34.97% | 65.03% | 11.48% | 0.0616 | 1.00 | 1.00 | 1.00 |

*Note on Phase 8 vs Phase 10 Benchmark Labels*: In the baseline benchmark, "Phase 8" and "Integrated Phase 10" instantiated the identical scheduler (`SpatialScheduler` with identical weights). This label redundancy must be cleaned up to clearly distinguish the spatial algorithmic module from the full system integration / multi-receiver freeze.

---

## 4. Identified Areas for Upgrade in Part 1
1. **Benchmark Labelling**: Eliminate identical duplicate definitions between Phase 8 and Phase 10; clearly separate single-receiver progression, multi-receiver spatial scheduling, and full system integration.
2. **Intercept-Time Evaluation System**:
   - Compute P90, worst-case intercept time, time-to-first-intercept, coverage, and worst-case staleness.
   - Separate episode latency from overall mission performance.
3. **RF Knowledge Map**:
   - Provide an explicit data structure aggregating per (receiver, band): belief, uncertainty, last observation time, last detection time, staleness, temporal score, prediction score, prediction confidence, pattern score, spatial evidence, activity state, and environment change indicator.
4. **Uncertainty Representation**:
   - Provide principled uncertainty estimation incorporating observation count, time elapsed without observation, and predictive variability.
5. **Anti-Starvation / Staleness Mechanism**:
   - Explicit starvation penalty / revisit priority to ensure stale bands receive scheduled observation without destroying exploitation of active targets.
6. **Decision Explainability**:
   - Provide structured explanations showing the exact mathematical score contribution for each candidate action.
