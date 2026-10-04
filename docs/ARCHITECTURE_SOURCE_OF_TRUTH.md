# CASS-EW Architecture Source of Truth

**Project**: Cognitive Adaptive Smart Scan for Electronic Warfare (CASS-EW)  
**SIH Problem Statement**: 26055  
**Version**: Part 2 RF Intelligence & Receiver Realism Upgrade  

---

## 1. Actual Executable Pipeline

The real operational and executable pipeline across all modes (Synthetic, Replay, and Live Dashboard) is structured as follows:

```
                  ┌──────────────────────────────────────────────┐
                  │          Real or Synthetic RF Source         │
                  │   - Synthetic: SpatialRFEnvironment          │
                  │   - Replay: PDWReplayEngine / Mock Fixtures  │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │             Virtual Receiver(s)              │
                  │   - VirtualReceiver / MultiReceiverSystem    │
                  │   - Hardware constraints: switching delay,   │
                  │     dwell time, detection probability (Pd),  │
                  │     false alarm probability (Pfa)            │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │             Observation Contract             │
                  │   - tuned_band, observation_time, dwell,     │
                  │     detection_result, switching_time         │
                  │   - STRICT ISOLATION: No ground-truth state, │
                  │     no future data, no emitter truth leakage │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │               RF Knowledge Map               │
                  │   Per (Receiver, Band) state representation: │
                  │   - Belief state (Bayesian hit/miss update)  │
                  │   - Explicit Uncertainty metric              │
                  │   - Staleness / Starvation tracker           │
                  │   - Temporal rhythm / Interval history       │
                  │   - Next-detection Prediction & Confidence   │
                  │   - Inter-band Signal Pattern correlation    │
                  │   - Multi-receiver Spatial evidence          │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │            Core Cognitive Scheduler          │
                  │   - Balance: Exploitation vs Exploration     │
                  │   - Anti-Starvation / Maximum revisit bounds │
                  │   - Causal Evidence Fusion Priority Formula: │
                  │     P(r, b) = w_b * Belief +                 │
                  │               w_t * Temporal +               │
                  │               w_p * Prediction +             │
                  │               w_pat * Pattern +              │
                  │               w_s * Spatial +                │
                  │               w_u * Uncertainty +            │
                  │               w_st * Staleness               │
                  │   - Full Mathematical Decision Explanation   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                             (Receiver, Band, Dwell)
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │              Observation Feedback            │
                  │   - Detected / Missed / False Alarm          │
                  │   - Online update to RF Knowledge Map        │
                  └──────────────────────────────────────────────┘
```

---

## 2. Actual Modules and Classes Used in Production

| Layer | Module Path | Class / Function | Role in System |
|:---|:---|:---|:---|
| **RF Environment** | `simulator.environment` | `RFEnvironment`, `Emitter` | Physics simulation of multi-band emitters and spectrum state |
| **Spatial Environment** | `simulator.multi_receiver` | `SpatialRFEnvironment`, `ReceiverProxyEnvironment` | Attenuation modeling based on emitter-receiver physical geometry |
| **Receiver System** | `simulator.receiver`, `simulator.multi_receiver` | `VirtualReceiver`, `MultiReceiverSystem` | Simulates RF tuning, switching delays, receiver dwell, detection fidelity |
| **RF Knowledge Map** | `algorithms.rf_knowledge_map` | `RFKnowledgeMap`, `BandKnowledgeEntry` | Unified observable state: belief, uncertainty, staleness, rhythms, patterns |
| **Temporal Reasoning** | `algorithms.phase5_temporal_profile` | `BandTemporalProfile`, `MultiBandTemporalProfileScheduler` | Learns pulse intervals, median period, MAD rhythm stability, freshness |
| **Prediction** | `algorithms.phase6_predictive_scheduler` | `BandPredictiveProfile`, `PredictiveScheduler` | Evaluates next expected detection time, predictive score and confidence |
| **Pattern Correlation** | `algorithms.phase7_signal_pattern` | `PatternRelationship`, `PatternAwareScheduler` | Cross-band emitter correlation and harmonic/coincidence tracking |
| **Spatial Reasoning** | `algorithms.phase8_spatial_scheduler` | `ReceiverSpatialProfile`, `SpatialScheduler` | Multi-receiver signal strength and regional presence aggregation |
| **PRI / Interval Analysis** | `algorithms.temporal_interval_analyzer` | `TemporalIntervalAnalyzer` | Observation-driven inter-arrival interval statistics, periodicity estimation, and prediction quality evaluation |
| **Frequency Hop Prediction** | `algorithms.frequency_hop_predictor` | `FrequencyHopPredictor` | Laplace-smoothed Markov transition matrix for band-hopping prediction, with top-k accuracy evaluation |
| **RF Change Detection** | `algorithms.rf_change_detector` | `RFEnvironmentChangeDetector`, `EnvironmentChangeEvent` | Detects NEW_ACTIVITY, EMITTER_DISAPPEARANCE, PERIODICITY_CHANGE, HOPPING_CHANGE, SPECTRUM_DRIFT from online observations |
| **Anti-Starvation** | `algorithms.phase8_spatial_scheduler` | Built into Knowledge Map & Scheduler | Tracks dwell history, maximum revisit deadlines, and starvations |
| **Decision Explainability**| `algorithms.phase8_spatial_scheduler` | `ActionExplanation`, `SpatialScheduler.explain_last_decision` | Mathematical breakdown of exact score components per action |
| **Evaluation & Metrics** | `evaluation.intercept_time`, `evaluation.metrics` | `InterceptTimeTracker`, `compute_benchmark_metrics` | Intercept latency, P90, worst-case, miss rates, without scheduler leakage |
| **PDW Adapter** | `data.real_pdws`, `data.causal_replay` | `PDWReplayEngine`, `validate_dataset` | Replay external EW Pulse Descriptor Words through the identical pipeline |

---

## 3. Resolution of the Q-Learning / RL Mismatch

1. **Origin of RL in Repository**:
   During initial scoping, an elementary discrete tabular Q-learning model (`RLScheduler`) was explored (`algorithms/rl_scheduler.py`). It discretized a single receiver's 10-band belief state into 3 coarse bins per band ($3^{10} = 59,049$ states).
2. **Operational Reality**:
   - The state space explosion for multi-receiver, continuous-dwell, high-temporal-resolution EW spectrum scanning renders tabular Q-learning unscalable without massive sample inefficiency.
   - More crucially, black-box RL lacks the mission-critical auditable causality, explainability, and guaranteed anti-starvation required for defence electronic warfare.
3. **Official Role of Q-Learning**:
   - `RLScheduler` is classified as an **Experimental / Baseline** scheduler.
   - The official, primary production scheduler of CASS-EW is the **Interpretable Adaptive Evidence-Fusion Cognitive Scheduler** (`SpatialScheduler` backed by `RFKnowledgeMap`).
   - Terminology in all documentation and dashboards reflects this truth.

---

## 4. Benchmark Phase Taxonomy

To eliminate redundant and confusing phase tags, benchmarks are organized strictly as:
- **Baseline Non-Adaptive**: `Random`, `Sequential`
- **Single-Receiver Cognitive Progression**:
  - `Phase 3 (AdaptiveBelief)`: Direct detection-probability belief updating.
  - `Phase 4 (TemporalBelief)`: Global temporal history and periodicity scoring.
  - `Phase 5 (MultiBandTemporal)`: Per-band temporal profiling, rhythm reliability (MAD), and freshness.
  - `Phase 6 (Predictive)`: Explicit next-detection event prediction and confidence weighting.
  - `Phase 7 (PatternAware)`: Cross-band emitter association and coincidence reasoning.
- **Multi-Receiver Spatial & Anti-Starvation**:
  - `Phase 8 (Spatial)`: Distributed multi-receiver spatial correlation.
  - `Full Integrated Cognitive Scheduler`: Spatial + Pattern + Prediction + Temporal + Belief + Anti-Starvation + Uncertainty reasoning.

---

## 5. Part 2 RF Intelligence & Receiver Realism

### Observation-Driven RF Intelligence Modules

| Module | Capability | Evaluation Metric |
|:---|:---|:---|
| **TemporalIntervalAnalyzer** | PRI estimation: inter-arrival intervals, mean/median/variance/jitter, periodicity confidence, predicted next-event time | `evaluate_prediction_accuracy()`: MAE, hit rate within tolerance |
| **FrequencyHopPredictor** | Markov chain band-transition learning, top-k next-band prediction | `evaluate_prediction_quality()`: top-1/top-3 accuracy, prediction coverage |
| **RFEnvironmentChangeDetector** | Five event types from online scan observations | Event count, severity distribution, confidence per event type |

### RF Environment Change Events

| Event Type | Trigger Condition | Severity |
|:---|:---|:---|
| `NEW_ACTIVITY` | Consecutive detections on previously quiet band | HIGH (if baseline < 5%) / MEDIUM |
| `EMITTER_DISAPPEARANCE` | Extended miss streak on previously active band | MEDIUM |
| `PERIODICITY_CHANGE` | Predicted periodic detection missed repeatedly | LOW |
| `HOPPING_CHANGE` | Unexpected band transition (low Markov probability) | MEDIUM |
| `SPECTRUM_DRIFT` | Jensen-Shannon divergence between old/new activity distribution > 0.3 | HIGH (JS > 0.5) / MEDIUM |

### Receiver Realism

The `VirtualReceiver` models:
- **SNR-based detection**: Sigmoid probability `Pd = 1/(1 + exp(-steepness * (SNR - threshold)))` when `power_model_enabled=True`
- **Free-space path loss**: `PL = 20*log10(distance) + 20.0 dB` for emitter-receiver geometry
- **Noise floor**: Configurable `noise_floor_dbm` (default -100 dBm)
- **Instantaneous receiver bandwidth**: Configurable receiver bandwidth (default 20.0 MHz)
- **Switching/settling time**: Time penalty for retuning to a different band
- **Dwell integration**: Multiple sub-dwell steps increase detection probability
- **Independent receivers**: Each `VirtualReceiver` maintains its own position, scan history, and detection state

### Multi-Receiver Validation & Metrics
Multi-receiver scheduling tracks:
- **Receiver utilization**: Proportion of scanning steps allocated to each receiver (`scans(r) / total_scans`).
- **Receiver coverage**: Unique bands scanned by each receiver over an observation window.
- **Receiver starvation**: Maximum elapsed scan steps without visiting a receiver or band.
- **Per-receiver detection rate**: Detection efficiency conditioned on receiver-specific SNR and path geometry.

### PDW Integration & Causality Boundary
- **Causality contract**: Observations at time $t$ are ingested strictly monotonically. The scheduler state is never updated with future observations ($t' > t$).
- **Boundary classification**:
  1. *Synthetic RF simulation*: Configurable physical path loss and Markov dynamics.
  2. *PDW schema compatibility*: Pulse Descriptor Word formats ingested and verified.
  3. *PDW replay*: Streamed event replay respecting strict historical causality.
  4. *Real RF validation*: All current benchmarks and tests use synthetic simulations or synthetic PDWs. Unless explicit authorized real RF recordings are present in `data/`, all evaluations remain synthetic.

