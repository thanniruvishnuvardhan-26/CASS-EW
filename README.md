# CASS-EW (Cognitive Adaptive Smart Scan for Electronic Warfare)
**SIH Problem Statement**: 26055  
**Version**: Part 3 Final Audit & Master Upgrade  

---

## 1. Project Objective

CASS-EW is a cognitive, adaptive RF spectrum scan scheduler designed for Electronic Warfare (EW) surveillance and interception operations. It dynamically balances exploitation of known periodic/hopping radar emitters with intelligent exploration of unobserved bands to minimize intercept latency while strictly preventing frequency starvation.

---

## 2. Core Architecture

The system operates strictly on observable RF detector outputs without ground-truth leakage:

```
OBSERVE (Receiver Detection / Miss)
  → UPDATE RF KNOWLEDGE MAP (Belief, Intervals, Markov Transitions, Spatial Profiles)
  → ESTIMATE UNCERTAINTY & STALENESS
  → PREDICT NEXT PULSE / FREQUENCY HOP
  → EXPLORE vs EXPLOIT vs ANTI-STARVATION
  → SELECT RECEIVER NODE
  → SELECT RF BAND
  → SELECT DWELL TIME
  → EXECUTE HARDWARE SCAN (Switching / Settling / Integration)
  → REPEAT
```

### Official Primary Scheduler
- **`CognitiveAdaptive_Integrated` (`algorithms/phase8_spatial_scheduler.py`)**: Multi-factor causal evidence fusion scheduler backed by the `RFKnowledgeMap`. Combines Bayesian belief, temporal interval history, next-pulse prediction, cross-band pattern correlation, multi-receiver spatial geometry, uncertainty estimation, staleness deadlines, Markov frequency-hop predictions, and dynamic RF environment-change dampening.

### Experimental / Baseline Schedulers
1. **`Sequential` (`algorithms/sequential.py`)**: Deterministic round-robin baseline.
2. **`Random` (`algorithms/random_scheduler.py`)**: Stochastic uniform baseline.
3. **`AdaptiveBelief` (`algorithms/adaptive_belief.py`)**: Bayesian belief updating with $\epsilon$-greedy exploration.
4. **`UCB1` (`algorithms/ucb_scheduler.py`)**: Multi-armed bandit balancing empirical detection rate with logarithmic exploration bonus ($c = 1.414$).
5. **`RL_Experimental` (`algorithms/rl_scheduler.py`)**: Tabular Q-learning prototype (retained strictly as an experimental baseline).

---

## 3. Installation & Dependencies

Requires Python 3.9+. Install dependencies:
```bash
pip install numpy flask matplotlib scipy
```

---

## 4. Execution Commands

### Run Full Test Suite (252 Tests Passing)
```bash
python -m unittest discover tests -v
```

### Run Official S1-S10 Scenario Benchmark
Runs across 10 operational scenarios and 5 random seeds (42, 43, 44, 45, 46):
```bash
python evaluation/run_scenarios_benchmark.py
```
Outputs:
- `results/final_benchmark.csv`
- `results/final_benchmark.json`

### Run Component Ablation Study (A through H)
```bash
python evaluation/run_ablation_study.py
```
Outputs:
- `results/ablation.csv`
- `docs/ABLATION_REPORT.md`

### Launch Official Engineering Dashboard
```bash
python web_app.py
```
Access in browser: `http://localhost:5000`

---

## 5. Standardized Evaluation Metrics

For every scenario and scheduler, the benchmarking suite calculates:
- **Interception Rate (IR)**: Fraction of active emitter pulse opportunities intercepted.
- **Miss Rate**: Complement of interception rate ($1 - \text{IR}$).
- **False Alarm Rate (FAR)**: Spurious detection rate on inactive bands.
- **Mean / Median / P90 Intercept Time**: Latency from emitter episode onset to first true-positive detection.
- **Scan Efficiency**: Detections per unit dwell time.
- **Spectrum Coverage**: Percentage of channels actively monitored.
- **Worst-Case Staleness**: Maximum elapsed steps between consecutive revisits to any channel.
- **Receiver Utilization**: Distribution of scan workload across distributed nodes.

---

## 6. Reproducibility Guarantee

All experiments and benchmarks are bit-for-bit deterministic when executed with fixed seeds. The default benchmarking suite runs on seeds `[42, 43, 44, 45, 46]`. Repeated executions with identical seeds produce zero variance.

---

## 7. Limitations & Synthetic Data Disclaimers

Detailed technical limits are documented in [`docs/FINAL_LIMITATIONS.md`](docs/FINAL_LIMITATIONS.md):
- **Synthetic RF Environment**: Propagation is modeled via deterministic free-space path loss ($20\log_{10}(d) + 20\text{ dB}$). Multipath, terrain shadowing, and atmospheric fading are not modeled.
- **No Raw I/Q Processing**: Receivers output scalar detections and estimated SNR; raw digitized ADC waveforms are outside scope.
- **Absence of Real Field Data**: All evaluations use synthetic simulations or synthetic Pulse Descriptor Words (PDWs). No field validation on operational EW hardware or sovereign radar targets is claimed.
- **Scan Scheduling Scope**: CASS-EW strictly solves receiver antenna tuning and dwell allocation. It does not perform emitter identification or threat classification.
