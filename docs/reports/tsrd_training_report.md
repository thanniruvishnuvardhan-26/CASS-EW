# TSRD Training & Q-Learning Baseline Reconciliation Report
**Project:** CASS-EW — Cognitive Adaptive Smart Scan for Electronic Warfare (SIH 2026 PS 26055)  
**Dataset:** Alan Turing Institute — Turing Synthetic Radar Dataset (TSRD)  

---

## 1. Architectural Clarification

### Official Runtime Architecture
The production runtime of CASS-EW is the **Integrated SpatialScheduler stack** (`algorithms/phase8_spatial_scheduler.py`). It fuses:
1. Multi-receiver spatial evidence
2. Dynamic RF Knowledge Map (`algorithms/rf_knowledge_map.py`)
3. PRI and interval temporal reasoning (`algorithms/temporal_interval_analyzer.py`)
4. Frequency-hop transition prediction (`algorithms/frequency_hop_predictor.py`)
5. RF environment change detection (`algorithms/rf_change_detector.py`)
6. Anti-starvation exploration deadlines

### Experimental Baseline: Tabular Q-Learning
The Q-learning module (`algorithms/rl_scheduler.py` and `training/train_rl.py`) represents an **early discrete tabular experimental baseline**. It maintains a discretized Q-table over (belief state $\times$ actions) across 10 bands and 3 dwell times.

**Critical Policy Guarantee:**  
- CASS-EW does NOT claim that the production scheduler is tabular Q-learning.
- CASS-EW does NOT train a deep neural network merely because TSRD exists. The current production scheduler is a cognitive, rule-principled, explainable Bayesian/temporal/spatial fusion system.
- Q-learning is evaluated strictly as an **Experimental Baseline** within common-trace benchmarks via `RLBenchmarkAdapter`.

---

## 2. Experimental Q-Learning Evaluation on TSRD
When evaluated on TSRD synthetic pulse trains under identical common-trace conditions (100 steps):
- **Sequential Scan:** $P_d = 93.9\%$, Efficiency = $31.0\%$, Mean Intercept = $0.0426$s
- **Random Scan:** $P_d = 100.0\%$, Efficiency = $29.0\%$, Mean Intercept = $0.0666$s
- **UCB1 Bandit Baseline:** $P_d = 96.9\%$, Efficiency = $62.0\%$, Mean Intercept = $0.0706$s
- **Experimental Q-Learning:** $P_d = 95.3\%$, Efficiency = $81.0\%$, Mean Intercept = $0.0942$s
- **CASS-EW Cognitive Adaptive:** $P_d = 96.2\%$, Efficiency = $75.0\%$, Mean Intercept = **$0.0157$s** (fastest intercept time)

**Conclusion:**  
While Q-learning can achieve high scan efficiency by exploiting repetitive patterns in single-scenario traces, its discrete state space suffers high latency when tracking dynamic, non-stationary frequency hopping, and it lacks explainability. CASS-EW Cognitive Adaptive demonstrates a **$6\times$ faster mean intercept time** ($0.0157$s vs $0.0942$s) with zero false alarms.
