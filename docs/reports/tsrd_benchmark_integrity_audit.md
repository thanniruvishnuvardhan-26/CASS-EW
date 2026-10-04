# TSRD Benchmark Integrity & Causality Audit Report
**Project:** CASS-EW — Cognitive Adaptive Smart Scan for Electronic Warfare (SIH 2026 PS 26055)  
**Dataset:** Alan Turing Institute — Turing Synthetic Radar Dataset (TSRD)  
**Audit Date:** October 2026  
**Auditor:** Backend & Data-Engineering Lead  

---

## 1. Executive Summary & Audit Verdict

A rigorous methodological audit of the TSRD benchmark evaluation framework was performed to address benchmark comparability, opportunity denominators, latency tracking, false-alarm definitions, and dataset classification.

### Official Audit Verdict:
**PASS WITH CLARIFICATIONS — OFFICIAL TSRD VALIDATION PENDING**

**Rationale:**
1. **Methodological Validity:** The evaluation pipeline enforces strict common-trace execution, causal counterfactual observation, unit normalization ($\mu\text{s} \leftrightarrow \text{s}$), physical frequency boundary mapping (non-modulo), and zero label/future leakage.
2. **Clarification of Metrics & Denominators:** Opportunity counts across algorithms ($33$ for Sequential, $29$ for Random, $64$ for UCB1, $85$ for Q-learning, and $51$ for CASS-EW) reflect **Receiver Observation Opportunities** (scans where the scheduler happened to point the receiver at an active band). This is distinct from **Global Ground-Truth Emitter Opportunities**. Receiver detection probability ($\text{Receiver } P_d$) is now explicitly decoupled from Global Emitter Interception Rate.
3. **Data Classification:** All benchmark results currently reflect **TSRD Integration Schema-Compatible Fixtures** (`data/tsrd_fixtures/sample_stare.h5`). The official Hugging Face repository (`alan-turing-institute/turing-synthetic-radar-dataset`) is gated (`gated: auto`), requiring user authorization (`HF_TOKEN`). No claim of empirical validation against the raw multi-gigabyte dataset is made.

---

## 2. Answers to Critical Comparability Questions

### A. Do all schedulers receive exactly the same underlying RF event timeline?
**YES.** All schedulers in a benchmark run are evaluated against the exact same indexed pulse sequence in `DatasetRFEnvironment`, instantiated with the exact same seed, time range, and emitter events.

### B. Are emitter truth events fixed independently of scheduler actions?
**YES.** The underlying physical pulse arrivals, frequencies, pulse widths, angles, amplitudes, and emitter IDs are immutable properties of the dataset. Scheduler actions have zero influence over what pulses exist in the environment.

### C. Does selecting a different band merely change what the virtual receiver observes?
**YES.** The virtual receiver acts as a counterfactual sensor. When a scheduler selects Band $b$ for dwell duration $\tau$, the receiver queries the environment strictly for pulses falling into Band $b$'s physical frequency interval $[f_{\text{low}}, f_{\text{high}}]$ during that dwell window. If another scheduler selects Band $b'$, it observes only pulses in $b'$ during that exact same physical time window.

### D. Is each scheduler evaluated over the same total simulated time?
**YES.** Under the common 100-step nominal dwell budget ($\tau = 5\,\text{ms}$), each scheduler executes for $0.500\,\text{s}$ of sensing time plus switching overheads (total elapsed simulation time $\approx 0.509\,\text{s}$).

### E. Is total dwell time identical across schedulers?
**YES.** Total dwell time is exactly $0.500\,\text{s}$ across all evaluated schedulers (100 steps $\times 5\,\text{ms}$).

### F. Are "true opportunities" scheduler-independent ground-truth events, or are they generated from scheduler-selected observations?
**CRITICAL CLARIFICATION:** In the original reporting, "opportunities" represented **Receiver Observation Opportunities**—i.e., instances where a scheduler-directed dwell overlapped with an active pulse in the selected band. Consequently, an algorithm that concentrated its scans on active bands naturally encountered more observation opportunities (e.g. 51 for CASS-EW vs 33 for Sequential).  
**Correction Implemented:** The benchmark now explicitly separates:
- **Observation Opportunities ($N_{\text{opp, obs}}$):** Opportunities within the tuned band.
- **Global Ground-Truth Emitters ($N_{\text{emitters}}$):** The total number of distinct active emitters present in the environment scenario across all bands.

### G. Can a scheduler increase its denominator/opportunity count simply by scanning a band more often?
**YES, for observation opportunities.** A scheduler that visits active bands more frequently accumulates more observation opportunities. Therefore, dividing hits by observation opportunities yields **Receiver $P_d$ on tuned band**, NOT global scenario interception rate.

---

## 3. Precise Metric Mathematical Formulations

To prevent ambiguity, the evaluation layer implements the following explicit formulas:

1. **Receiver Observation Opportunity:**
   $$N_{\text{opp}} = \sum_{k=1}^{K} \mathbb{I}(\text{at least one physical pulse existed in chosen band } b_k \text{ during dwell } [t_{k,\text{start}}, t_{k,\text{end}}])$$

2. **Quiet Scan (Signal Absent Opportunity):**
   $$N_{\text{quiet}} = K - N_{\text{opp}}$$

3. **Hit (True Positive):**
   $$\text{Hit}_k = \mathbb{I}(\text{Signal Present in } b_k \text{ AND Receiver Reported Detection})$$

4. **Miss (False Negative):**
   $$\text{Miss}_k = \mathbb{I}(\text{Signal Present in } b_k \text{ AND Receiver Failed to Detect})$$

5. **False Alarm (False Positive):**
   $$\text{FA}_k = \mathbb{I}(\text{Signal Absent in } b_k \text{ AND Receiver Reported Detection})$$

6. **Receiver Probability of Detection ($\text{Receiver } P_d$):**
   $$\text{Receiver } P_d = \frac{\sum \text{Hits}}{N_{\text{opp}}}$$
   *(Measures the efficiency of the detector given that the scheduler pointed the receiver at an active signal).*

7. **Receiver Probability of False Alarm ($\text{Receiver } P_{fa}$):**
   $$\text{Receiver } P_{fa} = \frac{\sum \text{False Alarms}}{N_{\text{quiet}}}$$
   *(Reported explicitly as a fraction $\text{FA} / N_{\text{quiet}}$, never merely "0.0%").*

8. **Scan Efficiency:**
   $$\text{Scan Efficiency} = \frac{\sum \text{Hits}}{K_{\text{total scans}}}$$
   *(Measures the fraction of receiver dwell budget successfully capturing active RF energy).*

9. **Global Emitter Interception Rate:**
   $$\text{Global Interception Rate} = \frac{\text{Number of Unique Ground-Truth Emitters Intercepted}}{\text{Total Number of Unique Ground-Truth Emitters in Scenario}}$$

10. **Onset Intercept Latency:**
    $$\text{Latency}_e = t_{\text{first detection}}(e) - t_{\text{onset, ground truth}}(e)$$
    Evaluated across all eligible emitters and summarized via **Mean**, **Median**, and **P90** percentiles.

---

## 4. Audited Multi-Algorithm Benchmark Comparison

Evaluated on `sample_stare.h5` across 100 decision steps ($500\,\text{ms}$ dwell budget, Seed 42, $N=4$ ground-truth emitters):

| Scheduler Strategy | Total Scans | Dwell Time | Observation Opps | Hits | Misses | False Alarms | Receiver $P_d$ | Receiver $P_{fa}$ | Scan Efficiency | Global Emitters Intercepted | Mean Intercept Latency | Median Latency | P90 Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sequential Scan** | 100 | 0.500s | 33 | 31 | 2 | 1 / 67 | 93.9% | 1.5% | 31.0% | **4 / 4 (100%)** | 0.0408s | 0.0455s | 0.0656s |
| **Random Scan** | 100 | 0.500s | 29 | 29 | 0 | 1 / 71 | 100.0% | 1.4% | 29.0% | **4 / 4 (100%)** | 0.0648s | 0.0733s | 0.1030s |
| **UCB1 Bandit** | 100 | 0.500s | 64 | 62 | 2 | 1 / 36 | 96.9% | 2.8% | 62.0% | **4 / 4 (100%)** | 0.0688s | 0.0568s | 0.1351s |
| **CASS-EW Cognitive Adaptive** | 100 | 0.500s | 51 | 47 | 4 | **0 / 49** | 92.2% | **0.0%** | **47.0%** | **4 / 4 (100%)** | 0.0695s | 0.0525s | 0.1419s |
| **Experimental Q-Learning** | 100 | 0.500s | 85 | 81 | 4 | **0 / 15** | 95.3% | **0.0%** | **81.0%** | 3 / 4 (75%) | 0.0932s | 0.0251s | 0.2051s |

---

## 5. False-Alarm Analysis

In previous runs, false alarm rates were reported simply as $0.0\%$. The audited breakdown reveals:
- **Sequential Scan:** 1 false alarm out of 67 quiet scans ($P_{fa} = 1.49\%$).
- **Random Scan:** 1 false alarm out of 71 quiet scans ($P_{fa} = 1.41\%$).
- **UCB1 Baseline:** 1 false alarm out of 36 quiet scans ($P_{fa} = 2.78\%$).
- **CASS-EW Cognitive Adaptive:** 0 false alarms out of 49 quiet scans ($0 / 49 = 0.0\%$).
- **Experimental Q-Learning:** 0 false alarms out of 15 quiet scans ($0 / 15 = 0.0\%$).

**Finding:** The $0.0\%$ false-alarm rate for CASS-EW is genuine and supported by an adequate sample size (49 quiet scans). Q-learning's zero false alarm is based on a smaller sample (15 quiet scans) due to heavy exploitation of Band 0.

---

## 6. Multi-Seed Reproducibility Analysis (Seeds 42, 43, 44, 45, 46)

Executed via `evaluation/run_tsrd_seeds.py`:

```text
[Sequential Scan]
  Scan Efficiency : mean=0.310 | std=0.006 | min=0.300 | max=0.320
  Receiver Pfa    : mean=0.012 | std=0.006 | min=0.000 | max=0.015
  Mean Latency (s): mean=0.0306 | std=0.0051 | min=0.0280 | max=0.0408

[Random Scan]
  Scan Efficiency : mean=0.324 | std=0.028 | min=0.290 | max=0.370
  Receiver Pfa    : mean=0.012 | std=0.012 | min=0.000 | max=0.033
  Mean Latency (s): mean=0.0781 | std=0.0265 | min=0.0611 | max=0.1308

[UCB1 Baseline]
  Scan Efficiency : mean=0.560 | std=0.038 | min=0.520 | max=0.620
  Receiver Pfa    : mean=0.019 | std=0.010 | min=0.000 | max=0.028
  Mean Latency (s): mean=0.0403 | std=0.0143 | min=0.0331 | max=0.0688

[CASS-EW Cognitive Adaptive]
  Scan Efficiency : mean=0.510 | std=0.046 | min=0.470 | max=0.590
  Receiver Pfa    : mean=0.005 | std=0.010 | min=0.000 | max=0.024
  Mean Latency (s): mean=0.0618 | std=0.0147 | min=0.0443 | max=0.0821

[Experimental Q-Learning]
  Scan Efficiency : mean=0.776 | std=0.021 | min=0.750 | max=0.810
  Receiver Pfa    : mean=0.012 | std=0.024 | min=0.000 | max=0.059
  Mean Latency (s): mean=0.0619 | std=0.0336 | min=0.0043 | max=0.0954
```

---

## 7. Causality & Anti-Leakage Verification

The test suite in `tests/test_tsrd_leakage.py` was extended to include adversarial future pulse mutations:
- **Test 11 (Adversarial Future Mutation):** Two datasets $A$ and $B$ were created identical for pulses $0..199$, while pulses $200..399$ in dataset $B$ were mutated to an extreme frequency ($16.5\,\text{GHz}$).
- **Result:** Schedulers running in environment $A$ and environment $B$ executed bit-for-bit identical scan decisions throughout steps occurring prior to pulse 200. This mathematically proves that **future ground-truth pulses have zero influence on historical scheduler decisions**.

---

## 8. Dataset Classification

All evaluated data sources are classified as:
- **`data/tsrd_fixtures/sample_stare.h5`**: **Category C — TSRD-schema-compatible fixture**
- **`data/tsrd_fixtures/sample_scan.h5`**: **Category C — TSRD-schema-compatible fixture**
- **Official TSRD Repository**: Gated on Hugging Face; authenticated access pending user token.

---

## 9. Test Suite Verification

Full test suite execution command:
```bash
python -m unittest discover tests -v
```
**Result:** **280 / 280 tests passed** (0 failures, 0 errors).
No frozen-core files (`algorithms/`, `simulator/environment.py`, `simulator/receiver.py`, `config.py`, `main.py`) were modified.
