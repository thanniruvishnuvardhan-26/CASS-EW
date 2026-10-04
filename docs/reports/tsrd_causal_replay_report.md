# TSRD Causal Replay & Counterfactual Evaluation Report
**Project:** CASS-EW — Cognitive Adaptive Smart Scan for Electronic Warfare (SIH 2026 PS 26055)  
**Dataset:** Alan Turing Institute — Turing Synthetic Radar Dataset (TSRD)  

---

## 1. The Scan-Bias Problem & Causality Resolution

A fundamental flaw in naive historical replay is the **Scan-Bias Assumption**:
- If a historical receiver executed a periodic scan and dwelled on Band 0 during $[t_1, t_2]$, only pulses emitted in Band 0 during that window were digitized into the dataset.
- If an adaptive cognitive scheduler (such as CASS-EW) decides at $t_1$ to dwell on Band 3, the absence of recorded pulses in a historical SCAN file **does not prove Band 3 was silent**. It merely proves the legacy receiver was not looking at Band 3.

### CASS-EW Causal Architecture
To eliminate scan bias and allow true counterfactual evaluation:
1. **STARE Mode Grounding:** TSRD STARE mode files (`stare/train_stare`, `stare/val_stare`, `stare/test_stare`) provide unsteered continuous reception across broad frequency regimes.
2. **DatasetRFEnvironment:** Pulses are indexed in an array ordered by physical time and frequency.
3. **DatasetVirtualReceiver:** When the CASS-EW scheduler commands dwell on $[f_{\text{low}}, f_{\text{high}}]$ during $[\tau_{\text{start}}, \tau_{\text{end}}]$, the virtual receiver queries the indexed environment for pulses that physically existed during that interval.
4. **Counterfactual Interception:** If an emitter transmitted in Band 3 during that dwell, CASS-EW receives a physical detection opportunity. If another scheduler dwelled on Band 1 during the same interval, it receives silence.

---

## 2. Rigorous Performance Definitions

| Outcome | Physical Signal Opportunity | Receiver Reported Detection | Interpretation |
|:---|:---:|:---:|:---|
| **True Opportunity** | **YES** | — | Pulse existed in selected band during dwell interval |
| **Hit (Interception)** | **YES** | **YES** | Receiver successfully detected the active emitter opportunity |
| **Miss** | **YES** | **NO** | Signal opportunity existed, but receiver failed to detect (e.g. sub-threshold SNR) |
| **False Alarm** | **NO** | **YES** | Noise/interference triggered detector when no signal opportunity existed |
| **Quiet Dwell** | **NO** | **NO** | Band was inactive during dwell and receiver correctly reported silence |

### Formula Denominators
- $\text{Interception Rate } (P_d) = \frac{\text{Total Hits}}{\text{Total Opportunities}}$
- $\text{Miss Rate } (P_m) = \frac{\text{Total Misses}}{\text{Total Opportunities}} = 1 - P_d$
- $\text{False Alarm Rate } (P_{fa}) = \frac{\text{Total False Alarms}}{\text{Total Scans} - \text{Total Opportunities}}$
- $\text{Scan Efficiency} = \frac{\text{Total Hits}}{\text{Total Dwells}}$
