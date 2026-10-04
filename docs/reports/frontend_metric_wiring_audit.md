# CASS-EW Frontend Metric Wiring Audit

## 1. Dashboard Data Flow

The data flow for the dashboard metrics operates as follows:

Python simulation (`web_app.py` stepping logic & `simulator/` components)
→ `web_app.py` (Constructs API/state payload in `/api/step` or `/api/benchmark`)
→ API/state payload (JSON sent over HTTP to frontend)
→ `app.js` (Parses JSON and updates DOM elements)
→ dashboard element (Live UI)

## 2. Metric Traceability Table

| Dashboard Metric | Source | Formula | Denominator | Live/Benchmark | Comparable? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Interception Rate** | `web_app.py` | `total_det / total_active` | Scanned active opportunities (not global) | Live | NO (Differs from benchmark) |
| **Total Obs** | `web_app.py` | `sim_state['total_obs']` | N/A (Count) | Live | N/A |
| **Detections** | `web_app.py` | `sim_state['total_det']` | N/A (Count) | Live | N/A |
| **Exploration** | `web_app.py` | `sched_state['exploration_selections']`| N/A (Count) | Live | N/A |
| **Exploitation** | `web_app.py` | `sched_state['exploitation_selections']`| N/A (Count) | Live | N/A |
| **Spatial Influenced** | `web_app.py` | `sched_state['spatial_influenced_selections']`| N/A (Count) | Live | N/A |
| **Mean Latency** | `intercept_time.py`| `mean(latencies)` | Intercepted episodes | Live | N/A |
| **Median Latency** | `intercept_time.py`| `median(latencies)` | Intercepted episodes | Live | N/A |
| **Eligible** | `intercept_time.py`| `len(tracker.episodes)` | N/A (Count) | Live | N/A |
| **Intercepted** | `intercept_time.py`| `len(intercepted_eps)` | N/A (Count) | Live | N/A |
| **Missed** | `intercept_time.py`| `len(missed_eps)` | N/A (Count) | Live | N/A |
| **Benchmark IR** | `web_app.py` | Hardcoded strings (e.g. "34.97%") | Global active opportunities (implied) | Benchmark | NO |
| **Benchmark Miss** | `web_app.py` | Hardcoded strings | Global active opportunities (implied) | Benchmark | N/A |
| **Benchmark FA** | `web_app.py` | Hardcoded strings | Inactive observations (implied) | Benchmark | N/A |
| **Benchmark Efficiency**| `web_app.py` | Hardcoded strings | Total observations (implied) | Benchmark | N/A |

## 3. Interception Rate Investigation

**Why does the Live Interception Rate display 63.64%?**

In `web_app.py`, the `total_active` denominator is incorrectly tracking only the active bands that the receiver *chose to scan*, not the global active bands across the spectrum. 
```python
was_active = band in gt['active_bands']
if was_active:
    sim_state['total_active'] += 1
```
Because of this, `total_det / total_active` calculates the receiver hardware's Conditional Probability of Detection (Pd)—essentially, "when the scanner pointed at an active signal, it detected it 63.64% of the time." This is a severe defect in `web_app.py`'s live calculation that matches the bug found in the Phase 7 audit, completely inflating the number.

## 4. Interception Event Investigation

**Why is Intercepted / Eligible = 8%?**

The Interception Events panel pulls data from `evaluation/intercept_time.py`.
- **Eligible (200)**: The total number of independent emitter activation "episodes" that occurred anywhere in the spectrum.
- **Intercepted (16)**: The number of those episodes where the receiver successfully pointed at the band and detected the signal before it turned off.
- **Missed (184)**: The number of episodes that started and ended without the receiver detecting them.
- **Mean/Median Latency**: The average/median time taken to intercept a signal (calculated only across the 16 successful interceptions).

The 8% figure (16 / 200) represents the *true* global episodic interception rate, contrasting sharply with the defective 63.64% shown in the Performance panel.

## 5. Benchmark Table Investigation

The values in the System Status & Benchmark table (Phase 8/10, Random, Sequential) originate from **hardcoded static strings** inside `web_app.py`'s `/api/benchmark` endpoint. They do not come from the current live simulation, nor do they run dynamically. They are simply loaded from a hardcoded dictionary representing a stored past benchmark.

## 6. Potential UI Ambiguities

1. **"Interception Rate" vs "Benchmark IR"**: The live Interception Rate (inflated hardware Pd) is labeled the same as the benchmark IR (true global interception rate), severely confusing the judge/user.
2. **"Interception Rate" vs "Intercepted"**: The true episodic interception rate (8%) is buried as a ratio in the Interception Events panel, contradicting the 63.64% Performance metric.
3. **Benchmark Data Source**: The Benchmark table is presented alongside live metrics without a clear indication that it contains static, hardcoded historical values rather than dynamically updating comparable run data.
4. **Denominator Obfuscation**: The live "Interception Rate" label does not describe its flawed denominator (scanned active ops vs global active ops).

## 7. Recommended Frontend Changes

1. **Rename Live IR**: Rename the live "Interception Rate" to "Hardware Pd (Scanned)" or fix the backend formula to track global active opportunities.
2. **Add Episodic IR**: Add a clear percentage representation of (Intercepted / Eligible) in the Interception Events panel.
3. **Clarify Benchmark Table**: Add a subtext/label indicating that the Benchmark table is "Historical Reference Data".
4. **Standardize Terminology**: Ensure "Interception Rate" strictly means `successful interceptions / global opportunities` across both the live dashboard and benchmark tables.

## 8. Frozen-Core Integrity

Confirmed:
- Algorithms were NOT modified.
- Scheduler behavior was NOT modified.
- Simulation behavior was NOT modified.
- Benchmark data was NOT modified.

FINAL VERDICT:
PASS WITH CLARIFICATIONS
