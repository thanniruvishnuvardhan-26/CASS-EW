# CASS-EW Final Limitations & Engineering Constraints

**Project**: Cognitive Adaptive Smart Scan for Electronic Warfare (CASS-EW)  
**SIH Problem Statement**: 26055  
**Version**: Part 3 Final Audit & Freeze  
**Document Classification**: Transparent Technical Self-Audit  

---

## 1. Scope of the System

CASS-EW is strictly a **cognitive scan scheduler** designed to command which RF channel and receiver antenna node to tune to next, for what dwell time, under receiver switching and settling constraints.

It is **NOT**:
- An Electronic Support Measures (ESM) signal deinterleaver.
- An emitter classification or threat identification system.
- An Electronic Countermeasures (ECM) jammer resource allocator.
- A raw I/Q signal processing chain.

---

## 2. RF Simulation & Environment Modeling Assumptions

### 1. Synthetic RF Environment
All RF environments evaluated within this repository are synthetic simulations modeling pulse presence, duty cycles, burst dynamics, and frequency agility.
- **Propagation Model**: Attenuation is modeled using deterministic log-distance free-space path loss:
  $$\text{PL}(d) = 20\log_{10}(d) + 20\text{ dB}$$
  Real-world multipath fading, Rayleigh/Rician scattering, atmospheric absorption (rain/fog), and terrain shadowing are **not modeled**.
- **Discrete Time Granularity**: The simulation progresses in integer time steps (ticks) representing normalized dwell time units. Sub-nanosecond intra-pulse modulations are abstracted away.

### 2. Receiver Hardware Model
- **Instantaneous Bandwidth (IBW)**: Modeled as discrete, non-overlapping channels (Bands 0 through 9). Adjacent-channel leakage, intermodulation distortion, and front-end preselector roll-off are idealized.
- **Detector Approximation**: Detection probability ($P_d$) is computed via a continuous logistic approximation of SNR relative to receiver threshold, rather than physical coherent/non-coherent matched filter integration of received RF waveforms.
- **No Raw I/Q Processing**: The receiver outputs scalar telemetry (boolean detection, estimated SNR, effective dwell duration, switching latency). The software does not process raw in-phase and quadrature (I/Q) ADC voltage samples.

---

## 3. Absence of Field Validation & Authorized Real RF Data

- **No Real RF Recordings Present**: The repository does not include classified or proprietary live RF spectrum recordings from operational defence assets.
- **No Claims of Operational Readiness**: All claims of performance, latency reduction, and anti-starvation guarantees are valid **only within the documented simulation models and synthetic PDW schemas**.
- **PDW Boundary**: The system features complete Pulse Descriptor Word (PDW) schema compatibility and causal replay engines. However, testing with external PDWs remains synthetic unless authorized sovereign datasets are provided under appropriate security protocols.

---

## 4. Algorithmic Boundaries & Scalability

### 1. Q-Learning Status
Tabular Q-learning (`algorithms/rl_scheduler.py`) is retained solely as an experimental baseline. Due to state-space explosion ($3^{10} = 59,049$ states for single-receiver 10 bands), tabular RL fails to scale to continuous dwell, dense hopping, or multi-receiver deployments and suffers severe band starvation.

### 2. Markov Frequency-Hop Transition Order
The `FrequencyHopPredictor` uses a first-order Markov transition matrix:
$$P(\text{band}_{t+1} \mid \text{band}_t)$$
While highly effective for pseudo-random and cyclic hopping patterns, it does not reconstruct long-period cryptographic hopping sequences without higher-order sequence tracking.

### 3. Starvation vs. Exploitation Trade-off
Enforcing strict maximum revisit intervals (anti-starvation) inherently reduces the time available to camp on known active emitters. The system intentionally accepts a marginal reduction in peak single-band interception rate to ensure zero blind spots across unobserved frequencies.
