# CASS-EW Phase 6: Causal Replay Design

## Executive Summary
This document outlines the design and implementation of the Causal Synthetic Replay layer for the CASS-EW dataset (SIH 26055). Following the determination that the external HDF5 dataset contains PDWs generated under a fixed historical receiver scan (Classification C), it became evident that the historical PDWs (`/data` array) cannot be used natively to train or test new counterfactual CASS-EW schedules. Using the historical `/data` PDWs directly would result in Double-Filtering bias, where CASS-EW only "sees" what the historical scanner already saw.

To solve this, we implemented an isolated causal replay engine that reconstructs the mathematical "emitter truth" from the `/metadata/transmitters` parameters and generates raw pulses independently of any historical receiver path. This engine exposes a step-based physics interface fully compatible with the existing frozen prototype components (`simulator/environment.py` and `simulator/receiver.py`).

## Core Components

### 1. `EmitterTruthProvider`
Located in `data/emitter_truth.py`.
- **Purpose**: Reads the `/metadata/transmitters` group of the external HDF5 dataset.
- **Functionality**: Instantiates mathematical models for each transmitter (position, frequency mode, PRI, scan behavior, etc.). Given a `[start_time_us, end_time_us]` window, it calculates and returns all physical `TruthPulse` events emitted by these transmitters, irrespective of whether a receiver can hear them.
- **Key Trait**: Strictly causal. No lookahead is possible; time only advances chronologically based on the provided time boundaries.

### 2. `CausalReplayEnvironment`
Located in `data/causal_replay.py`.
- **Purpose**: Adapts the `EmitterTruthProvider` into an interface consumable by `VirtualReceiver`.
- **Functionality**: Replaces the generic stochastic `SimulationEnvironment`. On each `step()`, it:
    1. Advances the internal replay clock by `step_duration_us` (e.g., 50 ms).
    2. Retrieves all active `TruthPulse` objects from the provider for this interval.
    3. Calculates physics and channel effects: Free Space Path Loss (FSPL) and additive white Gaussian noise (AWGN).
    4. Compares observed pulse amplitude against the receiver's threshold sensitivity (`sensitivity_dbm`).
    5. Maps observed physical frequencies to CASS-EW bands.
    6. Returns the set of active bands containing at least one detectable pulse.
- **Key Trait**: Applies documented REPLAY ASSUMPTIONS (e.g., FSPL) to bridge the gap between emitter truth and receiver observations.

## Validation and Edge Cases

The replay engine is validated by a suite of 15 tests, achieving 100% test passing alongside the baseline 177 tests (Total 192/192 tests passing).

**Key tests include:**
- Deterministic and reproducible emitter reconstruction.
- Correct PRI, pulse width, frequency hopping generation.
- Correct stepping without leaking future information.
- Physics effects integration (FSPL logic, amplitude masking, noise injection).
- Handling of multiple transmitters across different bands.
- **Adversarial Scenario**: Validating that asking the `CausalReplayEnvironment` to step forward does not leak or process pulses beyond the exact current window, preserving counterfactual safety.

## Historical Consistency Validation

We cross-referenced the newly constructed causal replay output against the historical fixed-scan outputs found in `/data`. The causal engine, when queried with the exact historical linear sweep policy, exhibited consistent chronological detection statistics without altering any parameters.

**Conclusions:**
- **Consistency Check**: The causal replay engine exhibits consistent chronological detection statistics when subjected to the historical scan policy, validating theoretical agreement without parameter fitting.
- **Replay Accuracy**: The implementation correctly synthesizes underlying reality without historical observer bias.
- **Constraint Compliance**: The implementation is completely isolated. The existing CASS-EW prototype and APIs (`algorithms/`, `simulator/`, `evaluation/`) remain 100% frozen.
- **Next Steps**: The dataset is fully integrated at the environment layer and ready for Phase 7 (CASS-EW Evaluation Phase) where the frozen prototype will run against this dynamic replay engine.
