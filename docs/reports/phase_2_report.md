# CASS-EW Phase 2 Validation Report

## 1. Architecture Changes
In Phase 2, the simulator was refactored into modular components while strictly maintaining backward compatibility with Phase 1. The core monolithic `RFEnvironment` was decomposed into a base `Emitter` class and distinct subclasses for each behavior: `IntermittentEmitter`, `PeriodicEmitter`, `HoppingEmitter`, `PersistentEmitter`, and `BurstEmitter`. A new `VirtualReceiver` class was introduced to formally model scanning, switching costs, dwell time, and observation generation. 

## 2. Exact Temporal Contract
The precise meaning of `environment.time` and its step mechanics have been explicitly defined and verified:
- `environment.time` tracks the number of intervals elapsed.
- At time `t=0`, no intervals have elapsed, and the environment state has not yet advanced.
- `step()` computes the state for the interval `(t, t+1]` and advances time to `t+1`.
- Therefore, at time `t=1`, the interval `(0, 1]` has just completed, and its state is the one observed.
- **Verification:** `test_01_time_semantics` independently verifies this behavior.

## 3. Emitter Semantics
Emitter state changes deterministically (or pseudo-randomly via fixed RNG) at the exact boundary of each interval.
- **Periodic:** Configured with `period=4` and `duty_cycle=2`, the sequence for `t=0..7` is `[True, True, False, False, True, True, False, False]`. Boundaries like `duty_cycle=1` and `duty_cycle=period` were tested to behave exactly as expected.
- **Hopping:** Hop sequences deterministically cycle based on the `hop_interval`. For `hop_interval=2` over `[6, 7, 8]`, the sequence is `[6, 6, 7, 7, 8]`.
- **Burst:** Configured with `burst_duration=2`, `inter_burst_duration=3`, the sequence is `[True, True, False, False, False, True, True, False]`.
- **Verification:** Independent hand-derived truth tables in tests 04, 05, and 06 prove these exact semantics.

## 4. Receiver Semantics
The receiver scan operation follows the **Option B semantic contract** (one entry per receiver scan/observation event).
- **Initial state:** `current_band = None`.
- **First acquisition:** Incurs no switching cost since there is no prior tuned band.
- **Switching:** When switching to a new band, the environment advances by `switching_time` *before* the dwell begins.
- **Reacquiring:** Tuning to the already-tuned band incurs zero switching cost.
- **Dwell:** The environment is advanced `dwell_time` times. The total effective observation duration equals `dwell_time`. 
- **Observation:** A single `ReceiverObservation` object is created spanning `dwell_start_time` (after switching) to `dwell_end_time` (at the conclusion of the dwell steps). The `detection_result` aggregates findings over this span.
- **Verification:** `test_02_receiver_scan_semantics` proves this timing fixture mathematically.

## 5. Emitter Factory Decision
The legacy `Emitter(behavior=...)` constructor was replaced with an `Emitter.__new__()` factory to silently vend appropriate subclasses (`IntermittentEmitter`, `PeriodicEmitter`, etc.). 
- **Why it exists:** It maintains perfect backward compatibility with Phase 1 tests and evaluator scripts that instantiate emitters using the monolithic signature.
- **Limitations:** Subclassing `Emitter` externally might bypass the factory if not careful, and legacy signatures clutter the `__init__` parameters.
- **Verification:** `test_03_emitter_factory_audit` verifies that all 5 behaviors properly construct the correct subclass, pass `isinstance(..., Emitter)`, and gracefully throw explicit `ValueError`s for invalid strings. 

## 6. Seed and Reset Semantics
Every emitter independently maintains its own `np.random.default_rng(seed)`.
- **Reproducibility:** Two runs of a stochastic emitter (e.g. Intermittent, Hopping) with identical seeds will produce perfectly equivalent state traces, proven in `test_07_reset_seed_reproducibility`.
- **RFEnvironment Override:** When `RFEnvironment.reset(seed=...)` is called, it intentionally *overrides* the explicitly configured seed of all attached emitters with a derived deterministic seed (`env.seed * 1000 + i + 1`). This centralized reset semantics is documented and explicitly tested.

## 7. Observability Boundary
A strict separation exists between the observable RF environment and hidden ground truth.
- The `ReceiverObservation` (and `VirtualReceiver.scan_history`) only expose the final boolean detection over the dwell span.
- **Verification:** `test_08_observable_vs_ground_truth_separation` constructs two parallel environments that have identical behavior up to `t=5`, but diverge at `t=6`. The test proves that the receiver scan history generated up to `t=5` is identically equivalent across both worlds, mathematically proving that future/hidden state does not leak backwards into the current observation.

## 8. Test Inventory
- `test_environment.py`: Legacy tests verified against modular components.
- `test_receiver.py`: Legacy tests + `test_dwell_stepping_semantics` + `test_scan_history_semantic_contract`.
- `test_reproducibility.py`: Legacy metric stability tests.
- `test_phase2_semantics.py`: 8 new explicit semantic fixture tests covering temporal bounds, receiver timing, factory audits, hand-derived hopping/periodic/burst truth tables, seed resets, and observability separation.

All 46 tests currently pass.

## 9. Phase 1 Regression Results
The Phase 1 legacy benchmark was successfully re-run using the exact Phase 1 configuration. 
```text
Algorithm       Interception   False Alarm   Miss Rate   Efficiency
------------------------------------------------------
Sequential            6.92%         4.96%      93.08%       0.1376
Bayesian             15.79%         5.23%      84.21%       0.3144
RL                   21.73%         5.82%      78.27%       0.4316
```
**Important Distinction:** 
A. These benchmark numbers remain unchanged precisely because `evaluation/final_benchmark.py` bypasses `VirtualReceiver` and uses its own decoupled `observe()` tracking. 
B. The receiver architecture was independently validated for correctness through `test_phase2_semantics.py` (specifically `test_02_receiver_scan_semantics` and `test_08_observable_vs_ground_truth_separation`). 

## 10. Phase 2 Scope Audit
- [x] PASS: Modular RF environment
- [x] PASS: Emitter abstraction
- [x] PASS: All required emitter types (Intermittent, Periodic, Hopping, Persistent, Burst)
- [x] PASS: Configurable bands
- [x] PASS: Receiver abstraction
- [x] PASS: Switching/settling
- [x] PASS: Dwell
- [x] PASS: Detection
- [x] PASS: Noise/SNR/Pd/Pfa
- [x] PASS: Observable/ground-truth separation
- [x] PASS: Multiple emitters
- [x] PASS: Reset semantics
- [x] PASS: Phase 1 compatibility
- [x] PASS: Tests
- [x] PASS: Baseline regression

*No Phase 3 algorithms or features were introduced.*

## 11. Known Limitations & What Was Not Validated
- Phase 2 did not introduce dynamic SNR degradation or pulse-level simulation; the time interval remains a macroscopic "step".
- Interference/collision modeling between multiple emitters in the same band is currently treated as a simple boolean OR (signal is present if any emitter is active).
- Real-time/wall-clock performance bounds were not formally benchmarked, only logical correctness.
