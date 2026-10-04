# CASS-EW Final Prototype Freeze

Status: FROZEN

Project: SIH 26055
Version: CASS-EW-SIH26055-FINAL-PROTOTYPE

Date/Time of freeze: 2026-09-30T09:02:14+05:30
Commit Identifier: f2f1f58fe1f9ca353e51dc54112bc3bfcec11de8

Tests:
163 passing

Validation:
Synthetic / Replay

Real RF/PDW:
No authorized real dataset available

Benchmark seeds:
42, 43, 44, 45, 46

Benchmark Configuration:
- Dashboard demo entry point: `dashboard_demo.py`
- Final benchmark: `evaluation/final_cass_ew_benchmark.py`
- Dependencies: Standard library + external libs (if any) as captured in environment.

Frozen components:
- Belief adaptive scheduling
- Temporal reasoning
- Predictive scheduling
- Signal-pattern reasoning
- Multi-emitter reasoning
- Multi-receiver reasoning
- Spatial reasoning
- Event-based intercept-time evaluation
- PDW integration boundary
- Unified pipeline
- Dashboard
- Final benchmark

Known Limitations:
- **Synthetic Only**: All results are from synthetic simulation. No real RF data was available.
- **SNR Model**: Uses simplified monotonic attenuation (SNR = 100 / (1 + distance)).
- **Static Receivers**: Receiver positions are fixed. Mobile receivers are not supported.
- **No Waveform Analysis**: PDW adapter handles metadata only, not raw I/Q data.
