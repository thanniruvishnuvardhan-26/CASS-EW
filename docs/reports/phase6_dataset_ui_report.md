# Phase 6 Implementation Report: User-Facing Dataset Import, Validation & Replay Workflow

**Project:** CASS-EW — Cognitive Adaptive Smart Scan for Electronic Warfare (SIH 26055)  
**Phase:** Phase 6 — User-Facing Dataset Import, Validation & Replay Workflow  
**Date:** October 3, 2026  
**Status:** COMPLETE (342/342 tests passing)

---

## 1. Objective
To turn the validated Phase 5 backend dataset capability into a complete, usable, transparent demonstration and engineering console workflow:
```
SELECT / UPLOAD DATASET
      ↓
INSPECT DATASET (Columns / Keys / Preview)
      ↓
MAP FIELDS (ToA, FoA, PW, AoA, Amplitude, Emitter Label)
      ↓
DECLARE / CONFIRM UNITS (Explicit: s, ms, us, ns | Hz, kHz, MHz, GHz | dBm)
      ↓
VALIDATE DATASET (Validation Report & Strict Error/Ambiguity Check)
      ↓
REVIEW VALIDATION RESULT (Record counts, time & frequency coverage, warnings)
      ↓
CONFIGURE RUN (Seed, Steps, Dwell Time, Initial Band)
      ↓
RUN CASS-EW (Causal Replay via Frozen SpatialScheduler)
      ↓
VIEW RESULTS (Audited metrics, opportunity counts, band decisions, explainability)
      ↓
EXPORT RESULT (Download pure machine-readable DatasetRunResult JSON)
```

---

## 2. Existing Frontend Architecture Used
Phase 6 builds directly on top of the established CASS-EW web dashboard architecture:
- **Server**: Flask (`web_app.py`) providing REST endpoints serving JSON contracts.
- **Markup**: Semantic HTML (`frontend/templates/index.html`) retaining all 9 existing tiers. Phase 6 integrates Tier 10: "USER DATASET IMPORT, VALIDATION & REPLAY WORKFLOW".
- **Styling**: Flat technical RF/EW console visual language (`frontend/static/style.css`), avoiding generic SaaS styling, excessive rounded cards, or decorative graphics.
- **Client Logic**: Vanilla JavaScript (`frontend/static/app.js`) handling dynamic DOM updates, asynchronous fetch requests, and state management.

---

## 3. Dataset Workflow & User Interaction
1. **Dataset Selection / Upload**: User can select any pre-existing local fixture (`sample_pdws.csv`, `sample_pdws.json`, `sample_stare.h5`, `sample_scan.h5`) or upload a new `.csv`, `.json`, `.h5`, or `.hdf5` file.
2. **Schema Inspection**: Clicking "INSPECT" or completing an upload fetches the column headers / dictionary keys without executing the simulation.
3. **Field Mapping & Unit Selection**:
   - The user maps logical fields (`timestamp`, `frequency`, `pulse_width`, `aoa`, `amplitude`, `emitter_label`).
   - The interface explicitly marks `Emitter Label` with a prominent `[EVALUATOR ONLY]` badge to remind the user that emitter labels are never fed to the scheduler.
   - Units are explicitly confirmed (`timestamp`: s/ms/us/ns; `frequency`: Hz/kHz/MHz/GHz; etc.).
4. **Validation**: The user triggers "VALIDATE DATASET". The backend executes `load_dataset()`, runs schema bound checks, detects ambiguities, and returns a detailed `DatasetValidationReport`.
5. **Replay Configuration**: If validation passes (`VALIDATED (PASS)`), the run configuration panel appears, allowing setting Seed, Steps, Dwell Time (s), and Initial Band.
6. **Execution & Metric Presentation**: Clicking "RUN CASS-EW ON DATASET" invokes `DatasetRuntimeRunner`. Audited metrics (Observation Opportunities, Hits, Misses, False Alarms, Receiver Pd, Pfa, Scan Efficiency, Global Emitter Interception Rate, Latency) are rendered.
7. **JSON Export**: Clicking "EXPORT JSON" downloads the authentic `DatasetRunResult` as a timestamped JSON file.

---

## 4. Files Created
1. `tests/test_phase6_dataset_ui.py` — Dedicated test suite containing 18 unit and integration tests verifying all UI workflow endpoints, validation, error handling, file upload, export, and synthetic mode isolation.
2. `docs/reports/phase6_dataset_ui_report.md` — This comprehensive implementation and verification report.

---

## 5. Files Modified
1. `web_app.py`:
   - Added `/api/dataset/upload` (secure file upload with extension validation and temporary storage).
   - Added `/api/dataset/inspect` (safe inspection of column headers and sample records).
   - Added `/api/dataset/validate` (runs `DatasetConfig` through `BaseDatasetAdapter` validation).
   - Added `/api/dataset/run` (executes `DatasetRuntimeRunner` with user configuration).
   - Added `/api/dataset/export` (exports pure JSON of the latest `DatasetRunResult`).
2. `frontend/templates/index.html`:
   - Added Tier 10: "USER DATASET IMPORT, VALIDATION & REPLAY WORKFLOW" panel with technical console styling.
   - Fixed unclosed table tags in Section 8 reference benchmark.
3. `frontend/static/app.js`:
   - Implemented event handlers for file upload, fixture inspection, mapping population, unit selection, validation execution, replay execution, metrics display, and JSON export.

---

## 6. Backend Core & Phase 5 Integrity
- **Backend Core**: `algorithms/`, `simulator/environment.py`, `simulator/receiver.py`, `config.py`, `main.py` remained 100% frozen and unmodified.
- **Phase 5 Runtime**: `data/dataset_runtime.py` remained 100% unmodified; `web_app.py` interfaces with it strictly as an external client.

---

## 7. Security & File Handling
- File uploads are validated against an allowlist of extensions (`.csv`, `.json`, `.h5`, `.hdf5`).
- Filenames are sanitized using `werkzeug.utils.secure_filename` to prevent path traversal attacks.
- Uploaded files are stored in a controlled temporary directory (`tempfile.gettempdir()/cass_ew_uploads`).
- No arbitrary remote execution, arbitrary network fetching, or persistent database mutation.

---

## 8. No-Fake-Data Audit
The implementation strictly enforces truthfulness:
- When ground truth is absent (e.g. unannotated dataset), `Global Emitter Interception Rate` and `Latency` are explicitly displayed as `N/A — NO GROUND TRUTH`, never fabricated as 0.0% or 100.0%.
- When observation opportunities are 0, Receiver Pd is displayed as `N/A (0 Opportunities)`, never fabricated as 0.0% or 1.0.
- When quiet scans are 0, Receiver Pfa is displayed as `N/A (0 Quiet Scans)`.
- Decisions, scan counts, and hits accurately reflect backend replay results.

---

## 9. Synthetic Mode Regression Verification
- The existing synthetic simulation mode (`/api/reset`, `/api/step`, live cognitive status, spatial map, multi-receiver display) continues to work completely intact.
- Operating mode is clearly distinguished between `MODE: SYNTHETIC SIMULATION` and `MODE: DATASET REPLAY`.
- Running dataset replay does not corrupt or overwrite the live synthetic simulation state.

---

## 10. Test Suite Results
- **Previous Tests (Phase 5):** 324 passing.
- **New Phase 6 Tests:** 18 passing (`tests/test_phase6_dataset_ui.py`).
- **Total Test Suite:** 342 passing (`python -m unittest discover tests -v`), 0 failures, 0 errors.
- **TSRD Smoke Benchmark:** PASS (`python -m evaluation.tsrd_benchmark --steps 20`).

---

## 11. Known Limitations
1. Large multi-gigabyte HDF5 files may take several seconds to stream into memory during replay; memory usage scales with file record count.
2. In-browser upload is tailored for demonstration files up to typical demonstration limits (tested up to standard operational fixture sizes).
3. Emitter onset latency evaluation requires ground truth timestamps; in blind non-annotated datasets, this metric is omitted by design.

---

## 12. What Phase 6 Does NOT Claim
- Does not claim universal automatic mapping for unformatted binary files; user must confirm or specify mapping and units for unknown schemas.
- Does not claim real-world operational electronic warfare performance; it provides a validated interactive user console for replaying datasets through the cognitive scheduler.
