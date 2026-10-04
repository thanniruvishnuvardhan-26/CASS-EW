# External Synthetic PDW Dataset — HDF5 Adapter Audit

## 1. Dataset Schema Observed
The external HDF5 dataset structure adheres to the following required schema:
- `/data`: Two-dimensional array (N records x 5 features) containing floating-point data.
- `/labels`: Array of length N containing emitter/ground-truth labels.
- `/metadata/feature_names`: Array containing exact names `["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"]`.
- `/metadata/receiver`: Group or dataset containing receiver-specific parameters (e.g., position, sensitivity).
- `/metadata/transmitters`: Group or dataset containing transmitter-specific parameters (e.g., emitter count).

## 2. Adapter Design
The `H5PDWAdapter` (located at `data/h5_pdws.py`) serves as a strict, read-only interface to the HDF5 files. It forces validation at initialization, preventing any malformed datasets or schema violations from progressing. It separates observables from labels using separate explicit iterators.

## 3. Observable Fields
The scheduler interface (`iter_observations()`) yields `PDWRecord` dataclass instances. These instances contain:
- `timestamp`
- `frequency`
- `pulse_width`
- `angle_of_arrival`
- `amplitude`

## 4. Ground-Truth Fields
The ground truth (`iter_ground_truth()`) yields raw `label` data independent of the observable records.

## 5. Leakage Boundary
**STRICT ENFORCEMENT:** `iter_observations()` returns `PDWRecord`s that intrinsically *cannot* contain ground truth labels. This safely partitions ground truth behind `iter_ground_truth()`, mathematically preventing the scheduler from observing future records or labels. Leakage tests confirm `PDWRecord`s do not possess `label` or `transmitter_id` attributes.

## 6. Chronological Handling
Chronological order is strictly validated during file initialization. The adapter verifies that `ToA` is monotonically increasing. The adapter does not artificially sort the dataset; if ToA ordering is violated, it throws a `ValueError`, mandating correct data ingestion. 

## 7. Intentional Non-Inferences
This phase acts strictly as a data-layer passthrough:
- **No Units Guessed:** Raw numerical values are preserved entirely without conversion.
- **No Band Mapping:** Frequency boundaries and CASS-EW band indices are not inferred.
- **No Detection Mapping:** Labels remain distinct ground-truth and do not map to simulated detections yet.
- **No Spatial Semantics:** `AoA` values and Receiver metadata remain separated without imposing geometric meaning.

## 8. config_0.h5 Smoke-Test Result
The actual real-world `config_0.h5` dataset was successfully located at `C:\Anti Gravity\Datasets\scan\train_scan\config_0.h5`.
The `H5PDWAdapter` successfully ingested the real file with the following exact metadata:
- **Records:** 169,617
- **Unique Labels:** 72
- **Transmitter Count:** 96 (configured in metadata)
- **Features:** Exactly `['ToA', 'Frequency', 'PulseWidth', 'AoA', 'Amplitude']`
- **ToA Monotonicity:** Mathematically verified (True)
- **ToA Range:** `200142.53` to `29195164.0`
- **Frequency Range:** `10.36` to `10999.57`
- **Pulse Width Range:** `0.0069` to `346.27`
- **AoA Range:** `-179.99` to `179.98`
- **Amplitude Range:** `-170.47` to `-1.25`

The file schema precisely matches the expectations implemented in the adapter. Ground truth separation properly isolated arrays holding index IDs (e.g., `[30]`), preventing any semantic linkage to observational data records.

## 9. Test Results
- **New Tests:** 14 rigorous isolation and validation tests created (`tests/test_h5_pdws.py`).
- **Regression Tests:** Maintained 163/163 passing unit tests prior to this phase.
- **Total Suite:** 177 tests currently passing (0 failures).

## 10. Remaining Semantic Questions
Before proceeding to replay integration, the following semantics must be clarified:
- What are the units for ToA, Frequency, PulseWidth, AoA, and Amplitude?
- How should Frequency be mapped to the existing `0..N` CASS-EW discrete receiver bands?
- How should existing spatial algorithms interpret AoA and abstract receiver coordinates?
- How is the `label` mapping transformed to existing `detection` semantics during evaluation?
