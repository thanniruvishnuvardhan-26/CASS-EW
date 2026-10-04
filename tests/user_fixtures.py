"""
Test Fixtures Generator for General User Datasets
CASS-EW SIH Problem Statement 26055

Generates deterministic test fixtures for Phase 2 & Phase 3:
1. Canonical CSV format (standard headers)
2. Alternate-column CSV format (custom headers e.g. ToA, freq_mhz, PW, AoA, Power_dBm)
3. Canonical JSON format (list of dicts)
4. Generic HDF5 format
5. Malformed / Invalid dataset (non-numeric, missing headers, NaN)
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import h5py
except ImportError:
    h5py = None

import numpy as np


def generate_canonical_csv(file_path: str, num_pulses: int = 50, seed: int = 101) -> str:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    fieldnames = ["timestamp", "frequency", "pulse_width", "angle", "amplitude", "emitter_id", "receiver_id"]
    t = 0.001

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(num_pulses):
            t += float(rng.uniform(0.0005, 0.002))
            writer.writerow({
                "timestamp": f"{t:.6f}",
                "frequency": f"{float(rng.choice([1500.0, 3200.0, 6000.0, 9500.0])):.2f}",
                "pulse_width": f"{float(rng.uniform(0.8, 5.0)):.3f}",
                "angle": f"{float(rng.uniform(-180.0, 180.0)):.1f}",
                "amplitude": f"{float(rng.uniform(-80.0, -30.0)):.2f}",
                "emitter_id": int(rng.integers(1, 4)),
                "receiver_id": "RX_0"
            })
    return str(path)


def generate_alternate_column_csv(file_path: str, num_pulses: int = 50, out_of_order: bool = True, seed: int = 102) -> str:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    fieldnames = ["ToA_ms", "carrier_freq_ghz", "PW_ns", "AoA_deg", "Power_dBm", "Emitter_Code"]
    timestamps_ms = []
    t = 1.0
    for i in range(num_pulses):
        t += float(rng.uniform(0.5, 2.0))
        timestamps_ms.append(t)

    if out_of_order:
        # Shuffle timestamps to test automatic chronological sorting
        rng.shuffle(timestamps_ms)

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, t_ms in enumerate(timestamps_ms):
            writer.writerow({
                "ToA_ms": f"{t_ms:.4f}",
                "carrier_freq_ghz": f"{float(rng.choice([1.5, 3.2, 6.0, 9.5])):.3f}",
                "PW_ns": f"{float(rng.uniform(800.0, 4000.0)):.1f}",
                "AoA_deg": f"{float(rng.uniform(-180.0, 180.0)):.1f}",
                "Power_dBm": f"{float(rng.uniform(-80.0, -30.0)):.2f}",
                "Emitter_Code": f"RADAR_{int(rng.integers(1, 4))}"
            })
    return str(path)


def generate_canonical_json(file_path: str, num_pulses: int = 50, seed: int = 103) -> str:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    records = []
    t = 0.002
    for i in range(num_pulses):
        t += float(rng.uniform(0.0005, 0.002))
        records.append({
            "timestamp": round(t, 6),
            "frequency": round(float(rng.choice([2100.0, 4800.0, 8400.0])), 2),
            "pulse_width": round(float(rng.uniform(1.0, 4.0)), 3),
            "angle": round(float(rng.uniform(-90.0, 90.0)), 1),
            "amplitude": round(float(rng.uniform(-75.0, -40.0)), 2),
            "emitter_id": int(rng.integers(1, 3)),
            "receiver_id": "RX_0"
        })

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    return str(path)


def generate_generic_hdf5(file_path: str, num_pulses: int = 100, seed: int = 104) -> str:
    if h5py is None:
        raise ImportError("h5py required to generate HDF5 test fixture.")

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    toas_us = np.sort(rng.uniform(100.0, 1000000.0, size=num_pulses)).astype(np.float64)
    freqs_mhz = rng.choice([1200.0, 3600.0, 7200.0], size=num_pulses).astype(np.float64)
    pws_us = rng.uniform(0.5, 3.0, size=num_pulses).astype(np.float64)
    aoas_deg = rng.uniform(-180.0, 180.0, size=num_pulses).astype(np.float64)
    amps_dbm = rng.uniform(-85.0, -35.0, size=num_pulses).astype(np.float64)
    labels = rng.integers(1, 4, size=num_pulses, dtype=np.int32)

    matrix = np.column_stack([toas_us, freqs_mhz, pws_us, aoas_deg, amps_dbm])

    with h5py.File(path, 'w') as f:
        f.create_dataset("/pulse_data", data=matrix)
        f.create_dataset("/target_labels", data=labels)

    return str(path)


def generate_malformed_csv(file_path: str) -> str:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "frequency", "pulse_width", "amplitude"])
        # Row 1: Valid
        writer.writerow(["0.001", "2400.0", "1.2", "-50.0"])
        # Row 2: Corrupted non-numeric timestamp
        writer.writerow(["CORRUPTED_TOA", "3600.0", "1.0", "-60.0"])
        # Row 3: NaN frequency
        writer.writerow(["0.003", "NaN", "1.5", "-45.0"])
        # Row 4: Valid
        writer.writerow(["0.004", "5800.0", "2.0", "-55.0"])

    return str(path)
