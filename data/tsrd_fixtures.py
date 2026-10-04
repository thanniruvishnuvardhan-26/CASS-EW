"""
TSRD Fixture Generator
CASS-EW SIH Problem Statement 26055

Generates verified high-fidelity TSRD-compliant HDF5, JSON, and CSV test fixtures
matching the exact verified schema:
- /data (shape N x 5: ToA [us], Frequency [MHz], PulseWidth [us], AoA [deg], Amplitude [dBm])
- /labels (shape N: emitter int ID)
- /metadata/feature_names (['ToA', 'Frequency', 'PulseWidth', 'AoA', 'Amplitude'])
- /metadata/receiver (attributes & datasets matching TSRD receiver parameters)
- /metadata/transmitters (transmitter groups with PRI, frequency, and scan configs)

Both STARE (continuous broad coverage) and SCAN (stepping receiver) fixtures
are supported.
"""

import h5py
import numpy as np
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List


def generate_tsrd_h5(
    file_path: str,
    num_pulses: int = 1500,
    scan_mode: str = "Staring",
    seed: int = 42,
    num_emitters: int = 4,
    collection_time_s: float = 1.0,
    inject_nan: bool = False,
    inject_non_monotonic: bool = False
) -> str:
    """
    Creates an HDF5 file with the exact TSRD schema.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Generate pulse data
    # ToA in microseconds (0 to collection_time_s * 1e6 us)
    max_toa_us = collection_time_s * 1e6
    toas = np.sort(rng.uniform(100.0, max_toa_us, size=num_pulses)).astype(np.float32)
    if inject_non_monotonic and len(toas) > 10:
        toas[5] = toas[2] # Break monotonicity

    # Emitter assignments
    labels = rng.integers(0, num_emitters, size=num_pulses, dtype=np.int8)

    # Carrier frequencies per emitter in MHz
    emitter_base_freqs = [1200.0, 2400.0, 5800.0, 9300.0, 14200.0][:num_emitters]
    freqs = np.zeros(num_pulses, dtype=np.float32)
    pws = np.zeros(num_pulses, dtype=np.float32)
    aoas = np.zeros(num_pulses, dtype=np.float32)
    amps = np.zeros(num_pulses, dtype=np.float32)

    for i in range(num_pulses):
        e_id = labels[i]
        base_f = emitter_base_freqs[e_id % len(emitter_base_freqs)]
        freqs[i] = float(base_f + rng.normal(0, 0.5))
        pws[i] = float(rng.uniform(0.5, 20.0))
        aoas[i] = float(rng.uniform(-180.0, 180.0))
        amps[i] = float(rng.uniform(-110.0, -30.0))

    if inject_nan and len(freqs) > 0:
        freqs[0] = np.nan

    data_matrix = np.column_stack([toas, freqs, pws, aoas, amps]).astype(np.float32)

    with h5py.File(path, 'w') as f:
        # 1. /data
        f.create_dataset('/data', data=data_matrix, dtype=np.float32)

        # 2. /labels
        f.create_dataset('/labels', data=labels, dtype=np.int8)

        # 3. /metadata
        meta_grp = f.create_group('/metadata')
        feature_names = [np.bytes_(s) for s in ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"]]
        meta_grp.create_dataset('feature_names', data=feature_names)

        # 4. /metadata/receiver
        rx_grp = meta_grp.create_group('receiver')
        rx_grp.attrs['bandwith_mhz'] = 500.0
        rx_grp.attrs['collection_time_s'] = float(collection_time_s)
        rx_grp.attrs['freq_noise_scale_mhz'] = 0.5
        rx_grp.attrs['gain_db'] = 10.0
        rx_grp.attrs['pw_noise_scale_us'] = 0.005
        rx_grp.attrs['pw_res_us'] = 0.0069
        rx_grp.attrs['scan_mode'] = np.bytes_(scan_mode)
        rx_grp.attrs['sensitivity_dbm'] = -110.0
        rx_grp.attrs['speed_km_s'] = 0.0
        rx_grp.attrs['toa_noise_scale_us'] = 0.025
        rx_grp.attrs['travel_angle_deg'] = 45.0

        dwell_centres = np.array([250.0, 750.0, 1250.0, 1750.0, 2250.0, 2750.0, 3250.0, 3750.0, 4250.0, 4750.0], dtype=np.float32)
        rx_grp.create_dataset('dwell_centres_mhz', data=dwell_centres)

        dwell_times = np.array([0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05], dtype=np.float32)
        rx_grp.create_dataset('dwell_times_s', data=dwell_times)

        rx_grp.create_dataset('freq_range_mhz', data=np.array([500.0, 18000.0], dtype=np.float32))
        rx_grp.create_dataset('start_position_km', data=np.array([100.0, 100.0], dtype=np.float32))

        # 5. /metadata/transmitters
        tx_grp = meta_grp.create_group('transmitters')
        for e_id in range(num_emitters):
            tx_sub = tx_grp.create_group(f'transmitter_{e_id}')
            freq_cfg = tx_sub.create_group('frequency_config')
            freq_cfg.create_dataset('freqs_mhz', data=np.array([emitter_base_freqs[e_id % len(emitter_base_freqs)]], dtype=np.float32))

            pos_cfg = tx_sub.create_group('position_config')
            pos_cfg.create_dataset('start_position_km', data=np.array([50.0 + e_id * 10, 50.0 - e_id * 5], dtype=np.float32))

            tx_sub.create_group('power_config')
            pri_cfg = tx_sub.create_group('pri_config')
            pri_cfg.create_dataset('pris_us', data=np.array([1000.0 + e_id * 500.0], dtype=np.float32))

            pw_cfg = tx_sub.create_group('pulse_width_config')
            pw_cfg.create_dataset('pws_us', data=np.array([1.0 + e_id * 0.5], dtype=np.float32))

            tx_sub.create_group('scan_config')

    return str(path)


def generate_tsrd_json(file_path: str, num_pulses: int = 100, seed: int = 42) -> str:
    """Generates a JSON fixture matching CASS-EW normalized PDW contract."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    records = []
    current_time = 0.001
    for i in range(num_pulses):
        current_time += float(rng.uniform(0.0001, 0.002))
        records.append({
            "timestamp": round(current_time, 6),
            "frequency": round(float(rng.choice([1200.0, 2400.0, 5800.0, 9300.0])), 2),
            "pulse_width": round(float(rng.uniform(0.5, 10.0)), 3),
            "amplitude": round(float(rng.uniform(-95.0, -40.0)), 2),
            "angle": round(float(rng.uniform(-180.0, 180.0)), 1),
            "receiver_id": "R1"
        })

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    return str(path)


def generate_tsrd_csv(file_path: str, num_pulses: int = 100, seed: int = 42) -> str:
    """Generates a CSV fixture matching CASS-EW normalized PDW contract."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    fieldnames = ["timestamp", "frequency", "pulse_width", "amplitude", "angle", "receiver_id"]
    current_time = 0.001

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(num_pulses):
            current_time += float(rng.uniform(0.0001, 0.002))
            writer.writerow({
                "timestamp": f"{current_time:.6f}",
                "frequency": f"{float(rng.choice([1200.0, 2400.0, 5800.0, 9300.0])):.2f}",
                "pulse_width": f"{float(rng.uniform(0.5, 10.0)):.3f}",
                "amplitude": f"{float(rng.uniform(-95.0, -40.0)):.2f}",
                "angle": f"{float(rng.uniform(-180.0, 180.0)):.1f}",
                "receiver_id": "R1"
            })
    return str(path)


if __name__ == "__main__":
    h5_stare = generate_tsrd_h5("data/tsrd_fixtures/sample_stare.h5", scan_mode="Staring", seed=42)
    h5_scan = generate_tsrd_h5("data/tsrd_fixtures/sample_scan.h5", scan_mode="Scanning", seed=43)
    json_path = generate_tsrd_json("data/tsrd_fixtures/sample_pdws.json")
    csv_path = generate_tsrd_csv("data/tsrd_fixtures/sample_pdws.csv")
    print(f"Generated fixtures:\n - {h5_stare}\n - {h5_scan}\n - {json_path}\n - {csv_path}")
