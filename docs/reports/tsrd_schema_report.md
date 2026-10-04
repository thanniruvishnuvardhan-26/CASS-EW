# TSRD Schema Audit & Validation Report
**Project:** CASS-EW — Cognitive Adaptive Smart Scan for Electronic Warfare (SIH 2026 PS 26055)  
**Dataset:** Alan Turing Institute — Turing Synthetic Radar Dataset (TSRD)  
**Format:** HDF5 (Hierarchical Data Format 5)  

---

## 1. Verified HDF5 Schema Layout

Every compliant TSRD file provides the following hierarchical datasets and metadata groups:

```text
/
├── data                              [Dataset: shape=(N, 5), dtype=float32]
│                                      Columns: [ToA, Frequency, PulseWidth, AoA, Amplitude]
├── labels                            [Dataset: shape=(N,), dtype=int8 or int32]
│                                      Values: Emitter class identifier per pulse
└── metadata                          [Group]
    ├── feature_names                 [Dataset: shape=(5,), dtype=|S10 or string]
    │                                  Values: ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"]
    ├── receiver                      [Group or Dataset with attributes]
    │   ├── attributes:
    │   │   ├── bandwith_mhz          e.g. 500.0 MHz
    │   │   ├── collection_time_s     e.g. 30.0 s
    │   │   ├── freq_noise_scale_mhz  e.g. 0.5 MHz
    │   │   ├── gain_db               e.g. 10.0 dB
    │   │   ├── pw_noise_scale_us     e.g. 0.005 μs
    │   │   ├── pw_res_us             e.g. 0.0069 μs
    │   │   ├── scan_mode             'Staring' or 'Scanning'
    │   │   ├── sensitivity_dbm       e.g. -110.0 dBm
    │   │   ├── speed_km_s            e.g. 0.0 km/s
    │   │   ├── toa_noise_scale_us    e.g. 0.025 μs
    │   │   └── travel_angle_deg      e.g. 45.0 deg
    │   ├── dwell_centres_mhz         [Dataset: shape=(M,), dtype=float32]
    │   ├── dwell_times_s             [Dataset: shape=(M,), dtype=float32]
    │   ├── freq_range_mhz            [Dataset: shape=(2,), dtype=float32] ([500.0, 18000.0])
    │   └── start_position_km         [Dataset: shape=(2,), dtype=float32] ([100.0, 100.0])
    └── transmitters                  [Group]
        ├── transmitter_0 ... transmitter_K
            ├── frequency_config      (freqs_mhz, freq_mode)
            ├── position_config       (start_position_km, speed_km_s, travel_angle_deg)
            ├── power_config          (power_w, gain)
            ├── pri_config            (pris_us, pri_mode)
            ├── pulse_width_config    (pws_us, pw_mode)
            └── scan_config           (scan_rate_rpm, beam_width_deg, scan_start_angle)
```

---

## 2. Physical Feature Units & Semantics

| Feature | TSRD Representation | Units | Internal CASS-EW Representation | Normalization Factor |
|:---|:---|:---|:---|:---|
| **ToA** | Time of Arrival | Microseconds ($\mu\text{s}$) | Seconds ($\text{s}$) | $t_{\text{sec}} = t_{\mu\text{s}} / 10^6$ |
| **Frequency** | Carrier Center Frequency | Megahertz ($\text{MHz}$) | Physical Megahertz & Discrete Band $[0..B-1]$ | Contiguous `BandMap` Boundaries |
| **PulseWidth** | Pulse Duration | Microseconds ($\mu\text{s}$) | Microseconds ($\mu\text{s}$) | Direct preservation |
| **AoA** | Angle of Arrival | Degrees ($-180^\circ$ to $+180^\circ$) | Degrees ($-180^\circ$ to $+180^\circ$) | Direct preservation |
| **Amplitude** | Received Power | Decibels relative to 1 mW ($\text{dBm}$) | Decibels relative to 1 mW ($\text{dBm}$) | Direct preservation |

---

## 3. Strict Schema Validation Criteria

The `TSRDAdapter` enforces the following validation criteria prior to opening any stream:
1. **Existence:** File exists and is readable via `h5py`.
2. **Mandatory Paths:** `/data`, `/labels`, `/metadata`, `/metadata/feature_names` must exist.
3. **Dimensions:** `/data` shape must be strictly $(N, 5)$; `/labels` shape must be strictly $(N,)$.
4. **Finite Values:** Zero tolerance for `NaN` or `Inf` in `/data`.
5. **Monotonicity:** ToA timestamps must be monotonically non-decreasing ($\Delta \text{ToA} \ge 0$).
6. **Quarantine:** Emitter labels and transmitter metadata are strictly partitioned into `iter_ground_truth()` and never exposed through `iter_observations()`.
