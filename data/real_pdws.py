"""
CASS-EW Phase 9: Real RF / PDW Data Adapter

This module provides a clean adapter interface for ingesting external
Pulse Descriptor Word (PDW) or RF observation records into the CASS-EW
observation contract. It supports:

- PDW schema validation
- Observation normalization  
- Timestamp-ordered deterministic replay
- Causality enforcement (no future data exposure)
- Validation reporting (accepted/rejected/corrected counts)

The scheduler must not distinguish whether an observation came from
a synthetic environment, replay fixture, or authorized real dataset.

PDW Schema (all fields optional except timestamp and frequency):
    timestamp       : float   — time of observation (seconds)
    frequency       : float   — center frequency (Hz or band index)
    bandwidth       : float   — signal bandwidth (Hz), optional
    pulse_width     : float   — pulse duration (seconds), optional
    amplitude       : float   — signal amplitude (dB), optional
    receiver_id     : str     — receiver identifier, optional
    angle           : float   — angle of arrival (degrees), optional
    detection_result: bool    — whether signal was detected, optional

Signal Model:
    formula:    SNR = base_snr / (1 + distance)
    assumptions: free-space monotonic attenuation, synthetic only
    units:      arbitrary SNR units
    noise:      none (deterministic attenuation)
"""

import json
import csv
import copy
from typing import List, Dict, Any, Optional, Tuple


# ---- PDW Schema Definition ----

PDW_SCHEMA = {
    "required": ["timestamp", "frequency"],
    "optional": ["bandwidth", "pulse_width", "amplitude", "receiver_id",
                 "angle", "detection_result"],
    "types": {
        "timestamp": (int, float),
        "frequency": (int, float),
        "bandwidth": (int, float),
        "pulse_width": (int, float),
        "amplitude": (int, float),
        "receiver_id": (str,),
        "angle": (int, float),
        "detection_result": (bool, int),
    }
}


class PDWValidationReport:
    """Tracks validation statistics for a PDW dataset."""
    def __init__(self):
        self.accepted = 0
        self.rejected = 0
        self.corrected = 0
        self.rejection_reasons = []

    def accept(self):
        self.accepted += 1

    def reject(self, reason: str, record_index: int):
        self.rejected += 1
        self.rejection_reasons.append((record_index, reason))

    def correct(self):
        self.corrected += 1
        self.accepted += 1

    def summary(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "corrected": self.corrected,
            "rejection_reasons": self.rejection_reasons[:20],  # cap for display
        }


class PDWRecord:
    """A validated PDW record."""
    def __init__(self, data: Dict[str, Any]):
        self.timestamp = data["timestamp"]
        self.frequency = data["frequency"]
        self.bandwidth = data.get("bandwidth")
        self.pulse_width = data.get("pulse_width")
        self.amplitude = data.get("amplitude")
        self.receiver_id = data.get("receiver_id", "R0")
        self.angle = data.get("angle")
        self.detection_result = data.get("detection_result", True)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "timestamp": self.timestamp,
            "frequency": self.frequency,
            "receiver_id": self.receiver_id,
            "detection_result": self.detection_result,
        }
        if self.bandwidth is not None:
            d["bandwidth"] = self.bandwidth
        if self.pulse_width is not None:
            d["pulse_width"] = self.pulse_width
        if self.amplitude is not None:
            d["amplitude"] = self.amplitude
        if self.angle is not None:
            d["angle"] = self.angle
        return d


def validate_pdw_record(record: Dict[str, Any], index: int, report: PDWValidationReport) -> Optional[PDWRecord]:
    """
    Validate a single PDW record against the schema.
    Returns PDWRecord if valid (possibly corrected), None if rejected.
    """
    # Check required fields
    for field in PDW_SCHEMA["required"]:
        if field not in record or record[field] is None:
            report.reject(f"Missing required field: {field}", index)
            return None

    # Type checking
    for field, expected_types in PDW_SCHEMA["types"].items():
        if field in record and record[field] is not None:
            if not isinstance(record[field], expected_types):
                report.reject(f"Invalid type for {field}: expected {expected_types}, got {type(record[field])}", index)
                return None

    # Value validation
    if record["timestamp"] < 0:
        report.reject(f"Negative timestamp: {record['timestamp']}", index)
        return None

    if isinstance(record["frequency"], (int, float)) and record["frequency"] < 0:
        report.reject(f"Negative frequency: {record['frequency']}", index)
        return None

    if "pulse_width" in record and record["pulse_width"] is not None:
        if record["pulse_width"] < 0:
            report.reject(f"Negative pulse_width: {record['pulse_width']}", index)
            return None

    if "bandwidth" in record and record["bandwidth"] is not None:
        if record["bandwidth"] < 0:
            report.reject(f"Negative bandwidth: {record['bandwidth']}", index)
            return None

    # Corrective: coerce detection_result int to bool
    corrected = False
    clean = dict(record)
    if "detection_result" in clean and isinstance(clean["detection_result"], int) and not isinstance(clean["detection_result"], bool):
        clean["detection_result"] = bool(clean["detection_result"])
        corrected = True

    if corrected:
        report.correct()
    else:
        report.accept()

    return PDWRecord(clean)


def validate_dataset(records: List[Dict[str, Any]]) -> Tuple[List[PDWRecord], PDWValidationReport]:
    """
    Validate an entire dataset of PDW records.
    Returns list of valid PDWRecords and a validation report.
    """
    report = PDWValidationReport()
    valid = []

    seen_timestamps = set()
    last_timestamp = -float('inf')

    for i, rec in enumerate(records):
        pdw = validate_pdw_record(rec, i, report)
        if pdw is None:
            continue

        # Check timestamp ordering
        if pdw.timestamp < last_timestamp:
            report.reject(f"Timestamp not in order: {pdw.timestamp} < {last_timestamp}", i)
            # We still accept but flag it — don't silently discard
            # Actually per spec: "Do not silently discard invalid records" — we report and skip
            continue

        # Duplicate detection (exact timestamp + frequency + receiver)
        dup_key = (pdw.timestamp, pdw.frequency, pdw.receiver_id)
        if dup_key in seen_timestamps:
            report.reject(f"Duplicate record at timestamp={pdw.timestamp}, freq={pdw.frequency}, rx={pdw.receiver_id}", i)
            continue

        seen_timestamps.add(dup_key)
        last_timestamp = pdw.timestamp
        valid.append(pdw)

    return valid, report


def normalize_to_observation(pdw: PDWRecord, band_mapping: Optional[Dict[float, int]] = None,
                              num_bands: int = 10) -> Dict[str, Any]:
    """
    Convert a PDWRecord into the CASS-EW observation contract format.
    This is the normalization boundary — the scheduler sees the same format
    regardless of data source.
    """
    # Map frequency to band
    if band_mapping and pdw.frequency in band_mapping:
        band = band_mapping[pdw.frequency]
    elif isinstance(pdw.frequency, int) and 0 <= pdw.frequency < num_bands:
        band = pdw.frequency
    elif isinstance(pdw.frequency, float) and pdw.frequency == int(pdw.frequency):
        band = int(pdw.frequency)
    else:
        band = int(pdw.frequency) % num_bands

    obs = {
        "observation_time": pdw.timestamp,
        "time": pdw.timestamp,
        "tuned_band": band,
        "band": band,
        "detection_result": pdw.detection_result,
        "detected": pdw.detection_result,
        "effective_duration": pdw.pulse_width if pdw.pulse_width else 1,
        "dwell_start_time": pdw.timestamp,
        "dwell_end_time": pdw.timestamp + (pdw.pulse_width if pdw.pulse_width else 1),
    }

    if pdw.amplitude is not None:
        obs["signal_strength"] = pdw.amplitude

    return obs


class PDWReplayEngine:
    """
    Deterministic replay engine that feeds PDW observations to a scheduler
    one at a time, respecting timestamp order.
    
    CAUSALITY: The scheduler receives ONLY the current observation and its
    accumulated history. No future records are exposed.
    """

    def __init__(self, validated_records: List[PDWRecord], scheduler,
                 band_mapping: Optional[Dict[float, int]] = None,
                 num_bands: int = 10):
        self.records = sorted(validated_records, key=lambda r: r.timestamp)
        self.scheduler = scheduler
        self.band_mapping = band_mapping
        self.num_bands = num_bands
        self.replay_index = 0
        self.action_log = []

    def reset(self, seed=None):
        self.replay_index = 0
        self.action_log = []
        if hasattr(self.scheduler, 'reset'):
            self.scheduler.reset(seed=seed)

    def step(self) -> Optional[Dict[str, Any]]:
        """
        Process next PDW record. Returns the observation dict or None if replay done.
        The scheduler only sees history up to and including the current record.
        """
        if self.replay_index >= len(self.records):
            return None

        pdw = self.records[self.replay_index]
        obs = normalize_to_observation(pdw, self.band_mapping, self.num_bands)

        # Get scheduler's action based on current observable state
        last_obs = self.action_log[-1]["observation"] if self.action_log else None
        action = self.scheduler.select_action(last_obs)

        # Update scheduler with the observation
        band = obs["tuned_band"]
        detected = obs["detection_result"]
        self.scheduler.update(obs, band, detected)

        self.action_log.append({
            "replay_index": self.replay_index,
            "observation": obs,
            "scheduler_action": action,
        })

        self.replay_index += 1
        return obs

    def run_full_replay(self) -> List[Dict[str, Any]]:
        """Run the entire replay and return the action log."""
        self.reset()
        while self.step() is not None:
            pass
        return self.action_log

    def get_replay_metrics(self) -> Dict[str, Any]:
        """Return summary metrics from the replay."""
        if not self.action_log:
            return {"total_records": 0}
        return {
            "total_records": len(self.records),
            "replayed_records": len(self.action_log),
            "valid_record_rate": len(self.action_log) / max(1, len(self.records)),
        }


def load_pdw_json(filepath: str) -> List[Dict[str, Any]]:
    """Load PDW records from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_pdw_csv(filepath: str) -> List[Dict[str, Any]]:
    """Load PDW records from a CSV file."""
    records = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {}
            for k, v in row.items():
                if v == '':
                    continue
                try:
                    if '.' in v:
                        rec[k] = float(v)
                    else:
                        rec[k] = int(v)
                except (ValueError, TypeError):
                    rec[k] = v
            records.append(rec)
    return records


def search_for_authorized_datasets(search_paths=None) -> Dict[str, Any]:
    """
    Search for authorized real RF/PDW datasets.
    Returns status and any found dataset information.
    """
    import os
    if search_paths is None:
        search_paths = [
            "data/",
            "dataset/",
            "datasets/",
            "real_data/",
        ]
    
    found_datasets = []
    for path in search_paths:
        if os.path.exists(path):
            for f in os.listdir(path):
                if f.endswith(('.json', '.csv', '.pdw', '.dat')):
                    full = os.path.join(path, f)
                    found_datasets.append({
                        "path": full,
                        "size_bytes": os.path.getsize(full),
                        "name": f,
                    })
    
    return {
        "authorized_dataset_found": len(found_datasets) > 0,
        "datasets": found_datasets,
        "status": "Authorized datasets found" if found_datasets else 
                  "No authorized real RF/PDW dataset was available for validation."
    }
