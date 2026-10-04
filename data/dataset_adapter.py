"""
Generalized Dataset Adapter Architecture & Canonical PDW Contract
CASS-EW SIH Problem Statement 26055 - Phase 2, 3, & 4

Provides:
- Explicit Unit Model & Deterministic Conversions:
    timestamp_unit:  s, ms, us, ns -> timestamp_s
    frequency_unit:  Hz, kHz, MHz, GHz -> frequency_mhz
    pulse_width_unit: s, ms, us, ns -> pulse_width_us
    aoa_unit:        deg -> aoa_deg
    amplitude_unit:  dBm, dB -> amplitude_dbm (preserves project convention)
- No Silent Unit Guessing:
    Strict error on unknown/unresolved units unless adapter contract guarantees standard defaults.
- Robust Field Mapping & Ambiguity Detection:
    Translates source dataset columns/keys to canonical logical fields:
    (timestamp, frequency, pulse_width, aoa, amplitude, receiver_id, emitter_label).
    Flags multiple matching candidate columns as AMBIGUOUS FIELD MAPPING.
- Clean DatasetConfig object usable across CSV, JSON, HDF5 without scheduler alteration.
- Extended DatasetValidationReport detailing mapping status, conversions, unresolved/ambiguous fields.
"""

import os
import json
import csv
import math
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Iterator, Tuple, List, Dict, Any, Optional, Union

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from data.time_normalization import TimeNormalizer


# =====================================================================
# CANONICAL PDW REPRESENTATION (PHASE 2)
# =====================================================================

@dataclass(frozen=True)
class ObservablePDW:
    """
    Scheduler-visible normalized canonical PDW.
    CRITICAL: Contains NO emitter label or ground-truth future metadata.
    """
    index: int
    timestamp_us: float
    timestamp_s: float
    frequency_mhz: float
    pulse_width_us: float
    angle_of_arrival_deg: float
    amplitude_dbm: float
    receiver_id: str = "RX_0"

    @property
    def aoa_deg(self) -> float:
        """Alias for angle_of_arrival_deg."""
        return self.angle_of_arrival_deg

    @property
    def amplitude_db(self) -> float:
        """Alias for amplitude_dbm."""
        return self.amplitude_dbm


@dataclass(frozen=True)
class GroundTruthPDW:
    """
    Evaluator-only ground truth.
    CRITICAL: NEVER exposed to scheduler or virtual receiver decision logic.
    """
    index: int
    emitter_label: Any
    transmitter_id: Optional[str] = None


# =====================================================================
# EXPLICIT UNIT MODEL & DETERMINISTIC CONVERSION (PHASE 4)
# =====================================================================

# Supported explicit units
SUPPORTED_UNITS = {
    "timestamp": {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9},
    "frequency": {"hz": 1e-6, "khz": 1e-3, "mhz": 1.0, "ghz": 1e3},
    "pulse_width": {"s": 1e6, "ms": 1e3, "us": 1.0, "ns": 1e-3},
    "aoa": {"deg": 1.0, "degrees": 1.0},
    "amplitude": {"dbm": 1.0, "db": 1.0}
}

CANONICAL_TARGET_UNITS = {
    "timestamp": "s",
    "frequency": "MHz",
    "pulse_width": "us",
    "aoa": "deg",
    "amplitude": "dBm"
}

class UnitConverter:
    """Deterministic, explicit unit conversion system for CASS-EW."""

    @classmethod
    def get_scale_factor(cls, dimension: str, input_unit: str) -> float:
        dim = dimension.lower()
        if dim not in SUPPORTED_UNITS:
            raise ValueError(f"Unknown dimension '{dimension}'. Supported: {list(SUPPORTED_UNITS.keys())}")
        unit = input_unit.strip().lower()
        if unit not in SUPPORTED_UNITS[dim]:
            valid = list(SUPPORTED_UNITS[dim].keys())
            raise ValueError(f"Unsupported unit '{input_unit}' for {dimension}. Supported: {valid}")
        return SUPPORTED_UNITS[dim][unit]

    @classmethod
    def convert(cls, value: float, dimension: str, input_unit: str) -> float:
        scale = cls.get_scale_factor(dimension, input_unit)
        return float(value) * scale


# =====================================================================
# DATASET CONFIGURATION OBJECT (PHASE 4)
# =====================================================================

@dataclass
class DatasetConfig:
    """
    Unified dataset ingestion configuration for CASS-EW.
    Allows specifying explicit column/field mappings and input units without editing Python code.
    """
    field_mapping: Dict[str, str] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)
    coverage_freq_range_mhz: Optional[Tuple[float, float]] = None
    receiver_id: str = "RX_0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# VALIDATION LAYER & REPORTING (PHASE 2 & 4)
# =====================================================================

@dataclass
class DatasetValidationReport:
    """Detailed validation and audit report for an ingested dataset."""
    source_path: str
    format: str
    record_count: int
    label_count: int
    feature_names: List[str]
    units: Dict[str, str]
    time_range_us: Tuple[float, float]
    time_range_s: Tuple[float, float]
    frequency_range_mhz: Tuple[float, float]
    pulse_width_range_us: Tuple[float, float]
    aoa_range_deg: Tuple[float, float]
    amplitude_range_dbm: Tuple[float, float]
    receiver_metadata: Dict[str, Any]
    transmitter_count: int
    scan_mode: str
    causal_suitability: str
    is_valid: bool
    warnings: List[str]
    validation_errors: List[str]
    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    defaults_used: Dict[str, Any] = field(default_factory=dict)
    source_field_mapping: Dict[str, str] = field(default_factory=dict)
    resolved_field_mapping: Dict[str, str] = field(default_factory=dict)
    input_units: Dict[str, str] = field(default_factory=dict)
    canonical_units: Dict[str, str] = field(default_factory=dict)
    conversion_summary: Dict[str, str] = field(default_factory=dict)
    unresolved_fields: List[str] = field(default_factory=list)
    ambiguous_fields: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.total_rows == 0 and self.record_count > 0:
            self.total_rows = self.record_count
        if self.valid_rows == 0 and self.is_valid:
            self.valid_rows = self.record_count
        if self.invalid_rows == 0 and not self.is_valid and self.total_rows > 0:
            self.invalid_rows = self.total_rows - self.valid_rows
        if not self.canonical_units:
            self.canonical_units = dict(CANONICAL_TARGET_UNITS)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text_summary(self) -> str:
        status_str = "VALID" if self.is_valid else "INVALID"
        summary = (
            f"=== DATASET VALIDATION REPORT ({status_str}) ===\n"
            f"Source: {self.source_path}\n"
            f"Format: {self.format.upper()} | Scan Mode: {self.scan_mode}\n"
            f"Total Rows: {self.total_rows} | Valid Rows: {self.valid_rows} | Invalid Rows: {self.invalid_rows}\n"
            f"Records Indexed: {self.record_count} | Unique Emitters: {self.label_count}\n"
            f"Causal Suitability: {self.causal_suitability}\n"
            f"Time Range (s): [{self.time_range_s[0]:.4f}, {self.time_range_s[1]:.4f}] ({self.time_range_us[1] - self.time_range_us[0]:.1f} us)\n"
            f"Freq Range (MHz): [{self.frequency_range_mhz[0]:.2f}, {self.frequency_range_mhz[1]:.2f}]\n"
            f"PW Range (us): [{self.pulse_width_range_us[0]:.3f}, {self.pulse_width_range_us[1]:.3f}]\n"
            f"AoA Range (deg): [{self.aoa_range_deg[0]:.1f}, {self.aoa_range_deg[1]:.1f}]\n"
            f"Amplitude Range (dBm): [{self.amplitude_range_dbm[0]:.1f}, {self.amplitude_range_dbm[1]:.1f}]\n"
        )
        if self.resolved_field_mapping:
            summary += f"Field Mapping: {self.resolved_field_mapping}\n"
        if self.conversion_summary:
            summary += f"Units Converted: {self.conversion_summary}\n"
        if self.ambiguous_fields:
            summary += f"Ambiguous Fields: {self.ambiguous_fields}\n"
        if self.unresolved_fields:
            summary += f"Unresolved Fields: {self.unresolved_fields}\n"
        if self.defaults_used:
            summary += f"Defaults Applied: {self.defaults_used}\n"
        summary += f"Warnings: {len(self.warnings)} | Errors: {len(self.validation_errors)}\n"
        return summary


# Standard documented fallback defaults for optional attributes
DEFAULT_PULSE_WIDTH_US = 1.0
DEFAULT_AOA_DEG = 0.0
DEFAULT_AMPLITUDE_DBM = -60.0
DEFAULT_RECEIVER_ID = "RX_0"
DEFAULT_EMITTER_LABEL = 0


# =====================================================================
# TABLE / RECORD NORMALIZER HELPER (PHASE 4)
# =====================================================================

class _RecordTableNormalizer:
    """
    Normalizes tabular and dictionary records with explicit field mappings,
    ambiguity detection, explicit unit scaling, missing-field defaults,
    chronological sorting, and full validation.
    """

    # Common canonical alias candidates (case-insensitive)
    ALIAS_CANDIDATES = {
        "timestamp": ["timestamp", "toa", "time", "time_s", "timestamp_s", "toa_s", "timestamp_us", "toa_us", "arrival_time"],
        "frequency": ["frequency", "freq", "frequency_mhz", "freq_mhz", "carrier_freq", "center_freq", "rf_mhz", "freq_ghz", "frequency_ghz"],
        "pulse_width": ["pulse_width", "pw", "pulsewidth", "pulse_width_us", "pw_us", "width", "duration"],
        "aoa": ["aoa", "angle", "angle_of_arrival", "aoa_deg", "bearing", "azimuth"],
        "amplitude": ["amplitude", "power", "amplitude_dbm", "power_dbm", "level", "rssi", "amp", "signal_strength"],
        "emitter_label": ["emitter_label", "label", "emitter_id", "emitter", "tx_id", "source", "class", "target"],
        "receiver_id": ["receiver_id", "rx_id", "receiver", "sensor"]
    }

    @classmethod
    def resolve_field_mapping(
        cls,
        available_keys: List[str],
        user_mapping: Dict[str, str]
    ) -> Tuple[Dict[str, str], List[str], List[str]]:
        """
        Resolves mapping from logical canonical fields to dataset columns/keys.
        Returns:
            resolved: Dict[canonical_field, source_col]
            ambiguous: List of canonical fields that matched multiple candidate columns
            unresolved: List of canonical fields that could not be found
        """
        resolved: Dict[str, str] = {}
        ambiguous: List[str] = []
        unresolved: List[str] = []

        avail_clean = [k.strip() for k in available_keys]
        avail_lower_map: Dict[str, str] = {k.lower(): k for k in avail_clean}

        # 1. First process user-explicit mappings
        claimed_sources = set()
        for canonical_key, user_target in user_mapping.items():
            canonical_key_clean = canonical_key.strip().lower()
            target_clean = user_target.strip()
            # Match exact or case-insensitive
            matched = None
            if target_clean in avail_clean:
                matched = target_clean
            elif target_clean.lower() in avail_lower_map:
                matched = avail_lower_map[target_clean.lower()]
            else:
                matched = target_clean  # preserve user target so caller can report missing header

            resolved[canonical_key_clean] = matched
            if matched in avail_clean:
                claimed_sources.add(matched)

        # 2. For remaining unmapped canonical keys, attempt auto-aliasing with ambiguity check
        for canonical_key, candidates in cls.ALIAS_CANDIDATES.items():
            if canonical_key in resolved:
                continue

            matches = []
            for col in avail_clean:
                if col in claimed_sources:
                    continue
                col_low = col.lower()
                for cand in candidates:
                    if col_low == cand:
                        matches.append(col)
                        break

            if len(matches) == 1:
                resolved[canonical_key] = matches[0]
                claimed_sources.add(matches[0])
            elif len(matches) > 1:
                ambiguous.append(f"{canonical_key} matches multiple columns {matches}")
            else:
                unresolved.append(canonical_key)

        return resolved, ambiguous, unresolved

    @classmethod
    def resolve_units_and_scales(
        cls,
        user_units: Dict[str, str],
        unit_conversions: Dict[str, float],
        format_name: str
    ) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, str], List[str]]:
        """
        Resolves unit conversion multipliers and human-readable conversion summaries.
        Returns:
            scales: Dict[dimension, scale_multiplier]
            input_units: Dict[dimension, input_unit_str]
            conversion_summary: Dict[dimension, summary_str]
            errors: List[str]
        """
        scales: Dict[str, float] = {}
        input_units: Dict[str, str] = {}
        conversion_summary: Dict[str, str] = {}
        errors: List[str] = []

        # Standard canonical dimensions
        dimensions = ["timestamp", "frequency", "pulse_width", "aoa", "amplitude"]

        # Default units when format guarantees them:
        # TSRD / HDF5 standard format has ToA in us, Freq in MHz, PW in us, AoA in deg, Amp in dBm
        # CSV / JSON canonical standard: ToA in s, Freq in MHz, PW in us, AoA in deg, Amp in dBm
        format_defaults = {
            "csv": {"timestamp": "s", "frequency": "mhz", "pulse_width": "us", "aoa": "deg", "amplitude": "dbm"},
            "json": {"timestamp": "s", "frequency": "mhz", "pulse_width": "us", "aoa": "deg", "amplitude": "dbm"},
            "hdf5": {"timestamp": "us", "frequency": "mhz", "pulse_width": "us", "aoa": "deg", "amplitude": "dbm"},
        }

        fmt_def = format_defaults.get(format_name.lower(), format_defaults["csv"])

        for dim in dimensions:
            if dim in user_units:
                u_str = str(user_units[dim]).strip().lower()
                try:
                    factor = UnitConverter.get_scale_factor(dim, u_str)
                    scales[dim] = factor
                    input_units[dim] = u_str
                    canonical_u = CANONICAL_TARGET_UNITS[dim]
                    conversion_summary[dim] = f"{u_str} -> {canonical_u}"
                except ValueError as ve:
                    errors.append(str(ve))
            elif dim in unit_conversions:
                # Raw multiplier supplied
                factor = float(unit_conversions[dim])
                scales[dim] = factor
                input_units[dim] = "custom_factor"
                canonical_u = CANONICAL_TARGET_UNITS[dim]
                conversion_summary[dim] = f"factor({factor}) -> {canonical_u}"
            else:
                # Use format default
                def_u = fmt_def[dim]
                factor = UnitConverter.get_scale_factor(dim, def_u)
                scales[dim] = factor
                input_units[dim] = def_u
                canonical_u = CANONICAL_TARGET_UNITS[dim]
                conversion_summary[dim] = f"{def_u} -> {canonical_u}"

        return scales, input_units, conversion_summary, errors

    @classmethod
    def parse_float(cls, val: Any) -> Optional[float]:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            f = float(val)
            return f if (math.isfinite(f) and not math.isnan(f)) else None
        s = str(val).strip()
        if s == "" or s.lower() in ["nan", "null", "none", "inf", "-inf"]:
            return None
        try:
            f = float(s)
            return f if (math.isfinite(f) and not math.isnan(f)) else None
        except ValueError:
            return None


# =====================================================================
# BASE DATASET ADAPTER (PHASE 3 & 4)
# =====================================================================

class BaseDatasetAdapter(ABC):
    """
    Abstract Base Class for all CASS-EW dataset ingestion adapters.
    Ensures strict separation between observable PDWs and ground-truth metadata.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        config: Optional[DatasetConfig] = None,
        field_mapping: Optional[Dict[str, str]] = None,
        units: Optional[Dict[str, str]] = None,
        unit_conversions: Optional[Dict[str, float]] = None,
        coverage_freq_range_mhz: Optional[Tuple[float, float]] = None
    ):
        self.file_path = str(file_path)
        self.path = Path(file_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {file_path}")

        # Consolidate config vs individual parameters
        if config is not None:
            self.config = config
            self.field_mapping = dict(config.field_mapping)
            self.units = dict(config.units)
            self.unit_conversions = unit_conversions or {}
            self.coverage_freq_range_mhz = config.coverage_freq_range_mhz or coverage_freq_range_mhz
        else:
            self.field_mapping = field_mapping or {}
            self.units = units or {}
            self.unit_conversions = unit_conversions or {}
            self.coverage_freq_range_mhz = coverage_freq_range_mhz
            self.config = DatasetConfig(
                field_mapping=self.field_mapping,
                units=self.units,
                coverage_freq_range_mhz=self.coverage_freq_range_mhz
            )

        self.report: Optional[DatasetValidationReport] = None

    @abstractmethod
    def validate(self) -> DatasetValidationReport:
        """Runs validation and produces the DatasetValidationReport."""
        pass

    def get_report(self) -> DatasetValidationReport:
        """Returns cached validation report or executes validate()."""
        if self.report is None:
            self.report = self.validate()
        return self.report

    @abstractmethod
    def iter_observations(self) -> Iterator[ObservablePDW]:
        """
        Yields scheduler-visible observable PDWs in strict chronological order.
        GUARANTEE: No labels or ground-truth future metadata.
        """
        pass

    @abstractmethod
    def iter_ground_truth(self) -> Iterator[GroundTruthPDW]:
        """
        Yields evaluator-only ground truth stream matching observation indices.
        GUARANTEE: Quarantined for post-scan evaluation only.
        """
        pass

    def metadata(self) -> Dict[str, Any]:
        """Returns dataset metadata dictionary."""
        rep = self.get_report()
        return {
            "source_path": self.file_path,
            "format": rep.format,
            "record_count": rep.record_count,
            "label_count": rep.label_count,
            "time_range_s": rep.time_range_s,
            "frequency_range_mhz": rep.frequency_range_mhz,
            "receiver_metadata": rep.receiver_metadata,
            "is_valid": rep.is_valid,
            "warnings": rep.warnings,
            "errors": rep.validation_errors,
            "field_mapping": rep.resolved_field_mapping,
            "units": rep.conversion_summary
        }

    def _empty_report(
        self,
        fmt: str,
        errors: List[str],
        warnings: List[str],
        units: Dict[str, str],
        total_rows: int = 0,
        valid_rows: int = 0,
        invalid_rows: int = 0,
        defaults: Optional[Dict[str, Any]] = None,
        resolved_mapping: Optional[Dict[str, str]] = None,
        input_units: Optional[Dict[str, str]] = None,
        conversion_summary: Optional[Dict[str, str]] = None,
        unresolved_fields: Optional[List[str]] = None,
        ambiguous_fields: Optional[List[str]] = None
    ) -> DatasetValidationReport:
        return DatasetValidationReport(
            source_path=self.file_path,
            format=fmt,
            record_count=0,
            label_count=0,
            feature_names=[],
            units=units,
            time_range_us=(0.0, 0.0),
            time_range_s=(0.0, 0.0),
            frequency_range_mhz=(0.0, 0.0),
            pulse_width_range_us=(0.0, 0.0),
            aoa_range_deg=(0.0, 0.0),
            amplitude_range_dbm=(0.0, 0.0),
            receiver_metadata={},
            transmitter_count=0,
            scan_mode="Unknown",
            causal_suitability="REQUIRES_RECONSTRUCTION",
            is_valid=False,
            warnings=warnings,
            validation_errors=errors,
            total_rows=total_rows,
            valid_rows=valid_rows,
            invalid_rows=invalid_rows,
            defaults_used=defaults or {},
            source_field_mapping=dict(self.field_mapping),
            resolved_field_mapping=resolved_mapping or {},
            input_units=input_units or {},
            canonical_units=dict(CANONICAL_TARGET_UNITS),
            conversion_summary=conversion_summary or {},
            unresolved_fields=unresolved_fields or [],
            ambiguous_fields=ambiguous_fields or []
        )


# =====================================================================
# CSV DATASET ADAPTER (PHASE 3 & 4)
# =====================================================================

class CSVDatasetAdapter(BaseDatasetAdapter):
    """
    Generalized CSV Adapter.
    Supports user datasets with arbitrary column names, explicit unit models,
    chronological sorting, missing-field defaulting, ambiguity detection,
    and strict anti-leakage.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        config: Optional[DatasetConfig] = None,
        field_mapping: Optional[Dict[str, str]] = None,
        units: Optional[Dict[str, str]] = None,
        unit_conversions: Optional[Dict[str, float]] = None,
        coverage_freq_range_mhz: Optional[Tuple[float, float]] = None
    ):
        super().__init__(file_path, config, field_mapping, units, unit_conversions, coverage_freq_range_mhz)
        self.cached_obs: List[ObservablePDW] = []
        self.cached_gt: List[GroundTruthPDW] = []
        self.report = self.validate()

    def validate(self) -> DatasetValidationReport:
        warnings = []
        errors = []
        defaults_used = {}

        # 1. Resolve Units & Scaling
        scales, in_units, conv_summary, unit_errors = _RecordTableNormalizer.resolve_units_and_scales(
            user_units=self.units,
            unit_conversions=self.unit_conversions,
            format_name="csv"
        )
        if unit_errors:
            errors.extend(unit_errors)
            return self._empty_report("csv", errors, warnings, CANONICAL_TARGET_UNITS, input_units=in_units, conversion_summary=conv_summary)

        # 2. Read Raw CSV Header & Records
        raw_rows: List[Dict[str, Any]] = []
        try:
            with open(self.file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    errors.append("CSV file is empty or has no header line.")
                    return self._empty_report("csv", errors, warnings, CANONICAL_TARGET_UNITS)
                fieldnames = [fn.strip() for fn in reader.fieldnames]
                for r in reader:
                    raw_rows.append(r)
        except Exception as e:
            errors.append(f"Failed to read CSV file: {str(e)}")
            return self._empty_report("csv", errors, warnings, CANONICAL_TARGET_UNITS)

        total_rows = len(raw_rows)
        if total_rows == 0:
            errors.append("CSV file has 0 data records.")
            return self._empty_report("csv", errors, warnings, CANONICAL_TARGET_UNITS, total_rows=0)

        # 3. Resolve Field Mapping & Detect Ambiguities
        resolved_map, ambiguous_fields, unresolved_fields = _RecordTableNormalizer.resolve_field_mapping(
            available_keys=fieldnames,
            user_mapping=self.field_mapping
        )

        if ambiguous_fields:
            for amb in ambiguous_fields:
                errors.append(f"AMBIGUOUS FIELD MAPPING: {amb}. Explicit mapping required.")
            return self._empty_report(
                "csv", errors, warnings, CANONICAL_TARGET_UNITS,
                total_rows=total_rows, invalid_rows=total_rows,
                resolved_mapping=resolved_map, ambiguous_fields=ambiguous_fields, unresolved_fields=unresolved_fields
            )

        col_time = resolved_map.get("timestamp")
        col_freq = resolved_map.get("frequency")
        col_pw = resolved_map.get("pulse_width")
        col_aoa = resolved_map.get("aoa")
        col_amp = resolved_map.get("amplitude")
        col_lbl = resolved_map.get("emitter_label")
        col_rx = resolved_map.get("receiver_id")

        if col_time is None or col_time not in fieldnames:
            errors.append(f"Required timestamp column '{self.field_mapping.get('timestamp', 'timestamp')}' not found in CSV headers {fieldnames}.")
        if col_freq is None or col_freq not in fieldnames:
            errors.append(f"Required frequency column '{self.field_mapping.get('frequency', 'frequency')}' not found in CSV headers {fieldnames}.")

        if errors:
            return self._empty_report(
                "csv", errors, warnings, CANONICAL_TARGET_UNITS,
                total_rows=total_rows, invalid_rows=total_rows,
                resolved_mapping=resolved_map, ambiguous_fields=ambiguous_fields, unresolved_fields=unresolved_fields
            )

        if col_pw is None or col_pw not in fieldnames:
            defaults_used["pulse_width_us"] = DEFAULT_PULSE_WIDTH_US
            warnings.append(f"Optional pulse width column missing; default {DEFAULT_PULSE_WIDTH_US} us applied.")
        if col_aoa is None or col_aoa not in fieldnames:
            defaults_used["aoa_deg"] = DEFAULT_AOA_DEG
            warnings.append(f"Optional AoA column missing; default {DEFAULT_AOA_DEG} deg applied.")
        if col_amp is None or col_amp not in fieldnames:
            defaults_used["amplitude_dbm"] = DEFAULT_AMPLITUDE_DBM
            warnings.append(f"Optional amplitude column missing; default {DEFAULT_AMPLITUDE_DBM} dBm applied.")

        # 4. Parse & Validate Rows
        time_scale = scales["timestamp"]
        freq_scale = scales["frequency"]
        pw_scale = scales["pulse_width"]
        aoa_scale = scales["aoa"]
        amp_scale = scales["amplitude"]

        parsed_records = []
        invalid_rows_count = 0
        row_errors = []

        for row_idx, r in enumerate(raw_rows):
            # Timestamp
            t_raw = _RecordTableNormalizer.parse_float(r.get(col_time))
            if t_raw is None:
                row_errors.append(f"Row {row_idx}: Invalid or non-numeric timestamp '{r.get(col_time)}'")
                invalid_rows_count += 1
                continue
            t_s = float(t_raw) * time_scale

            # Frequency
            f_raw = _RecordTableNormalizer.parse_float(r.get(col_freq))
            if f_raw is None:
                row_errors.append(f"Row {row_idx}: Invalid or non-numeric frequency '{r.get(col_freq)}'")
                invalid_rows_count += 1
                continue
            freq_mhz = float(f_raw) * freq_scale

            if self.coverage_freq_range_mhz is not None:
                f_min, f_max = self.coverage_freq_range_mhz
                if not (f_min <= freq_mhz <= f_max):
                    warnings.append(f"Row {row_idx}: Frequency {freq_mhz:.2f} MHz outside coverage [{f_min:.2f}, {f_max:.2f}] MHz.")

            # Optional Pulse Width
            pw_us = DEFAULT_PULSE_WIDTH_US
            if col_pw and col_pw in fieldnames and r.get(col_pw) is not None:
                parsed_pw = _RecordTableNormalizer.parse_float(r.get(col_pw))
                if parsed_pw is not None and parsed_pw > 0:
                    pw_us = float(parsed_pw) * pw_scale
                else:
                    defaults_used["pulse_width_fallback_count"] = defaults_used.get("pulse_width_fallback_count", 0) + 1

            # Optional AoA
            aoa_deg = DEFAULT_AOA_DEG
            if col_aoa and col_aoa in fieldnames and r.get(col_aoa) is not None:
                parsed_aoa = _RecordTableNormalizer.parse_float(r.get(col_aoa))
                if parsed_aoa is not None:
                    aoa_deg = float(parsed_aoa) * aoa_scale
                else:
                    defaults_used["aoa_fallback_count"] = defaults_used.get("aoa_fallback_count", 0) + 1

            # Optional Amplitude
            amp_dbm = DEFAULT_AMPLITUDE_DBM
            if col_amp and col_amp in fieldnames and r.get(col_amp) is not None:
                parsed_amp = _RecordTableNormalizer.parse_float(r.get(col_amp))
                if parsed_amp is not None:
                    amp_dbm = float(parsed_amp) * amp_scale
                else:
                    defaults_used["amp_fallback_count"] = defaults_used.get("amp_fallback_count", 0) + 1

            # Emitter Label (Ground Truth Only)
            lbl = DEFAULT_EMITTER_LABEL
            if col_lbl and col_lbl in fieldnames and r.get(col_lbl) is not None:
                lbl_val = str(r.get(col_lbl)).strip()
                try:
                    lbl = int(lbl_val)
                except ValueError:
                    lbl = lbl_val

            # Receiver ID
            rx_id = str(r.get(col_rx, DEFAULT_RECEIVER_ID)).strip() if (col_rx and col_rx in fieldnames) else DEFAULT_RECEIVER_ID

            parsed_records.append({
                "t_s": t_s,
                "freq_mhz": freq_mhz,
                "pw_us": pw_us,
                "aoa_deg": aoa_deg,
                "amp_dbm": amp_dbm,
                "label": lbl,
                "rx_id": rx_id,
                "orig_row": row_idx
            })

        if row_errors:
            errors.extend(row_errors[:10])
            if len(row_errors) > 10:
                errors.append(f"... and {len(row_errors) - 10} more invalid rows.")

        if invalid_rows_count > 0:
            return self._empty_report(
                "csv", errors, warnings, CANONICAL_TARGET_UNITS,
                total_rows=total_rows,
                valid_rows=len(parsed_records),
                invalid_rows=invalid_rows_count,
                defaults=defaults_used,
                resolved_mapping=resolved_map,
                input_units=in_units,
                conversion_summary=conv_summary
            )

        if len(parsed_records) == 0:
            errors.append("No valid records found after CSV parsing.")
            return self._empty_report("csv", errors, warnings, CANONICAL_TARGET_UNITS, total_rows=total_rows, invalid_rows=total_rows)

        # 5. Chronological Sorting Guarantee
        parsed_records.sort(key=lambda x: x["t_s"])
        was_sorted = all(parsed_records[i]["orig_row"] <= parsed_records[i + 1]["orig_row"] for i in range(len(parsed_records) - 1))
        if not was_sorted:
            warnings.append("CSV records were not in chronological order; records sorted automatically by timestamp.")

        self.cached_obs = []
        self.cached_gt = []

        toas_s = []
        toas_us = []
        freqs = []
        pws = []
        aoas = []
        amps = []
        labels = []

        for idx, rec in enumerate(parsed_records):
            t_s = rec["t_s"]
            t_us = TimeNormalizer.seconds_to_us(t_s)
            obs = ObservablePDW(
                index=idx,
                timestamp_us=t_us,
                timestamp_s=t_s,
                frequency_mhz=rec["freq_mhz"],
                pulse_width_us=rec["pw_us"],
                angle_of_arrival_deg=rec["aoa_deg"],
                amplitude_dbm=rec["amp_dbm"],
                receiver_id=rec["rx_id"]
            )
            gt = GroundTruthPDW(
                index=idx,
                emitter_label=rec["label"],
                transmitter_id=f"TX_{rec['label']}"
            )
            self.cached_obs.append(obs)
            self.cached_gt.append(gt)

            toas_s.append(t_s)
            toas_us.append(t_us)
            freqs.append(rec["freq_mhz"])
            pws.append(rec["pw_us"])
            aoas.append(rec["aoa_deg"])
            amps.append(rec["amp_dbm"])
            labels.append(rec["label"])

        return DatasetValidationReport(
            source_path=self.file_path,
            format="csv",
            record_count=len(self.cached_obs),
            label_count=len(set(labels)),
            feature_names=[col_time, col_freq, col_pw or "pulse_width", col_aoa or "aoa", col_amp or "amplitude"],
            units=CANONICAL_TARGET_UNITS,
            time_range_us=(min(toas_us), max(toas_us)),
            time_range_s=(min(toas_s), max(toas_s)),
            frequency_range_mhz=(min(freqs), max(freqs)),
            pulse_width_range_us=(min(pws), max(pws)),
            aoa_range_deg=(min(aoas), max(aoas)),
            amplitude_range_dbm=(min(amps), max(amps)),
            receiver_metadata={},
            transmitter_count=len(set(labels)),
            scan_mode="Synthetic_CSV",
            causal_suitability="CAUSAL_ENVIRONMENT_READY",
            is_valid=(len(errors) == 0),
            warnings=warnings,
            validation_errors=errors,
            total_rows=total_rows,
            valid_rows=len(self.cached_obs),
            invalid_rows=invalid_rows_count,
            defaults_used=defaults_used,
            source_field_mapping=dict(self.field_mapping),
            resolved_field_mapping=resolved_map,
            input_units=in_units,
            canonical_units=dict(CANONICAL_TARGET_UNITS),
            conversion_summary=conv_summary,
            unresolved_fields=unresolved_fields,
            ambiguous_fields=ambiguous_fields
        )

    def iter_observations(self) -> Iterator[ObservablePDW]:
        if not self.report.is_valid:
            raise ValueError(f"Cannot iterate over invalid dataset: {self.report.validation_errors}")
        for obs in self.cached_obs:
            yield obs

    def iter_ground_truth(self) -> Iterator[GroundTruthPDW]:
        if not self.report.is_valid:
            raise ValueError(f"Cannot iterate over invalid dataset: {self.report.validation_errors}")
        for gt in self.cached_gt:
            yield gt


# =====================================================================
# JSON DATASET ADAPTER (PHASE 3 & 4)
# =====================================================================

class JSONDatasetAdapter(BaseDatasetAdapter):
    """
    Generalized JSON Adapter.
    Supports JSON records (list of dicts) with custom keys, explicit unit models,
    chronological sorting, missing-field defaulting, and strict anti-leakage.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        config: Optional[DatasetConfig] = None,
        field_mapping: Optional[Dict[str, str]] = None,
        units: Optional[Dict[str, str]] = None,
        unit_conversions: Optional[Dict[str, float]] = None,
        coverage_freq_range_mhz: Optional[Tuple[float, float]] = None
    ):
        super().__init__(file_path, config, field_mapping, units, unit_conversions, coverage_freq_range_mhz)
        self.cached_obs: List[ObservablePDW] = []
        self.cached_gt: List[GroundTruthPDW] = []
        self.report = self.validate()

    def validate(self) -> DatasetValidationReport:
        warnings = []
        errors = []
        defaults_used = {}

        # 1. Resolve Units & Scaling
        scales, in_units, conv_summary, unit_errors = _RecordTableNormalizer.resolve_units_and_scales(
            user_units=self.units,
            unit_conversions=self.unit_conversions,
            format_name="json"
        )
        if unit_errors:
            errors.extend(unit_errors)
            return self._empty_report("json", errors, warnings, CANONICAL_TARGET_UNITS, input_units=in_units, conversion_summary=conv_summary)

        # 2. Read JSON
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            errors.append(f"Failed to read/parse JSON: {str(e)}")
            return self._empty_report("json", errors, warnings, CANONICAL_TARGET_UNITS)

        if not isinstance(data, list):
            errors.append(f"JSON root must be a list of PDW record objects, got {type(data).__name__}.")
            return self._empty_report("json", errors, warnings, CANONICAL_TARGET_UNITS)

        total_rows = len(data)
        if total_rows == 0:
            errors.append("JSON dataset contains 0 records.")
            return self._empty_report("json", errors, warnings, CANONICAL_TARGET_UNITS, total_rows=0)

        # Inspect available keys
        sample_keys = list(data[0].keys()) if (total_rows > 0 and isinstance(data[0], dict)) else []

        # 3. Resolve Mapping & Ambiguity
        resolved_map, ambiguous_fields, unresolved_fields = _RecordTableNormalizer.resolve_field_mapping(
            available_keys=sample_keys,
            user_mapping=self.field_mapping
        )

        if ambiguous_fields:
            for amb in ambiguous_fields:
                errors.append(f"AMBIGUOUS FIELD MAPPING: {amb}. Explicit mapping required.")
            return self._empty_report(
                "json", errors, warnings, CANONICAL_TARGET_UNITS,
                total_rows=total_rows, invalid_rows=total_rows,
                resolved_mapping=resolved_map, ambiguous_fields=ambiguous_fields, unresolved_fields=unresolved_fields
            )

        col_time = resolved_map.get("timestamp")
        col_freq = resolved_map.get("frequency")
        col_pw = resolved_map.get("pulse_width")
        col_aoa = resolved_map.get("aoa")
        col_amp = resolved_map.get("amplitude")
        col_lbl = resolved_map.get("emitter_label")
        col_rx = resolved_map.get("receiver_id")

        if col_time is None:
            errors.append(f"Required timestamp field '{self.field_mapping.get('timestamp', 'timestamp')}' not found in JSON keys.")
        if col_freq is None:
            errors.append(f"Required frequency field '{self.field_mapping.get('frequency', 'frequency')}' not found in JSON keys.")

        if errors:
            return self._empty_report(
                "json", errors, warnings, CANONICAL_TARGET_UNITS,
                total_rows=total_rows, invalid_rows=total_rows,
                resolved_mapping=resolved_map, ambiguous_fields=ambiguous_fields, unresolved_fields=unresolved_fields
            )

        if col_pw is None:
            defaults_used["pulse_width_us"] = DEFAULT_PULSE_WIDTH_US
            warnings.append(f"Optional pulse width missing; default {DEFAULT_PULSE_WIDTH_US} us applied.")
        if col_aoa is None:
            defaults_used["aoa_deg"] = DEFAULT_AOA_DEG
            warnings.append(f"Optional AoA missing; default {DEFAULT_AOA_DEG} deg applied.")
        if col_amp is None:
            defaults_used["amplitude_dbm"] = DEFAULT_AMPLITUDE_DBM
            warnings.append(f"Optional amplitude missing; default {DEFAULT_AMPLITUDE_DBM} dBm applied.")

        # 4. Parse Rows
        time_scale = scales["timestamp"]
        freq_scale = scales["frequency"]
        pw_scale = scales["pulse_width"]
        aoa_scale = scales["aoa"]
        amp_scale = scales["amplitude"]

        parsed_records = []
        invalid_rows_count = 0
        row_errors = []

        for row_idx, r in enumerate(data):
            if not isinstance(r, dict):
                row_errors.append(f"Record {row_idx}: Record is not a JSON object.")
                invalid_rows_count += 1
                continue

            # Timestamp
            t_raw = _RecordTableNormalizer.parse_float(r.get(col_time))
            if t_raw is None:
                row_errors.append(f"Record {row_idx}: Invalid timestamp '{r.get(col_time)}'")
                invalid_rows_count += 1
                continue
            t_s = float(t_raw) * time_scale

            # Frequency
            f_raw = _RecordTableNormalizer.parse_float(r.get(col_freq))
            if f_raw is None:
                row_errors.append(f"Record {row_idx}: Invalid frequency '{r.get(col_freq)}'")
                invalid_rows_count += 1
                continue
            freq_mhz = float(f_raw) * freq_scale

            if self.coverage_freq_range_mhz is not None:
                f_min, f_max = self.coverage_freq_range_mhz
                if not (f_min <= freq_mhz <= f_max):
                    warnings.append(f"Record {row_idx}: Frequency {freq_mhz:.2f} MHz outside coverage [{f_min:.2f}, {f_max:.2f}] MHz.")

            # Pulse Width
            pw_us = DEFAULT_PULSE_WIDTH_US
            if col_pw and r.get(col_pw) is not None:
                parsed_pw = _RecordTableNormalizer.parse_float(r.get(col_pw))
                if parsed_pw is not None and parsed_pw > 0:
                    pw_us = float(parsed_pw) * pw_scale
                else:
                    defaults_used["pulse_width_fallback_count"] = defaults_used.get("pulse_width_fallback_count", 0) + 1

            # AoA
            aoa_deg = DEFAULT_AOA_DEG
            if col_aoa and r.get(col_aoa) is not None:
                parsed_aoa = _RecordTableNormalizer.parse_float(r.get(col_aoa))
                if parsed_aoa is not None:
                    aoa_deg = float(parsed_aoa) * aoa_scale
                else:
                    defaults_used["aoa_fallback_count"] = defaults_used.get("aoa_fallback_count", 0) + 1

            # Amplitude
            amp_dbm = DEFAULT_AMPLITUDE_DBM
            if col_amp and r.get(col_amp) is not None:
                parsed_amp = _RecordTableNormalizer.parse_float(r.get(col_amp))
                if parsed_amp is not None:
                    amp_dbm = float(parsed_amp) * amp_scale
                else:
                    defaults_used["amp_fallback_count"] = defaults_used.get("amp_fallback_count", 0) + 1

            # Emitter Label
            lbl = DEFAULT_EMITTER_LABEL
            if col_lbl and r.get(col_lbl) is not None:
                lbl = r.get(col_lbl)

            # Receiver ID
            rx_id = str(r.get(col_rx, DEFAULT_RECEIVER_ID)).strip() if col_rx else DEFAULT_RECEIVER_ID

            parsed_records.append({
                "t_s": t_s,
                "freq_mhz": freq_mhz,
                "pw_us": pw_us,
                "aoa_deg": aoa_deg,
                "amp_dbm": amp_dbm,
                "label": lbl,
                "rx_id": rx_id,
                "orig_row": row_idx
            })

        if row_errors:
            errors.extend(row_errors[:10])
            if len(row_errors) > 10:
                errors.append(f"... and {len(row_errors) - 10} more invalid records.")

        if invalid_rows_count > 0:
            return self._empty_report(
                "json", errors, warnings, CANONICAL_TARGET_UNITS,
                total_rows=total_rows,
                valid_rows=len(parsed_records),
                invalid_rows=invalid_rows_count,
                defaults=defaults_used,
                resolved_mapping=resolved_map,
                input_units=in_units,
                conversion_summary=conv_summary
            )

        if len(parsed_records) == 0:
            errors.append("No valid records found in JSON dataset.")
            return self._empty_report("json", errors, warnings, CANONICAL_TARGET_UNITS, total_rows=total_rows, invalid_rows=total_rows)

        # Sort chronologically
        parsed_records.sort(key=lambda x: x["t_s"])
        was_sorted = all(parsed_records[i]["orig_row"] <= parsed_records[i + 1]["orig_row"] for i in range(len(parsed_records) - 1))
        if not was_sorted:
            warnings.append("JSON records were not in chronological order; records sorted automatically by timestamp.")

        self.cached_obs = []
        self.cached_gt = []

        toas_s = []
        toas_us = []
        freqs = []
        pws = []
        aoas = []
        amps = []
        labels = []

        for idx, rec in enumerate(parsed_records):
            t_s = rec["t_s"]
            t_us = TimeNormalizer.seconds_to_us(t_s)
            obs = ObservablePDW(
                index=idx,
                timestamp_us=t_us,
                timestamp_s=t_s,
                frequency_mhz=rec["freq_mhz"],
                pulse_width_us=rec["pw_us"],
                angle_of_arrival_deg=rec["aoa_deg"],
                amplitude_dbm=rec["amp_dbm"],
                receiver_id=rec["rx_id"]
            )
            gt = GroundTruthPDW(
                index=idx,
                emitter_label=rec["label"],
                transmitter_id=f"TX_{rec['label']}"
            )
            self.cached_obs.append(obs)
            self.cached_gt.append(gt)

            toas_s.append(t_s)
            toas_us.append(t_us)
            freqs.append(rec["freq_mhz"])
            pws.append(rec["pw_us"])
            aoas.append(rec["aoa_deg"])
            amps.append(rec["amp_dbm"])
            labels.append(rec["label"])

        return DatasetValidationReport(
            source_path=self.file_path,
            format="json",
            record_count=len(self.cached_obs),
            label_count=len(set(labels)),
            feature_names=[col_time, col_freq, col_pw or "pulse_width", col_aoa or "aoa", col_amp or "amplitude"],
            units=CANONICAL_TARGET_UNITS,
            time_range_us=(min(toas_us), max(toas_us)),
            time_range_s=(min(toas_s), max(toas_s)),
            frequency_range_mhz=(min(freqs), max(freqs)),
            pulse_width_range_us=(min(pws), max(pws)),
            aoa_range_deg=(min(aoas), max(aoas)),
            amplitude_range_dbm=(min(amps), max(amps)),
            receiver_metadata={},
            transmitter_count=len(set(labels)),
            scan_mode="Synthetic_JSON",
            causal_suitability="CAUSAL_ENVIRONMENT_READY",
            is_valid=(len(errors) == 0),
            warnings=warnings,
            validation_errors=errors,
            total_rows=total_rows,
            valid_rows=len(self.cached_obs),
            invalid_rows=invalid_rows_count,
            defaults_used=defaults_used,
            source_field_mapping=dict(self.field_mapping),
            resolved_field_mapping=resolved_map,
            input_units=in_units,
            canonical_units=dict(CANONICAL_TARGET_UNITS),
            conversion_summary=conv_summary,
            unresolved_fields=unresolved_fields,
            ambiguous_fields=ambiguous_fields
        )

    def iter_observations(self) -> Iterator[ObservablePDW]:
        if not self.report.is_valid:
            raise ValueError(f"Cannot iterate over invalid dataset: {self.report.validation_errors}")
        for obs in self.cached_obs:
            yield obs

    def iter_ground_truth(self) -> Iterator[GroundTruthPDW]:
        if not self.report.is_valid:
            raise ValueError(f"Cannot iterate over invalid dataset: {self.report.validation_errors}")
        for gt in self.cached_gt:
            yield gt


# =====================================================================
# HDF5 DATASET ADAPTER (PHASE 3 & 4)
# =====================================================================

class HDF5DatasetAdapter(BaseDatasetAdapter):
    """
    Generalized HDF5 Adapter for user datasets.
    Supports arbitrary dataset paths via field_mapping:
      e.g. {"data": "/pdws", "labels": "/emitter_ids"}
    Default paths fall back to standard CASS-EW /data and /labels.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        config: Optional[DatasetConfig] = None,
        field_mapping: Optional[Dict[str, str]] = None,
        units: Optional[Dict[str, str]] = None,
        unit_conversions: Optional[Dict[str, float]] = None,
        coverage_freq_range_mhz: Optional[Tuple[float, float]] = None
    ):
        super().__init__(file_path, config, field_mapping, units, unit_conversions, coverage_freq_range_mhz)
        self.cached_obs: List[ObservablePDW] = []
        self.cached_gt: List[GroundTruthPDW] = []
        self.report = self.validate()

    def validate(self) -> DatasetValidationReport:
        if h5py is None:
            raise ImportError("h5py package is required to parse HDF5 files.")

        warnings = []
        errors = []

        # Resolve Units & Scaling
        scales, in_units, conv_summary, unit_errors = _RecordTableNormalizer.resolve_units_and_scales(
            user_units=self.units,
            unit_conversions=self.unit_conversions,
            format_name="hdf5"
        )
        if unit_errors:
            errors.extend(unit_errors)
            return self._empty_report("hdf5", errors, warnings, CANONICAL_TARGET_UNITS, input_units=in_units, conversion_summary=conv_summary)

        data_path = self.field_mapping.get("data", "/data")
        labels_path = self.field_mapping.get("labels", "/labels")

        try:
            with h5py.File(self.file_path, 'r') as f:
                if data_path not in f:
                    errors.append(f"Missing required HDF5 dataset: '{data_path}'")
                    return self._empty_report("hdf5", errors, warnings, CANONICAL_TARGET_UNITS)

                data = f[data_path][:]
                if len(data.shape) != 2 or data.shape[1] < 2:
                    errors.append(f"HDF5 dataset '{data_path}' shape {data.shape} is malformed; expected (N, >=2).")
                    return self._empty_report("hdf5", errors, warnings, CANONICAL_TARGET_UNITS)

                total_rows = data.shape[0]
                if total_rows == 0:
                    errors.append(f"HDF5 dataset '{data_path}' contains 0 records.")
                    return self._empty_report("hdf5", errors, warnings, CANONICAL_TARGET_UNITS, total_rows=0)

                # Check for NaN / Inf
                if np.isnan(data).any():
                    errors.append("HDF5 dataset contains NaN values.")
                if np.isinf(data).any():
                    errors.append("HDF5 dataset contains Inf values.")

                # Labels
                labels = None
                if labels_path in f:
                    labels = f[labels_path][:]
                    if labels.shape[0] != total_rows:
                        errors.append(f"HDF5 data rows ({total_rows}) != labels rows ({labels.shape[0]}).")
                else:
                    warnings.append(f"HDF5 dataset '{labels_path}' not present; default ground truth labels applied.")
                    labels = np.zeros(total_rows, dtype=np.int32)

                if errors:
                    return self._empty_report("hdf5", errors, warnings, CANONICAL_TARGET_UNITS, total_rows=total_rows, invalid_rows=total_rows)

                # Time and column checks
                toas_raw = data[:, 0]
                time_scale = scales["timestamp"]
                toas_s = toas_raw * time_scale
                toas_us = toas_s * 1e6

                # Chronological check / sort
                if not np.all(np.diff(toas_s) >= 0):
                    warnings.append("HDF5 timestamps were not monotonically increasing; records sorted automatically.")
                    sort_indices = np.argsort(toas_s)
                    data = data[sort_indices]
                    labels = labels[sort_indices]
                    toas_s = toas_s[sort_indices]
                    toas_us = toas_us[sort_indices]

                # Extract features with column defaults
                freq_scale = scales["frequency"]
                pw_scale = scales["pulse_width"]
                aoa_scale = scales["aoa"]
                amp_scale = scales["amplitude"]

                freqs = data[:, 1] * freq_scale

                num_cols = data.shape[1]
                pws = (data[:, 2] * pw_scale) if num_cols > 2 else np.full(total_rows, DEFAULT_PULSE_WIDTH_US)
                aoas = (data[:, 3] * aoa_scale) if num_cols > 3 else np.full(total_rows, DEFAULT_AOA_DEG)
                amps = (data[:, 4] * amp_scale) if num_cols > 4 else np.full(total_rows, DEFAULT_AMPLITUDE_DBM)

                self.cached_obs = []
                self.cached_gt = []

                for idx in range(total_rows):
                    obs = ObservablePDW(
                        index=idx,
                        timestamp_us=float(toas_us[idx]),
                        timestamp_s=float(toas_s[idx]),
                        frequency_mhz=float(freqs[idx]),
                        pulse_width_us=float(pws[idx]),
                        angle_of_arrival_deg=float(aoas[idx]),
                        amplitude_dbm=float(amps[idx]),
                        receiver_id=DEFAULT_RECEIVER_ID
                    )
                    lbl_val = labels[idx]
                    lbl = lbl_val.decode('utf-8') if isinstance(lbl_val, bytes) else int(lbl_val)
                    gt = GroundTruthPDW(
                        index=idx,
                        emitter_label=lbl,
                        transmitter_id=f"TX_{lbl}"
                    )
                    self.cached_obs.append(obs)
                    self.cached_gt.append(gt)

                return DatasetValidationReport(
                    source_path=self.file_path,
                    format="hdf5",
                    record_count=total_rows,
                    label_count=len(np.unique(labels)),
                    feature_names=["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"],
                    units=CANONICAL_TARGET_UNITS,
                    time_range_us=(float(np.min(toas_us)), float(np.max(toas_us))),
                    time_range_s=(float(np.min(toas_s)), float(np.max(toas_s))),
                    frequency_range_mhz=(float(np.min(freqs)), float(np.max(freqs))),
                    pulse_width_range_us=(float(np.min(pws)), float(np.max(pws))),
                    aoa_range_deg=(float(np.min(aoas)), float(np.max(aoas))),
                    amplitude_range_dbm=(float(np.min(amps)), float(np.max(amps))),
                    receiver_metadata={},
                    transmitter_count=len(np.unique(labels)),
                    scan_mode="General_HDF5",
                    causal_suitability="CAUSAL_ENVIRONMENT_READY",
                    is_valid=True,
                    warnings=warnings,
                    validation_errors=[],
                    total_rows=total_rows,
                    valid_rows=total_rows,
                    invalid_rows=0,
                    defaults_used={},
                    source_field_mapping=dict(self.field_mapping),
                    resolved_field_mapping=dict(self.field_mapping),
                    input_units=in_units,
                    canonical_units=dict(CANONICAL_TARGET_UNITS),
                    conversion_summary=conv_summary
                )
        except Exception as e:
            errors.append(f"Failed opening/parsing HDF5: {str(e)}")
            return self._empty_report("hdf5", errors, warnings, CANONICAL_TARGET_UNITS)

    def iter_observations(self) -> Iterator[ObservablePDW]:
        if not self.report.is_valid:
            raise ValueError(f"Cannot iterate over invalid dataset: {self.report.validation_errors}")
        for obs in self.cached_obs:
            yield obs

    def iter_ground_truth(self) -> Iterator[GroundTruthPDW]:
        if not self.report.is_valid:
            raise ValueError(f"Cannot iterate over invalid dataset: {self.report.validation_errors}")
        for gt in self.cached_gt:
            yield gt


# =====================================================================
# SPECIALIZED TSRD ADAPTER (PHASE 3 - REFACTOR / PRESERVE)
# =====================================================================

class TSRDAdapter(BaseDatasetAdapter):
    """
    Unified adapter reading TSRD HDF5, JSON, or CSV datasets.
    Preserves exact TSRD scan-mode classification, metadata inspection,
    and backwards compatibility with all existing tests.
    """

    def __init__(self, file_path: Union[str, Path]):
        super().__init__(file_path)
        self.ext = self.path.suffix.lower()
        if self.ext in ['.h5', '.hdf5']:
            self.format = "hdf5"
        elif self.ext == '.json':
            self.format = "json"
        elif self.ext == '.csv':
            self.format = "csv"
        else:
            raise ValueError(f"Unsupported dataset format '{self.ext}'. Supported: .h5, .json, .csv")

        self.report = self.validate()

    def validate(self) -> DatasetValidationReport:
        warnings = []
        errors = []
        units = {
            "ToA": "microseconds",
            "Frequency": "MHz",
            "PulseWidth": "microseconds",
            "AoA": "degrees",
            "Amplitude": "dBm"
        }

        if self.format == "hdf5":
            return self._validate_hdf5(warnings, errors, units)
        elif self.format == "json":
            return self._validate_json(warnings, errors, units)
        else:
            return self._validate_csv(warnings, errors, units)

    def _validate_hdf5(self, warnings: List[str], errors: List[str], units: Dict[str, str]) -> DatasetValidationReport:
        if h5py is None:
            raise ImportError("h5py package is required to parse HDF5 files.")

        try:
            with h5py.File(self.file_path, 'r') as f:
                required = ['/data', '/labels', '/metadata', '/metadata/feature_names']
                for req in required:
                    if req not in f:
                        errors.append(f"Missing required HDF5 path: {req}")

                if errors:
                    return self._empty_report("hdf5", errors, warnings, units)

                data = f['/data'][:]
                labels = f['/labels'][:]

                if len(data.shape) != 2 or data.shape[1] != 5:
                    errors.append(f"Malformed /data shape: {data.shape}. Expected (N, 5).")
                    return self._empty_report("hdf5", errors, warnings, units)

                if data.shape[0] != labels.shape[0]:
                    errors.append(f"Data row count ({data.shape[0]}) does not match labels ({labels.shape[0]}).")

                total_rows = data.shape[0]
                if total_rows == 0:
                    errors.append("Dataset /data contains 0 records.")
                    return self._empty_report("hdf5", errors, warnings, units, total_rows=0)

                # Check for NaN / Inf
                if np.isnan(data).any():
                    errors.append("Dataset /data contains NaN values.")
                if np.isinf(data).any():
                    errors.append("Dataset /data contains Inf values.")

                # Monotonicity check
                toas_us = data[:, 0]
                if not np.all(np.diff(toas_us) >= 0):
                    errors.append("ToA is not strictly monotonically increasing.")

                # Read feature names
                fn_raw = f['/metadata/feature_names'][:]
                feature_names = [fn.decode('utf-8') if isinstance(fn, bytes) else str(fn) for fn in fn_raw]
                expected_fn = ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"]
                if feature_names != expected_fn:
                    warnings.append(f"Feature names differ from expected: got {feature_names}, expected {expected_fn}")

                # Receiver & transmitter metadata
                receiver_meta = {}
                scan_mode = "Unknown"
                if '/metadata/receiver' in f:
                    rx_obj = f['/metadata/receiver']
                    for k, v in rx_obj.attrs.items():
                        val = v.decode('utf-8') if isinstance(v, bytes) else v
                        receiver_meta[k] = val
                    if 'scan_mode' in receiver_meta:
                        scan_mode = str(receiver_meta['scan_mode'])
                else:
                    warnings.append("Missing /metadata/receiver group.")

                tx_count = 0
                if '/metadata/transmitters' in f:
                    tx_obj = f['/metadata/transmitters']
                    tx_count = len(tx_obj.keys()) if isinstance(tx_obj, h5py.Group) else 1

                # Classify causal suitability
                if scan_mode.lower() in ['staring', 'stare']:
                    suitability = "CAUSAL_ENVIRONMENT_READY"
                elif scan_mode.lower() in ['scanning', 'scan']:
                    suitability = "HISTORICAL_OBSERVATION_ONLY"
                    warnings.append("SCAN mode: Missing historical pulses cannot be treated as transmitter silence.")
                else:
                    suitability = "REQUIRES_RECONSTRUCTION"

                time_range_us = (float(np.min(toas_us)), float(np.max(toas_us)))
                time_range_s = (TimeNormalizer.us_to_seconds(time_range_us[0]), TimeNormalizer.us_to_seconds(time_range_us[1]))

                is_valid = (len(errors) == 0)
                return DatasetValidationReport(
                    source_path=self.file_path,
                    format="hdf5",
                    record_count=int(data.shape[0]) if is_valid else 0,
                    label_count=len(np.unique(labels)) if is_valid else 0,
                    feature_names=feature_names,
                    units=units,
                    time_range_us=time_range_us if is_valid else (0.0, 0.0),
                    time_range_s=time_range_s if is_valid else (0.0, 0.0),
                    frequency_range_mhz=(float(np.min(data[:, 1])), float(np.max(data[:, 1]))) if is_valid else (0.0, 0.0),
                    pulse_width_range_us=(float(np.min(data[:, 2])), float(np.max(data[:, 2]))) if is_valid else (0.0, 0.0),
                    aoa_range_deg=(float(np.min(data[:, 3])), float(np.max(data[:, 3]))) if is_valid else (0.0, 0.0),
                    amplitude_range_dbm=(float(np.min(data[:, 4])), float(np.max(data[:, 4]))) if is_valid else (0.0, 0.0),
                    receiver_metadata=receiver_meta,
                    transmitter_count=tx_count,
                    scan_mode=scan_mode,
                    causal_suitability=suitability,
                    is_valid=is_valid,
                    warnings=warnings,
                    validation_errors=errors,
                    total_rows=total_rows,
                    valid_rows=total_rows if is_valid else 0,
                    invalid_rows=0 if is_valid else total_rows,
                    source_field_mapping={"data": "/data", "labels": "/labels"},
                    resolved_field_mapping={"data": "/data", "labels": "/labels"},
                    input_units={"timestamp": "us", "frequency": "mhz", "pulse_width": "us", "aoa": "deg", "amplitude": "dbm"},
                    conversion_summary={"timestamp": "us -> s", "frequency": "MHz -> MHz", "pulse_width": "us -> us", "aoa": "deg -> deg", "amplitude": "dBm -> dBm"}
                )
        except Exception as e:
            errors.append(f"Failed opening/parsing HDF5: {str(e)}")
            return self._empty_report("hdf5", errors, warnings, units)

    def _validate_json(self, warnings: List[str], errors: List[str], units: Dict[str, str]) -> DatasetValidationReport:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                errors.append("JSON fixture root must be an array of PDW records.")
                return self._empty_report("json", errors, warnings, units)

            total_rows = len(data)
            if total_rows == 0:
                errors.append("JSON fixture is empty.")
                return self._empty_report("json", errors, warnings, units, total_rows=0)

            toas_s = []
            freqs = []
            pws = []
            aoas = []
            amps = []
            labels = []

            for i, r in enumerate(data):
                if "timestamp" not in r or "frequency" not in r:
                    errors.append(f"Record {i} missing required keys ('timestamp', 'frequency').")
                    continue
                toas_s.append(float(r["timestamp"]))
                freqs.append(float(r["frequency"]))
                pws.append(float(r.get("pulse_width", DEFAULT_PULSE_WIDTH_US)))
                aoas.append(float(r.get("angle", DEFAULT_AOA_DEG)))
                amps.append(float(r.get("amplitude", DEFAULT_AMPLITUDE_DBM)))
                labels.append(r.get("emitter_id", DEFAULT_EMITTER_LABEL))

            if errors:
                return self._empty_report("json", errors, warnings, units, total_rows=total_rows, invalid_rows=len(errors))

            toas_us = [TimeNormalizer.seconds_to_us(t) for t in toas_s]
            if not np.all(np.diff(toas_us) >= 0):
                errors.append("ToA timestamps are not monotonic.")

            is_valid = (len(errors) == 0)
            return DatasetValidationReport(
                source_path=self.file_path,
                format="json",
                record_count=len(data) if is_valid else 0,
                label_count=len(set(labels)) if is_valid else 0,
                feature_names=["timestamp", "frequency", "pulse_width", "angle", "amplitude"],
                units={"timestamp": "seconds", "frequency": "MHz", "pulse_width": "us", "angle": "deg", "amplitude": "dBm"},
                time_range_us=(min(toas_us), max(toas_us)) if is_valid else (0.0, 0.0),
                time_range_s=(min(toas_s), max(toas_s)) if is_valid else (0.0, 0.0),
                frequency_range_mhz=(min(freqs), max(freqs)) if is_valid else (0.0, 0.0),
                pulse_width_range_us=(min(pws), max(pws)) if is_valid else (0.0, 0.0),
                aoa_range_deg=(min(aoas), max(aoas)) if is_valid else (0.0, 0.0),
                amplitude_range_dbm=(min(amps), max(amps)) if is_valid else (0.0, 0.0),
                receiver_metadata={},
                transmitter_count=len(set(labels)),
                scan_mode="Synthetic_JSON",
                causal_suitability="CAUSAL_ENVIRONMENT_READY",
                is_valid=is_valid,
                warnings=warnings,
                validation_errors=errors,
                total_rows=total_rows,
                valid_rows=len(data) if is_valid else 0,
                invalid_rows=0 if is_valid else total_rows,
                resolved_field_mapping={"timestamp": "timestamp", "frequency": "frequency", "pulse_width": "pulse_width", "aoa": "angle", "amplitude": "amplitude"},
                input_units={"timestamp": "s", "frequency": "mhz", "pulse_width": "us", "aoa": "deg", "amplitude": "dbm"},
                conversion_summary={"timestamp": "s -> s", "frequency": "MHz -> MHz", "pulse_width": "us -> us", "aoa": "deg -> deg", "amplitude": "dBm -> dBm"}
            )
        except Exception as e:
            errors.append(f"JSON validation error: {str(e)}")
            return self._empty_report("json", errors, warnings, units)

    def _validate_csv(self, warnings: List[str], errors: List[str], units: Dict[str, str]) -> DatasetValidationReport:
        try:
            toas_s = []
            freqs = []
            pws = []
            aoas = []
            amps = []
            labels = []

            with open(self.file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, r in enumerate(reader):
                    if "timestamp" not in r or "frequency" not in r:
                        errors.append(f"CSV row {i} missing required headers.")
                        break
                    toas_s.append(float(r["timestamp"]))
                    freqs.append(float(r["frequency"]))
                    pws.append(float(r.get("pulse_width", DEFAULT_PULSE_WIDTH_US)))
                    aoas.append(float(r.get("angle", DEFAULT_AOA_DEG)))
                    amps.append(float(r.get("amplitude", DEFAULT_AMPLITUDE_DBM)))
                    labels.append(r.get("emitter_id", DEFAULT_EMITTER_LABEL))

            if errors or len(toas_s) == 0:
                if len(toas_s) == 0 and not errors:
                    errors.append("CSV fixture has zero records.")
                return self._empty_report("csv", errors, warnings, units, total_rows=len(toas_s), invalid_rows=len(toas_s))

            toas_us = [TimeNormalizer.seconds_to_us(t) for t in toas_s]
            if not np.all(np.diff(toas_us) >= 0):
                errors.append("ToA timestamps are not monotonic.")

            is_valid = (len(errors) == 0)
            return DatasetValidationReport(
                source_path=self.file_path,
                format="csv",
                record_count=len(toas_s) if is_valid else 0,
                label_count=len(set(labels)) if is_valid else 0,
                feature_names=["timestamp", "frequency", "pulse_width", "angle", "amplitude"],
                units={"timestamp": "seconds", "frequency": "MHz", "pulse_width": "us", "angle": "deg", "amplitude": "dBm"},
                time_range_us=(min(toas_us), max(toas_us)) if is_valid else (0.0, 0.0),
                time_range_s=(min(toas_s), max(toas_s)) if is_valid else (0.0, 0.0),
                frequency_range_mhz=(min(freqs), max(freqs)) if is_valid else (0.0, 0.0),
                pulse_width_range_us=(min(pws), max(pws)) if is_valid else (0.0, 0.0),
                aoa_range_deg=(min(aoas), max(aoas)) if is_valid else (0.0, 0.0),
                amplitude_range_dbm=(min(amps), max(amps)) if is_valid else (0.0, 0.0),
                receiver_metadata={},
                transmitter_count=len(set(labels)),
                scan_mode="Synthetic_CSV",
                causal_suitability="CAUSAL_ENVIRONMENT_READY",
                is_valid=is_valid,
                warnings=warnings,
                validation_errors=errors,
                total_rows=len(toas_s),
                valid_rows=len(toas_s) if is_valid else 0,
                invalid_rows=0 if is_valid else len(toas_s),
                resolved_field_mapping={"timestamp": "timestamp", "frequency": "frequency", "pulse_width": "pulse_width", "aoa": "angle", "amplitude": "amplitude"},
                input_units={"timestamp": "s", "frequency": "mhz", "pulse_width": "us", "aoa": "deg", "amplitude": "dbm"},
                conversion_summary={"timestamp": "s -> s", "frequency": "MHz -> MHz", "pulse_width": "us -> us", "aoa": "deg -> deg", "amplitude": "dBm -> dBm"}
            )
        except Exception as e:
            errors.append(f"CSV validation error: {str(e)}")
            return self._empty_report("csv", errors, warnings, units)

    def iter_observations(self) -> Iterator[ObservablePDW]:
        if not self.report.is_valid:
            raise ValueError(f"Cannot iterate over invalid dataset: {self.report.validation_errors}")

        if self.format == "hdf5":
            with h5py.File(self.file_path, 'r') as f:
                data = f['/data']
                for i in range(data.shape[0]):
                    row = data[i]
                    t_us = float(row[0])
                    yield ObservablePDW(
                        index=i,
                        timestamp_us=t_us,
                        timestamp_s=TimeNormalizer.us_to_seconds(t_us),
                        frequency_mhz=float(row[1]),
                        pulse_width_us=float(row[2]),
                        angle_of_arrival_deg=float(row[3]),
                        amplitude_dbm=float(row[4]),
                        receiver_id=DEFAULT_RECEIVER_ID
                    )
        elif self.format == "json":
            with open(self.file_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
            for i, r in enumerate(records):
                t_s = float(r["timestamp"])
                yield ObservablePDW(
                    index=i,
                    timestamp_us=TimeNormalizer.seconds_to_us(t_s),
                    timestamp_s=t_s,
                    frequency_mhz=float(r["frequency"]),
                    pulse_width_us=float(r.get("pulse_width", DEFAULT_PULSE_WIDTH_US)),
                    angle_of_arrival_deg=float(r.get("angle", DEFAULT_AOA_DEG)),
                    amplitude_dbm=float(r.get("amplitude", DEFAULT_AMPLITUDE_DBM)),
                    receiver_id=str(r.get("receiver_id", DEFAULT_RECEIVER_ID))
                )
        elif self.format == "csv":
            with open(self.file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, r in enumerate(reader):
                    t_s = float(r["timestamp"])
                    yield ObservablePDW(
                        index=i,
                        timestamp_us=TimeNormalizer.seconds_to_us(t_s),
                        timestamp_s=t_s,
                        frequency_mhz=float(r["frequency"]),
                        pulse_width_us=float(r.get("pulse_width", DEFAULT_PULSE_WIDTH_US)),
                        angle_of_arrival_deg=float(r.get("angle", DEFAULT_AOA_DEG)),
                        amplitude_dbm=float(r.get("amplitude", DEFAULT_AMPLITUDE_DBM)),
                        receiver_id=str(r.get("receiver_id", DEFAULT_RECEIVER_ID))
                    )

    def iter_ground_truth(self) -> Iterator[GroundTruthPDW]:
        if not self.report.is_valid:
            raise ValueError(f"Cannot iterate over invalid dataset: {self.report.validation_errors}")

        if self.format == "hdf5":
            with h5py.File(self.file_path, 'r') as f:
                labels = f['/labels']
                for i in range(labels.shape[0]):
                    val = labels[i]
                    emitter_lbl = val.decode('utf-8') if isinstance(val, bytes) else int(val)
                    yield GroundTruthPDW(index=i, emitter_label=emitter_lbl, transmitter_id=f"TX_{emitter_lbl}")
        elif self.format == "json":
            with open(self.file_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
            for i, r in enumerate(records):
                yield GroundTruthPDW(index=i, emitter_label=r.get("emitter_id", DEFAULT_EMITTER_LABEL))
        elif self.format == "csv":
            with open(self.file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, r in enumerate(reader):
                    yield GroundTruthPDW(index=i, emitter_label=r.get("emitter_id", DEFAULT_EMITTER_LABEL))


# =====================================================================
# DATASET FACTORY & AUTO-FORMAT DETECTION (PHASE 3 & 4)
# =====================================================================

def load_dataset(
    file_path: Union[str, Path],
    config: Optional[DatasetConfig] = None,
    field_mapping: Optional[Dict[str, str]] = None,
    units: Optional[Dict[str, str]] = None,
    unit_conversions: Optional[Dict[str, float]] = None,
    coverage_freq_range_mhz: Optional[Tuple[float, float]] = None,
    prefer_tsrd: bool = False
) -> BaseDatasetAdapter:
    """
    Factory function that automatically selects the appropriate dataset adapter
    based on file extension, dataset structure, and optional configuration.

    Supported formats:
    - .csv -> CSVDatasetAdapter
    - .json -> JSONDatasetAdapter
    - .h5 / .hdf5 -> TSRDAdapter (if standard TSRD groups present or prefer_tsrd=True)
                     or HDF5DatasetAdapter
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {file_path}")

    ext = path.suffix.lower()

    if ext == ".csv":
        return CSVDatasetAdapter(
            file_path=file_path,
            config=config,
            field_mapping=field_mapping,
            units=units,
            unit_conversions=unit_conversions,
            coverage_freq_range_mhz=coverage_freq_range_mhz
        )

    elif ext == ".json":
        return JSONDatasetAdapter(
            file_path=file_path,
            config=config,
            field_mapping=field_mapping,
            units=units,
            unit_conversions=unit_conversions,
            coverage_freq_range_mhz=coverage_freq_range_mhz
        )

    elif ext in [".h5", ".hdf5"]:
        if h5py is not None:
            try:
                with h5py.File(file_path, 'r') as f:
                    is_tsrd = ('/data' in f and '/metadata/feature_names' in f)
            except Exception:
                is_tsrd = False
        else:
            is_tsrd = False

        if (is_tsrd or prefer_tsrd) and (config is None and field_mapping is None and units is None):
            return TSRDAdapter(file_path=file_path)
        else:
            return HDF5DatasetAdapter(
                file_path=file_path,
                config=config,
                field_mapping=field_mapping,
                units=units,
                unit_conversions=unit_conversions,
                coverage_freq_range_mhz=coverage_freq_range_mhz
            )

    else:
        raise ValueError(
            f"Unsupported dataset format '{ext}'. Supported formats: .csv, .json, .h5, .hdf5"
        )
