"""
Causal RF Dataset Environment & Dataset-Backed Virtual Receiver
CASS-EW SIH Problem Statement 26055

CRITICAL ARCHITECTURAL ROLES:
1. DatasetRFEnvironment:
   Answers: "What pulses physically existed in this time/frequency region?"
   NOT: "What did the historical receiver happen to record?"
   Indexes observable pulses by timestamp and frequency.

2. DatasetVirtualReceiver:
   Executes cognitive receiver decisions: (receiver_id, band_idx, dwell_duration_s)
   Queries DatasetRFEnvironment across [current_time, current_time + dwell_s]
   Applies RF physics (P_d, P_fa, SNR/amplitude threshold).
   Emits standard CASS-EW Observation contract.
   Updates environment clock monotonically.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set

from data.tsrd_adapter import TSRDAdapter, ObservablePDW, GroundTruthPDW
from data.band_map import BandMap
from data.time_normalization import TimeNormalizer


@dataclass(frozen=True)
class DatasetObservation:
    """
    Standard CASS-EW observation contract emitted to the scheduler.
    CRITICAL: Contains NO emitter labels or ground-truth future state.
    """
    receiver_id: str
    band: int
    frequency_range_mhz: Tuple[float, float]
    dwell_start_s: float
    dwell_end_s: float
    effective_duration_s: float
    detected: bool
    signal_present: bool            # True if valid pulse opportunity existed in dwell
    signal_strength: float          # dBm or SNR
    num_pulses_intercepted: int     # Pulses intercepted during dwell
    observation_time_s: float
    pdw_features: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class EvaluatorPulseRecord:
    """
    Quarantined evaluation record created per dwell for evaluator metrics.
    NEVER seen by the scheduler.
    """
    step_idx: int
    receiver_id: str
    band: int
    dwell_start_s: float
    dwell_end_s: float
    opportunity: bool               # Signal was physically present
    detected: bool                  # Receiver reported detection
    is_hit: bool                    # opportunity AND detected
    is_miss: bool                   # opportunity AND NOT detected
    is_false_alarm: bool            # NOT opportunity AND detected
    underlying_emitter_labels: List[Any] = field(default_factory=list)


class DatasetRFEnvironment:
    """
    Causal RF environment backed by indexed pulses from a validated TSRD dataset.
    """

    def __init__(self, adapter: TSRDAdapter, band_map: Optional[BandMap] = None, seed: Optional[int] = None):
        self.adapter = adapter
        self.report = adapter.get_report()
        self.band_map = band_map or BandMap.from_tsrd_receiver(self.report.receiver_metadata)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Load and index pulses
        self._load_and_index()

        self.time_s = self.min_time_s
        self.step_count = 0

    def _load_and_index(self):
        """Loads pulses and sorts them into efficient contiguous arrays."""
        obs_pulses = list(self.adapter.iter_observations())
        gt_pulses = list(self.adapter.iter_ground_truth())

        if len(obs_pulses) != len(gt_pulses):
            raise ValueError("Observation and Ground Truth stream length mismatch!")

        self.num_pulses = len(obs_pulses)
        if self.num_pulses == 0:
            raise ValueError("DatasetRFEnvironment cannot be created from empty dataset.")

        self.timestamps_s = np.array([p.timestamp_s for p in obs_pulses], dtype=np.float64)
        self.frequencies_mhz = np.array([p.frequency_mhz for p in obs_pulses], dtype=np.float32)
        self.pulse_widths_us = np.array([p.pulse_width_us for p in obs_pulses], dtype=np.float32)
        self.aoas_deg = np.array([p.angle_of_arrival_deg for p in obs_pulses], dtype=np.float32)
        self.amplitudes_dbm = np.array([p.amplitude_dbm for p in obs_pulses], dtype=np.float32)
        self.emitter_labels = [gt.emitter_label for gt in gt_pulses]

        self.min_time_s = float(self.timestamps_s[0])
        self.max_time_s = float(self.timestamps_s[-1])

    def reset(self, seed: Optional[int] = None):
        """Resets the environment clock to dataset start."""
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        self.time_s = self.min_time_s
        self.step_count = 0

    def get_pulses_in_window(
        self,
        start_time_s: float,
        end_time_s: float,
        freq_low_mhz: float,
        freq_high_mhz: float
    ) -> Tuple[List[int], List[float], List[float], List[float], List[float], List[float], List[Any]]:
        """
        Retrieves all physical pulses in [start_time_s, end_time_s] within [freq_low_mhz, freq_high_mhz].
        Returns (indices, times, freqs, pws, aoas, amps, labels).
        """
        # Binary search for start and end indices using monotonic timestamps
        idx_start = np.searchsorted(self.timestamps_s, start_time_s, side='left')
        idx_end = np.searchsorted(self.timestamps_s, end_time_s, side='right')

        if idx_start >= idx_end:
            return [], [], [], [], [], [], []

        window_freqs = self.frequencies_mhz[idx_start:idx_end]
        mask = (window_freqs >= freq_low_mhz) & (window_freqs <= freq_high_mhz)
        matching_sub_indices = np.where(mask)[0]

        if len(matching_sub_indices) == 0:
            return [], [], [], [], [], [], []

        abs_indices = idx_start + matching_sub_indices
        times = self.timestamps_s[abs_indices].tolist()
        freqs = self.frequencies_mhz[abs_indices].tolist()
        pws = self.pulse_widths_us[abs_indices].tolist()
        aoas = self.aoas_deg[abs_indices].tolist()
        amps = self.amplitudes_dbm[abs_indices].tolist()
        labels = [self.emitter_labels[i] for i in abs_indices]

        return abs_indices.tolist(), times, freqs, pws, aoas, amps, labels


class DatasetVirtualReceiver:
    """
    Virtual receiver operating over DatasetRFEnvironment.
    Executes scheduler-selected dwell and returns observation.
    """

    def __init__(
        self,
        env: DatasetRFEnvironment,
        receiver_id: str = "RX_0",
        p_d_base: float = 0.95,
        p_fa_base: float = 0.01,
        snr_threshold_dbm: float = -105.0,
        switching_cost_s: float = 0.0001,  # 100 us retune latency
        seed: Optional[int] = None
    ):
        self.env = env
        self.receiver_id = receiver_id
        self.p_d_base = p_d_base
        self.p_fa_base = p_fa_base
        self.snr_threshold_dbm = snr_threshold_dbm
        self.switching_cost_s = switching_cost_s
        self.current_band: Optional[int] = None
        self.rng = np.random.default_rng(seed if seed is not None else env.seed)

        self.last_evaluator_record: Optional[EvaluatorPulseRecord] = None

    def execute_dwell(self, band: int, dwell_s: float) -> DatasetObservation:
        """
        Executes a scan decision on `band` for `dwell_s` seconds.
        Advances environment clock causally.
        """
        dwell_duration = TimeNormalizer.normalize_dwell(dwell_s)
        freq_low, freq_high = self.env.band_map.band_to_range(band)

        # Account for switching/settling latency if changing bands
        switch_latency = self.switching_cost_s if (self.current_band is not None and self.current_band != band) else 0.0
        self.env.time_s += switch_latency
        self.current_band = band

        dwell_start = self.env.time_s
        dwell_end = dwell_start + dwell_duration
        self.env.time_s = dwell_end
        self.env.step_count += 1

        # Query pulses physically existing in this window
        indices, times, freqs, pws, aoas, amps, labels = self.env.get_pulses_in_window(
            start_time_s=dwell_start,
            end_time_s=dwell_end,
            freq_low_mhz=freq_low,
            freq_high_mhz=freq_high
        )

        signal_present = len(indices) > 0
        detected = False
        peak_amp = -120.0
        pdw_stats = None

        if signal_present:
            peak_amp = float(max(amps))
            # Probability of detection: combines base P_d with threshold margin
            if peak_amp >= self.snr_threshold_dbm:
                detected = bool(self.rng.uniform() < self.p_d_base)
            else:
                # Signal below receiver threshold: degraded detection
                margin = max(0.01, self.p_d_base * np.exp((peak_amp - self.snr_threshold_dbm) / 10.0))
                detected = bool(self.rng.uniform() < margin)
            
            pdw_stats = {
                "mean_freq_mhz": float(np.mean(freqs)),
                "mean_pw_us": float(np.mean(pws)),
                "mean_aoa_deg": float(np.mean(aoas)),
                "peak_amp_dbm": peak_amp
            }
        else:
            # Noise-only: false alarm probability P_fa
            detected = bool(self.rng.uniform() < self.p_fa_base)
            peak_amp = -120.0 + float(self.rng.normal(0, 2.0))

        # Quarantine ground-truth evaluation record
        is_hit = bool(signal_present and detected)
        is_miss = bool(signal_present and not detected)
        is_fa = bool(not signal_present and detected)

        self.last_evaluator_record = EvaluatorPulseRecord(
            step_idx=self.env.step_count,
            receiver_id=self.receiver_id,
            band=band,
            dwell_start_s=dwell_start,
            dwell_end_s=dwell_end,
            opportunity=signal_present,
            detected=detected,
            is_hit=is_hit,
            is_miss=is_miss,
            is_false_alarm=is_fa,
            underlying_emitter_labels=labels
        )

        return DatasetObservation(
            receiver_id=self.receiver_id,
            band=band,
            frequency_range_mhz=(freq_low, freq_high),
            dwell_start_s=dwell_start,
            dwell_end_s=dwell_end,
            effective_duration_s=dwell_duration,
            detected=detected,
            signal_present=signal_present,
            signal_strength=peak_amp,
            num_pulses_intercepted=len(indices) if detected else 0,
            observation_time_s=dwell_end,
            pdw_features=pdw_stats
        )

    def get_last_evaluator_record(self) -> Optional[EvaluatorPulseRecord]:
        """Provides access to quarantined ground truth for evaluator metric calculation."""
        return self.last_evaluator_record
