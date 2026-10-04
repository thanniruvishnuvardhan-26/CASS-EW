"""
CASS-EW Standalone RF Environment-Change Detector.

Detects structural changes in the operational RF spectrum using strictly observable scan results:
- NEW_ACTIVITY: Previously quiet or low-belief band suddenly exhibits repeated detections.
- EMITTER_DISAPPEARANCE: Previously active/predictable emitter ceases detections for extended duration.
- PERIODICITY_CHANGE: Rhythm drift or interval variance surge on a periodic band.
- HOPPING_CHANGE: Emergence of unexpected transitions violating learned Markov patterns.
- SPECTRUM_DRIFT: Broad distribution shift across multiple bands.

Emits structured events without calling it "threat detection" (strictly RF environment change).
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class EnvironmentChangeEvent:
    """Structured record of an observed RF environment change."""
    time: int
    change_type: str  # 'NEW_ACTIVITY', 'EMITTER_DISAPPEARANCE', 'PERIODICITY_CHANGE', 'HOPPING_CHANGE', 'SPECTRUM_DRIFT'
    band: Optional[int]
    severity: str     # 'LOW', 'MEDIUM', 'HIGH'
    confidence: float # [0.0, 1.0]
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "change_type": self.change_type,
            "band": self.band,
            "severity": self.severity,
            "confidence": float(self.confidence),
            "description": self.description
        }


class RFEnvironmentChangeDetector:
    """
    Monitors online observation streams to flag significant RF changes.
    """

    def __init__(
        self,
        num_bands: int = 10,
        activity_burst_threshold: int = 2,
        quiet_streak_threshold: int = 40,
        drift_window_size: int = 50,
        hopping_surprise_threshold: float = 0.05
    ):
        self.num_bands = num_bands
        self.activity_burst_threshold = activity_burst_threshold
        self.quiet_streak_threshold = quiet_streak_threshold
        self.drift_window_size = drift_window_size
        self.hopping_surprise_threshold = hopping_surprise_threshold

        self.band_consecutive_hits = np.zeros(num_bands, dtype=int)
        self.band_consecutive_misses = np.zeros(num_bands, dtype=int)
        self.band_historical_activity = np.full(num_bands, 0.1)  # baseline activity estimate
        self.band_last_seen = np.full(num_bands, -1, dtype=int)
        self.band_was_considered_active = np.zeros(num_bands, dtype=bool)

        # Hopping change detection: track recent transition history
        self._last_detected_band: Optional[int] = None
        self._transition_counts = np.zeros((num_bands, num_bands), dtype=float)
        self._total_transitions = 0

        # Spectrum drift detection: sliding window of per-band detection counts
        self._drift_history: List[np.ndarray] = []

        self.recent_events: List[EnvironmentChangeEvent] = []

    def reset(self):
        """Reset internal detector state."""
        self.band_consecutive_hits.fill(0)
        self.band_consecutive_misses.fill(0)
        self.band_historical_activity.fill(0.1)
        self.band_last_seen.fill(-1)
        self.band_was_considered_active.fill(False)
        self._last_detected_band = None
        self._transition_counts.fill(0.0)
        self._total_transitions = 0
        self._drift_history = []
        self.recent_events = []

    def process_observation(
        self,
        time: int,
        band: int,
        detected: bool,
        current_belief: float = 0.5,
        predicted_detection: bool = False
    ) -> List[EnvironmentChangeEvent]:
        """
        Process an observation and return any newly triggered change events.
        """
        events: List[EnvironmentChangeEvent] = []

        if detected:
            self.band_consecutive_hits[band] += 1
            self.band_consecutive_misses[band] = 0
            self.band_last_seen[band] = time

            # 1. NEW_ACTIVITY Detection
            # If historical activity was very low (<0.15) and we get consecutive hits
            if not self.band_was_considered_active[band]:
                if self.band_consecutive_hits[band] >= self.activity_burst_threshold:
                    self.band_was_considered_active[band] = True
                    conf = min(0.95, 0.6 + 0.15 * self.band_consecutive_hits[band])
                    ev = EnvironmentChangeEvent(
                        time=time,
                        change_type="NEW_ACTIVITY",
                        band=band,
                        severity="HIGH" if self.band_historical_activity[band] < 0.05 else "MEDIUM",
                        confidence=conf,
                        description=f"New sustained signal activity detected on previously quiet Band {band}."
                    )
                    events.append(ev)

            # 4. HOPPING_CHANGE Detection
            # If we have a learned transition model and this transition is surprising
            if self._last_detected_band is not None and self._last_detected_band != band:
                from_b = self._last_detected_band
                self._transition_counts[from_b, band] += 1.0
                self._total_transitions += 1

                # Check if this transition is surprising given learned history
                if self._total_transitions >= 10:
                    row_total = float(np.sum(self._transition_counts[from_b]))
                    if row_total > 3:
                        # Smoothed probability of this transition
                        alpha = 0.5
                        trans_prob = (self._transition_counts[from_b, band] + alpha) / (row_total + alpha * self.num_bands)
                        if trans_prob < self.hopping_surprise_threshold:
                            ev = EnvironmentChangeEvent(
                                time=time,
                                change_type="HOPPING_CHANGE",
                                band=band,
                                severity="MEDIUM",
                                confidence=min(0.9, 0.5 + 0.1 * row_total),
                                description=f"Unexpected frequency transition Band {from_b} -> Band {band} (P={trans_prob:.3f}). Hopping pattern may have changed."
                            )
                            events.append(ev)

            self._last_detected_band = band

            # Smooth historical activity estimate upward
            self.band_historical_activity[band] = 0.9 * self.band_historical_activity[band] + 0.1 * 1.0

        else:
            self.band_consecutive_misses[band] += 1
            self.band_consecutive_hits[band] = 0

            # 2. EMITTER_DISAPPEARANCE Detection
            if self.band_was_considered_active[band]:
                if self.band_consecutive_misses[band] >= self.quiet_streak_threshold:
                    self.band_was_considered_active[band] = False
                    ev = EnvironmentChangeEvent(
                        time=time,
                        change_type="EMITTER_DISAPPEARANCE",
                        band=band,
                        severity="MEDIUM",
                        confidence=0.85,
                        description=f"Active signal on Band {band} ceased detections for {self.band_consecutive_misses[band]} scans."
                    )
                    events.append(ev)

            # 3. PERIODICITY_CHANGE Detection
            # If a detection was strongly predicted but missed multiple times in a row
            if predicted_detection and self.band_consecutive_misses[band] >= 3:
                ev = EnvironmentChangeEvent(
                    time=time,
                    change_type="PERIODICITY_CHANGE",
                    band=band,
                    severity="LOW",
                    confidence=0.75,
                    description=f"Expected periodic arrival on Band {band} missed repeatedly. Rhythm drift likely."
                )
                events.append(ev)

            # Smooth historical activity estimate downward
            self.band_historical_activity[band] = 0.95 * self.band_historical_activity[band]

        # 5. SPECTRUM_DRIFT Detection
        # Track per-observation band detection in sliding window
        obs_vec = np.zeros(self.num_bands)
        if detected:
            obs_vec[band] = 1.0
        self._drift_history.append(obs_vec)
        if len(self._drift_history) > self.drift_window_size * 2:
            self._drift_history = self._drift_history[-self.drift_window_size * 2:]

        if len(self._drift_history) >= self.drift_window_size * 2:
            half = self.drift_window_size
            old_window = np.sum(self._drift_history[:half], axis=0)
            new_window = np.sum(self._drift_history[half:], axis=0)
            old_total = max(1.0, float(np.sum(old_window)))
            new_total = max(1.0, float(np.sum(new_window)))
            old_dist = old_window / old_total
            new_dist = new_window / new_total

            # Jensen-Shannon divergence (numerically stable)
            eps = 1e-10
            old_dist_safe = old_dist + eps
            new_dist_safe = new_dist + eps
            m = 0.5 * (old_dist_safe + new_dist_safe)
            kl_old = float(np.sum(old_dist_safe * np.log(old_dist_safe / m)))
            kl_new = float(np.sum(new_dist_safe * np.log(new_dist_safe / m)))
            js_div = 0.5 * (kl_old + kl_new)

            if js_div > 0.3:
                # Significant distribution shift
                shifting_bands = np.where(np.abs(new_dist - old_dist) > 0.1)[0]
                ev = EnvironmentChangeEvent(
                    time=time,
                    change_type="SPECTRUM_DRIFT",
                    band=None,
                    severity="HIGH" if js_div > 0.5 else "MEDIUM",
                    confidence=min(0.95, float(js_div)),
                    description=f"Broad activity distribution shift detected (JS divergence={js_div:.3f}). Affected bands: {list(shifting_bands)}."
                )
                events.append(ev)

        for ev in events:
            self.recent_events.append(ev)
            if len(self.recent_events) > 50:
                self.recent_events.pop(0)

        return events

    def get_latest_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.recent_events[-limit:]]
