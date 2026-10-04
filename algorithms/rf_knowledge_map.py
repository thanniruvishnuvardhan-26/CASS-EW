"""
Module: RF Knowledge Map (algorithms/rf_knowledge_map.py)
CASS-EW Cognitive Adaptive Smart Scan for Electronic Warfare

Exposes the consolidated cognitive state of the RF spectrum across all
receivers and bands, strictly derived from historical observations and causal
reasoning (NO hidden ground truth access).

Each (receiver, band) entry tracks:
- belief: Posterior probability of emitter presence [0, 1]
- uncertainty: Information-theoretic uncertainty [0, 1]
- last_observation: Time of most recent scan
- last_detection: Time of most recent true detection
- last_visit: Step/time receiver visited this band
- staleness: Elapsed time since last scan
- starvation_count: Number of scheduling opportunities passed without visiting
- temporal_score: Periodic rhythm alignment [0, 1]
- prediction_score: Next-detection window proximity [0, 1]
- prediction_confidence: Reliability of prediction [0, 1]
- pattern_score: Cross-band harmonic / coincidence support [0, 1]
- spatial_evidence: Cross-receiver spatial consistency [0, 1]
- activity_state: Inferred operational state ('ACTIVE', 'QUIET', 'SUSPECTED', 'UNKNOWN')
- environment_change: Detected drift or rhythm anomaly flag
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class BandKnowledgeEntry:
    """Consolidated cognitive state for a single (receiver, band) pair."""
    receiver_id: str
    band: int
    belief: float = 0.5
    uncertainty: float = 1.0
    last_observation: int = -1
    last_detection: int = -1
    last_visit: int = -1
    staleness: int = 0
    starvation_count: int = 0
    temporal_score: float = 0.5
    prediction_score: float = 0.5
    prediction_confidence: float = 0.0
    pattern_score: float = 0.0
    spatial_evidence: float = 0.0
    activity_state: str = "UNKNOWN"
    environment_change: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receiver_id": self.receiver_id,
            "band": self.band,
            "belief": float(self.belief),
            "uncertainty": float(self.uncertainty),
            "last_observation": int(self.last_observation),
            "last_detection": int(self.last_detection),
            "last_visit": int(self.last_visit),
            "staleness": int(self.staleness),
            "starvation_count": int(self.starvation_count),
            "temporal_score": float(self.temporal_score),
            "prediction_score": float(self.prediction_score),
            "prediction_confidence": float(self.prediction_confidence),
            "pattern_score": float(self.pattern_score),
            "spatial_evidence": float(self.spatial_evidence),
            "activity_state": str(self.activity_state),
            "environment_change": bool(self.environment_change)
        }


class RFKnowledgeMap:
    """
    Maintains an auditable, strictly causal cognitive map of all RF bands
    across all monitored receivers.
    """

    def __init__(
        self,
        receiver_ids: List[str],
        num_bands: int = 10,
        uncertainty_decay_rate: float = 0.02,
        max_revisit_threshold: int = 40
    ):
        self.receiver_ids = list(receiver_ids)
        self.num_bands = num_bands
        self.uncertainty_decay_rate = uncertainty_decay_rate
        self.max_revisit_threshold = max_revisit_threshold
        self.current_time = 0

        self.map: Dict[str, List[BandKnowledgeEntry]] = {
            rid: [BandKnowledgeEntry(receiver_id=rid, band=b) for b in range(num_bands)]
            for rid in self.receiver_ids
        }

    def reset(self):
        """Reset the knowledge map to initial unvisited state."""
        self.current_time = 0
        self.map = {
            rid: [BandKnowledgeEntry(receiver_id=rid, band=b) for b in range(self.num_bands)]
            for rid in self.receiver_ids
        }

    def update_staleness(self, current_time: int):
        """
        Advance internal time and update staleness and starvation for all bands.
        """
        self.current_time = current_time
        for rid in self.receiver_ids:
            for b in range(self.num_bands):
                entry = self.map[rid][b]
                if entry.last_observation >= 0:
                    entry.staleness = max(0, current_time - entry.last_observation)
                else:
                    entry.staleness = current_time

                # Dynamic uncertainty grows as observations grow stale
                entry.uncertainty = self.compute_uncertainty(entry)

    def compute_uncertainty(self, entry: BandKnowledgeEntry) -> float:
        """
        Principled uncertainty calculation:
        1. Base uncertainty from belief entropy (beliefs near 0.5 have high entropy).
        2. Staleness penalty: staleness increases epistemic uncertainty toward 1.0.
        3. Inverse confidence: high prediction confidence reduces uncertainty.
        """
        # Entropy-based belief uncertainty: 1.0 at 0.5, 0.0 at 0.0 or 1.0
        p = np.clip(entry.belief, 1e-4, 1.0 - 1e-4)
        entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))  # in [0, 1]

        # Time-based decay: uncertainty grows toward 1 as time without observation elapses
        staleness_factor = 1.0 - np.exp(-self.uncertainty_decay_rate * entry.staleness)

        # Confidence mitigation
        conf_mitigation = (1.0 - entry.prediction_confidence) if entry.prediction_confidence > 0 else 1.0

        combined = 0.5 * entropy + 0.5 * staleness_factor
        return float(np.clip(combined * conf_mitigation + 0.1 * staleness_factor, 0.0, 1.0))

    def record_scan_result(
        self,
        receiver_id: str,
        band: int,
        obs_time: int,
        detected: bool,
        belief: float,
        temporal_score: float = 0.5,
        prediction_score: float = 0.5,
        prediction_confidence: float = 0.0,
        pattern_score: float = 0.0,
        spatial_evidence: float = 0.0
    ):
        """
        Record a scan event and integrate evidence into the knowledge entry.
        """
        if receiver_id not in self.map:
            return

        self.current_time = obs_time
        entry = self.map[receiver_id][band]

        entry.last_observation = obs_time
        entry.last_visit = obs_time
        entry.staleness = 0
        entry.starvation_count = 0  # reset starvation upon visit

        if detected:
            entry.last_detection = obs_time

        entry.belief = belief
        entry.temporal_score = temporal_score
        entry.prediction_score = prediction_score
        entry.prediction_confidence = prediction_confidence
        entry.pattern_score = pattern_score
        entry.spatial_evidence = spatial_evidence

        # Update activity state inference
        if detected:
            entry.activity_state = "ACTIVE"
        elif entry.belief > 0.7 or (prediction_score > 0.8 and prediction_confidence > 0.6):
            entry.activity_state = "SUSPECTED"
        elif entry.staleness > self.max_revisit_threshold:
            entry.activity_state = "UNKNOWN"
        else:
            entry.activity_state = "QUIET"

        # Check for environment change / rhythm shift
        if not detected and prediction_score > 0.85 and prediction_confidence > 0.7:
            entry.environment_change = True
        else:
            entry.environment_change = False

        # Recompute uncertainty
        entry.uncertainty = self.compute_uncertainty(entry)

        # Update staleness and starvation across all bands
        self.update_staleness(obs_time)
        entry.staleness = 0
        entry.starvation_count = 0  # reset starvation upon visit

        # Increment starvation count for all other unvisited bands
        for r in self.receiver_ids:
            for b in range(self.num_bands):
                if not (r == receiver_id and b == band):
                    self.map[r][b].starvation_count += 1

    def get_entry(self, receiver_id: str, band: int) -> BandKnowledgeEntry:
        return self.map[receiver_id][band]

    def get_all_entries(self) -> List[BandKnowledgeEntry]:
        entries = []
        for rid in self.receiver_ids:
            entries.extend(self.map[rid])
        return entries

    def get_max_staleness(self) -> int:
        return max(entry.staleness for entry in self.get_all_entries())

    def get_starving_bands(self, threshold: Optional[int] = None) -> List[Tuple[str, int]]:
        """Return list of (receiver, band) pairs exceeding starvation threshold."""
        thresh = threshold if threshold is not None else self.max_revisit_threshold
        starving = []
        for rid in self.receiver_ids:
            for b in range(self.num_bands):
                if self.map[rid][b].staleness >= thresh:
                    starving.append((rid, b))
        return starving
