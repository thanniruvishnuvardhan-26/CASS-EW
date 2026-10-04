import numpy as np
from typing import Dict, List, Optional, Tuple

class TemporalModel:
    """
    Explicit, interpretable temporal model for Phase 4.
    Learns temporal activity patterns from receiver observations.
    """
    def __init__(
        self,
        num_bands: int,
        history_size: int = 10,
        minimum_detections: int = 3,
        period_window_size: int = 5,
        temporal_tolerance: int = 2
    ):
        self.num_bands = num_bands
        self.history_size = history_size
        self.minimum_detections = minimum_detections
        self.period_window_size = period_window_size
        self.temporal_tolerance = temporal_tolerance
        
        self.reset()
        
    def reset(self):
        # Store lists of (observation_time, detection_result)
        self.history: Dict[int, List[Tuple[int, bool]]] = {b: [] for b in range(self.num_bands)}
        
    def update(self, band: int, time: int, detection: bool):
        self.history[band].append((time, detection))
        if len(self.history[band]) > self.history_size:
            self.history[band].pop(0)
            
    def get_detection_times(self, band: int) -> List[int]:
        return [t for t, d in self.history[band] if d]
        
    def estimate_period(self, band: int) -> Tuple[Optional[float], int, Optional[float]]:
        """
        Calculates simple rhythm estimator using median of recent intervals.
        Returns:
            estimated_period (float or None)
            number_of_supporting_intervals (int)
            interval_variability (float or None, measured as MAD)
        """
        det_times = self.get_detection_times(band)
        if len(det_times) < self.minimum_detections:
            return None, 0, None
            
        # Calculate intervals between detections
        intervals = [det_times[i] - det_times[i-1] for i in range(1, len(det_times))]
        
        # Use recent window
        recent_intervals = intervals[-self.period_window_size:]
        
        estimated_period = float(np.median(recent_intervals))
        
        # Variability (MAD = Median Absolute Deviation)
        mad = float(np.median(np.abs(np.array(recent_intervals) - estimated_period)))
        
        return estimated_period, len(recent_intervals), mad
        
    def temporal_score(self, band: int, current_time: int) -> float:
        """
        Creates an explicit temporal score in [0, 1].
        Neutral (no evidence) = 0.5.
        High = 1.0 (within tolerance).
        Low = decreases linearly outside tolerance.
        """
        period, _, mad = self.estimate_period(band)
        
        if period is None or period <= 0:
            return 0.5  # Neutral score for insufficient history
            
        det_times = self.get_detection_times(band)
        t_last = det_times[-1]
        
        elapsed = current_time - t_last
        
        # Distance to nearest expected multiple of period
        nearest_multiple = round(elapsed / period) * period
        distance = abs(elapsed - nearest_multiple)
        
        if distance <= self.temporal_tolerance:
            return 1.0
        else:
            # Linear decay outside tolerance, drops to 0 at 2 * tolerance + 1
            decay = (distance - self.temporal_tolerance) / max(1.0, float(self.temporal_tolerance))
            return max(0.0, 1.0 - decay)
