"""
CASS-EW Observation-Driven Temporal Interval & PRI Analyzer.

Strengthens temporal reasoning for EW spectrum scanning without ground-truth leakage:
- Inter-arrival interval tracking
- Mean, median, variance, and jitter (mean absolute deviation)
- Periodicity estimation with confidence assessment
- Support for periodic, jittered periodic, intermittent, and irregular emitters
- Principled uncertainty: explicitly outputs low confidence when data is sparse or irregular
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class TemporalIntervalAnalyzer:
    """
    Analyzes observation-driven detection timestamps for a single RF band or emitter track.
    Computes rigorous PRI / periodicity statistics and predicted next-arrival times.
    """

    def __init__(
        self,
        history_size: int = 15,
        minimum_detections: int = 3,
        window_size: int = 6,
        tolerance: int = 2,
        jitter_threshold_ratio: float = 0.35
    ):
        self.history_size = history_size
        self.minimum_detections = minimum_detections
        self.window_size = window_size
        self.tolerance = tolerance
        self.jitter_threshold_ratio = jitter_threshold_ratio

        self.detection_times: List[int] = []
        self.last_observation_time: Optional[int] = None

    def reset(self):
        """Reset historical detection records."""
        self.detection_times = []
        self.last_observation_time = None

    def update(self, time: int, detected: bool):
        """Record an observation event."""
        self.last_observation_time = time
        if detected:
            self.detection_times.append(time)
            if len(self.detection_times) > self.history_size:
                self.detection_times.pop(0)

    def get_intervals(self) -> List[int]:
        """Compute consecutive inter-arrival intervals."""
        if len(self.detection_times) < 2:
            return []
        return [
            self.detection_times[i] - self.detection_times[i - 1]
            for i in range(1, len(self.detection_times))
        ]

    def get_pri_stats(self) -> Dict[str, Any]:
        """
        Compute statistical breakdown of inter-arrival intervals.

        Returns:
            dict containing:
                count: Number of recent intervals analyzed
                intervals: List of recent intervals
                mean_interval: Average interval length
                median_interval: Robust median interval (PRI estimate)
                variance: Interval variance
                jitter: Mean absolute deviation from median (jitter metric)
                jitter_ratio: jitter / median_interval
                periodicity_estimate: Estimated period (if periodic) or None
                periodicity_confidence: Confidence score [0, 1]
                emitter_type: Inferred class ('PERIODIC', 'JITTERED', 'INTERMITTENT', 'IRREGULAR', 'INSUFFICIENT_DATA')
                predicted_next_time: Projected timestamp for next active pulse
        """
        intervals = self.get_intervals()
        if len(intervals) < (self.minimum_detections - 1):
            return {
                "count": len(intervals),
                "intervals": intervals,
                "mean_interval": None,
                "median_interval": None,
                "variance": None,
                "jitter": None,
                "jitter_ratio": None,
                "periodicity_estimate": None,
                "periodicity_confidence": 0.0,
                "emitter_type": "INSUFFICIENT_DATA",
                "predicted_next_time": None
            }

        recent = intervals[-self.window_size:]
        arr = np.array(recent, dtype=float)

        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        var_val = float(np.var(arr))
        jitter_val = float(np.mean(np.abs(arr - median_val)))
        jitter_ratio = jitter_val / max(1e-4, median_val)

        # Confidence calculation
        # 1. Sample size factor (saturates when we have at least minimum_detections - 1 intervals)
        required_intervals = max(2, self.minimum_detections - 1)
        sample_factor = min(1.0, len(recent) / float(required_intervals))
        # 2. Regularity factor (drops to 0 when jitter_ratio reaches jitter_threshold_ratio)
        regularity_factor = max(0.0, 1.0 - (jitter_ratio / self.jitter_threshold_ratio))
        confidence = float(sample_factor * regularity_factor)

        # Classification
        if confidence >= 0.7 and jitter_ratio < 0.10:
            emitter_type = "PERIODIC"
            period_est = median_val
        elif confidence >= 0.4 and jitter_ratio < self.jitter_threshold_ratio:
            emitter_type = "JITTERED"
            period_est = median_val
        elif median_val > 0 and jitter_ratio >= self.jitter_threshold_ratio:
            emitter_type = "INTERMITTENT"
            period_est = None
        else:
            emitter_type = "IRREGULAR"
            period_est = None

        # Prediction: only predict if we have positive periodicity estimate and moderate confidence
        predicted_next = None
        if period_est is not None and confidence >= 0.35 and self.detection_times:
            last_t = self.detection_times[-1]
            predicted_next = last_t + round(period_est)

        return {
            "count": len(recent),
            "intervals": [int(x) for x in recent],
            "mean_interval": mean_val,
            "median_interval": median_val,
            "variance": var_val,
            "jitter": jitter_val,
            "jitter_ratio": jitter_ratio,
            "periodicity_estimate": period_est,
            "periodicity_confidence": confidence,
            "emitter_type": emitter_type,
            "predicted_next_time": predicted_next
        }

    def evaluate_timing_score(self, current_time: int) -> Tuple[float, float]:
        """
        Return (temporal_score, confidence) for scheduling priority at current_time.
        Scores [0, 1].
        """
        stats = self.get_pri_stats()
        conf = stats["periodicity_confidence"]
        pred_time = stats["predicted_next_time"]

        if pred_time is None or conf < 0.25:
            return 0.5, 0.0

        dist = abs(current_time - pred_time)
        if dist <= self.tolerance:
            score = 1.0
        else:
            decay = (dist - self.tolerance) / max(1.0, float(self.tolerance * 2))
            score = max(0.0, 1.0 - decay)

        return float(score), float(conf)

    def evaluate_prediction_accuracy(self, test_detections: List[int]) -> Dict[str, Any]:
        """
        Evaluate prediction quality by replaying a sequence of detection times.
        Uses a rolling one-step-ahead prediction and checks against actual next detection.

        Args:
            test_detections: Sorted list of detection timestamps.

        Returns:
            dict with mean_absolute_error, hit_rate (within tolerance), total_predictions.
        """
        if len(test_detections) < self.minimum_detections + 1:
            return {
                "mean_absolute_error": None,
                "hit_rate": 0.0,
                "total_predictions": 0
            }

        # Replay and evaluate
        errors = []
        hits = 0
        total_preds = 0

        temp = TemporalIntervalAnalyzer(
            history_size=self.history_size,
            minimum_detections=self.minimum_detections,
            window_size=self.window_size,
            tolerance=self.tolerance,
            jitter_threshold_ratio=self.jitter_threshold_ratio
        )

        for i, t in enumerate(test_detections):
            if i >= self.minimum_detections:
                stats = temp.get_pri_stats()
                pred = stats["predicted_next_time"]
                if pred is not None:
                    total_preds += 1
                    err = abs(t - pred)
                    errors.append(err)
                    if err <= self.tolerance:
                        hits += 1
            temp.update(t, detected=True)

        mae = float(np.mean(errors)) if errors else None
        hit_rate = hits / max(1, total_preds)

        return {
            "mean_absolute_error": mae,
            "hit_rate": float(hit_rate),
            "total_predictions": total_preds
        }

