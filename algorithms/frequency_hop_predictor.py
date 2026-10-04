"""
CASS-EW Observation-Driven Frequency-Hop Transition Predictor.

Learns discrete Markov transition probabilities between RF bands based strictly on
chronological detections across time steps:
    P(next_band | current_band)
Features:
- Laplace / Dirichlet smoothing to avoid zero-probability estimates
- Dynamic confidence scaling based on sample size and transition entropy
- Top-1 and Top-k predictions
- Transition history and frequency mapping
- Support for unpredictable / pseudo-random hopping: low confidence triggers exploration
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class FrequencyHopPredictor:
    """
    Online transition matrix estimator for frequency-agile and hopping emitters.
    """

    def __init__(
        self,
        num_bands: int = 10,
        alpha_smoothing: float = 0.5,
        max_hop_gap: int = 25
    ):
        self.num_bands = num_bands
        self.alpha_smoothing = alpha_smoothing
        self.max_hop_gap = max_hop_gap

        # Transition count matrix: counts[from_band][to_band]
        self.transition_counts = np.zeros((num_bands, num_bands), dtype=float)
        self.last_detection_band: Optional[int] = None
        self.last_detection_time: Optional[int] = None
        self.total_observed_transitions = 0
        self.transition_history: List[Tuple[int, int, int]] = []  # (t, from_b, to_b)

    def reset(self):
        """Reset learned transition matrix and history."""
        self.transition_counts.fill(0.0)
        self.last_detection_band = None
        self.last_detection_time = None
        self.total_observed_transitions = 0
        self.transition_history = []

    def record_detection(self, time: int, band: int):
        """
        Record a detection on a given band and update transition statistics if
        a transition occurred within max_hop_gap.
        """
        if self.last_detection_band is not None and self.last_detection_time is not None:
            gap = time - self.last_detection_time
            if 0 < gap <= self.max_hop_gap:
                from_b = self.last_detection_band
                to_b = band
                self.transition_counts[from_b, to_b] += 1.0
                self.total_observed_transitions += 1
                self.transition_history.append((time, from_b, to_b))
                if len(self.transition_history) > 100:
                    self.transition_history.pop(0)

        self.last_detection_band = band
        self.last_detection_time = time

    def get_transition_probabilities(self, from_band: int) -> np.ndarray:
        """
        Compute smoothed transition probabilities P(next_band | from_band).
        Uses Laplace smoothing: (count + alpha) / (total_from + alpha * num_bands).
        """
        counts = self.transition_counts[from_band]
        total_from = np.sum(counts)
        denom = total_from + self.alpha_smoothing * self.num_bands
        probs = (counts + self.alpha_smoothing) / denom
        return probs

    def predict_next_band(self, current_band: Optional[int] = None) -> Dict[str, Any]:
        """
        Predict the most likely next band and return confidence and distribution.

        Returns:
            dict containing:
                current_band: The reference band (or last detected band)
                top_1_band: Most probable next band
                top_1_probability: Estimated probability of top-1
                top_k_bands: List of (band, prob) sorted descending
                confidence: Reliability score [0, 1] based on sample size & entropy
                distribution: Dict of band -> probability
                transition_count: Number of transitions observed from current_band
        """
        ref_band = current_band if current_band is not None else self.last_detection_band
        if ref_band is None:
            uniform_prob = 1.0 / self.num_bands
            return {
                "current_band": None,
                "top_1_band": None,
                "top_1_probability": uniform_prob,
                "top_k_bands": [],
                "confidence": 0.0,
                "distribution": {b: uniform_prob for b in range(self.num_bands)},
                "transition_count": 0
            }

        probs = self.get_transition_probabilities(ref_band)
        counts = self.transition_counts[ref_band]
        total_from = int(np.sum(counts))

        sorted_indices = np.argsort(probs)[::-1]
        top_1 = int(sorted_indices[0])
        top_1_prob = float(probs[top_1])

        # Confidence factors:
        # 1. Sample size: saturates at 5 observations from this band
        sample_factor = min(1.0, total_from / 5.0)

        # 2. Peakiness / Entropy: relative to uniform distribution
        # Normalized Shannon entropy in [0, 1]
        p_clean = np.clip(probs, 1e-6, 1.0)
        norm_entropy = -np.sum(p_clean * np.log2(p_clean)) / np.log2(self.num_bands)
        predictability_factor = max(0.0, 1.0 - norm_entropy)

        confidence = float(sample_factor * predictability_factor)

        top_k = [(int(idx), float(probs[idx])) for idx in sorted_indices[:3]]

        return {
            "current_band": ref_band,
            "top_1_band": top_1,
            "top_1_probability": top_1_prob,
            "top_k_bands": top_k,
            "confidence": confidence,
            "distribution": {b: float(probs[b]) for b in range(self.num_bands)},
            "transition_count": total_from
        }

    def evaluate_hop_score(self, current_band: Optional[int] = None) -> np.ndarray:
        """
        Return vector of shape (num_bands,) representing hop priority scores
        weighted by transition probabilities and confidence.
        """
        pred = self.predict_next_band(current_band)
        conf = pred["confidence"]
        scores = np.zeros(self.num_bands)
        if conf <= 0.05:
            return scores  # No strong transition evidence

        dist = pred["distribution"]
        for b in range(self.num_bands):
            scores[b] = dist[b] * conf
        return scores

    def evaluate_prediction_quality(self, test_trace: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Evaluate top-1 and top-k prediction accuracy over a test trace.

        Args:
            test_trace: List of (time, band) detections in chronological order.

        Returns:
            dict with top_1_accuracy, top_3_accuracy, prediction_coverage,
            total_transitions, correct_top1, correct_top3.
        """
        correct_top1 = 0
        correct_top3 = 0
        total_transitions = 0

        prev_band = None
        prev_time = None
        for t, band in test_trace:
            if prev_band is not None and prev_time is not None:
                gap = t - prev_time
                if 0 < gap <= self.max_hop_gap:
                    total_transitions += 1
                    pred = self.predict_next_band(prev_band)
                    if pred["top_1_band"] == band:
                        correct_top1 += 1
                    top_k_bands = [b for b, _ in pred["top_k_bands"]]
                    if band in top_k_bands:
                        correct_top3 += 1
            prev_band = band
            prev_time = t

        top1_acc = correct_top1 / max(1, total_transitions)
        top3_acc = correct_top3 / max(1, total_transitions)

        # Coverage: fraction of bands with at least one observed outgoing transition
        bands_with_data = sum(1 for b in range(self.num_bands) if np.sum(self.transition_counts[b]) > 0)
        coverage = bands_with_data / max(1, self.num_bands)

        return {
            "top_1_accuracy": float(top1_acc),
            "top_3_accuracy": float(top3_acc),
            "prediction_coverage": float(coverage),
            "total_transitions": total_transitions,
            "correct_top1": correct_top1,
            "correct_top3": correct_top3
        }

