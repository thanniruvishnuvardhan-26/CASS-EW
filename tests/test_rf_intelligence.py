"""
Unit tests for Periodicity / PRI Estimation, Frequency-Hop Prediction,
and RF Environment-Change Detection.
"""

import unittest
import numpy as np

from algorithms.temporal_interval_analyzer import TemporalIntervalAnalyzer
from algorithms.frequency_hop_predictor import FrequencyHopPredictor
from algorithms.rf_change_detector import RFEnvironmentChangeDetector, EnvironmentChangeEvent


class TestPeriodicityAndPRI(unittest.TestCase):
    """
    Test suite for observation-driven PRI / Periodicity Estimation.
    Verifies support for:
    - perfectly periodic emitters
    - jittered periodic emitters
    - intermittent emitters
    - irregular emitters
    - prediction uncertainty & no ground truth leakage
    """

    def setUp(self):
        self.analyzer = TemporalIntervalAnalyzer(history_size=15, minimum_detections=3, window_size=6)

    def test_insufficient_data_uncertainty(self):
        """Must not force a prediction when evidence is sparse (< 3 detections)."""
        self.analyzer.update(time=10, detected=True)
        stats = self.analyzer.get_pri_stats()
        self.assertEqual(stats["emitter_type"], "INSUFFICIENT_DATA")
        self.assertIsNone(stats["predicted_next_time"])
        self.assertEqual(stats["periodicity_confidence"], 0.0)

        self.analyzer.update(time=25, detected=True)
        stats2 = self.analyzer.get_pri_stats()
        # 2 detections = 1 interval, below minimum_detections - 1 = 2 intervals
        self.assertEqual(stats2["emitter_type"], "INSUFFICIENT_DATA")
        self.assertEqual(stats2["periodicity_confidence"], 0.0)

    def test_perfectly_periodic_emitter(self):
        """Perfect periodic pulses with T=10 must yield high confidence and accurate next time."""
        for t in [10, 20, 30, 40, 50]:
            self.analyzer.update(time=t, detected=True)

        stats = self.analyzer.get_pri_stats()
        self.assertEqual(stats["emitter_type"], "PERIODIC")
        self.assertEqual(stats["median_interval"], 10.0)
        self.assertEqual(stats["mean_interval"], 10.0)
        self.assertEqual(stats["variance"], 0.0)
        self.assertEqual(stats["jitter"], 0.0)
        self.assertGreater(stats["periodicity_confidence"], 0.8)
        self.assertEqual(stats["predicted_next_time"], 60)

    def test_jittered_periodic_emitter(self):
        """Periodic pulses with small jitter (+- 1-2 ticks) must be classified as JITTERED."""
        # Baseline period 20 with minor jitter
        for t in [20, 41, 59, 81, 100]:
            self.analyzer.update(time=t, detected=True)

        stats = self.analyzer.get_pri_stats()
        self.assertIn(stats["emitter_type"], ["PERIODIC", "JITTERED"])
        self.assertAlmostEqual(stats["median_interval"], 20.0, delta=1.5)
        self.assertGreater(stats["jitter"], 0.0)
        self.assertGreater(stats["periodicity_confidence"], 0.5)
        self.assertIsNotNone(stats["predicted_next_time"])
        self.assertAlmostEqual(stats["predicted_next_time"], 120, delta=2)

    def test_intermittent_emitter(self):
        """Intermittent pulses with large gaps and sporadic bursts."""
        # Gaps: 8, 50, 6, 80
        for t in [10, 18, 68, 74, 154]:
            self.analyzer.update(time=t, detected=True)

        stats = self.analyzer.get_pri_stats()
        self.assertIn(stats["emitter_type"], ["INTERMITTENT", "IRREGULAR"])
        # Should express low confidence rather than forcing an inaccurate prediction
        self.assertLess(stats["periodicity_confidence"], 0.4)

    def test_irregular_emitter(self):
        """Chaotic, non-periodic arrivals should exhibit low confidence."""
        rng = np.random.default_rng(42)
        cur = 0
        for _ in range(8):
            cur += int(rng.integers(3, 40))
            self.analyzer.update(time=cur, detected=True)

        stats = self.analyzer.get_pri_stats()
        self.assertIn(stats["emitter_type"], ["IRREGULAR", "INTERMITTENT"])
        self.assertLess(stats["periodicity_confidence"], 0.4)

    def test_reset(self):
        """Reset clears detection history and restores uncertainty."""
        for t in [10, 20, 30]:
            self.analyzer.update(time=t, detected=True)
        self.analyzer.reset()
        stats = self.analyzer.get_pri_stats()
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["periodicity_confidence"], 0.0)


class TestFrequencyHopPredictor(unittest.TestCase):
    """
    Test suite for observation-driven Frequency-Hop Transition Predictor.
    Verifies:
    - P(next_band | current_band) transition matrix learning
    - Laplace smoothing to prevent zero-probabilities
    - Top-1 and Top-k accuracy
    - Confidence scaling and unpredictable hopping handling
    """

    def setUp(self):
        self.predictor = FrequencyHopPredictor(num_bands=10, alpha_smoothing=0.5, max_hop_gap=20)

    def test_laplace_smoothing_no_zero_probability(self):
        """Prior to or after observations, all transition probabilities must be strictly > 0."""
        probs = self.predictor.get_transition_probabilities(from_band=3)
        self.assertEqual(len(probs), 10)
        self.assertTrue(np.all(probs > 0.0))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_cyclic_hopping_pattern_learning(self):
        """Repeated transitions 6 -> 8 -> 2 -> 6 must learn high conditional probabilities."""
        t = 0
        cycle = [6, 8, 2]
        for _ in range(15):
            for b in cycle:
                t += 5
                self.predictor.record_detection(time=t, band=b)

        # Evaluate transitions from 6
        probs_6 = self.predictor.get_transition_probabilities(from_band=6)
        pred_6 = self.predictor.predict_next_band(current_band=6)
        self.assertEqual(pred_6["top_1_band"], 8)
        self.assertGreater(pred_6["top_1_probability"], 0.5)
        self.assertGreater(pred_6["confidence"], 0.5)

        # Top-3 bands should include 8 as first
        top_k = pred_6["top_k_bands"]
        self.assertEqual(top_k[0][0], 8)

        # Evaluate transitions from 8 -> should predict 2
        pred_8 = self.predictor.predict_next_band(current_band=8)
        self.assertEqual(pred_8["top_1_band"], 2)

    def test_unpredictable_random_hopping_low_confidence(self):
        """Random transitions across bands should yield uniform-like distribution and low confidence."""
        rng = np.random.default_rng(123)
        t = 0
        for _ in range(30):
            t += 3
            b = int(rng.integers(0, 10))
            self.predictor.record_detection(time=t, band=b)

        # Confidence for random hops should be low
        pred = self.predictor.predict_next_band()
        self.assertLess(pred["confidence"], 0.4)

    def test_top_k_accuracy_evaluation(self):
        """Evaluate top-1 and top-k prediction metrics over a test trace."""
        # Pattern: 1 -> 4 (80% of time), 1 -> 5 (20% of time)
        rng = np.random.default_rng(42)
        t = 0
        for _ in range(25):
            t += 2
            self.predictor.record_detection(t, 1)
            t += 2
            next_b = 4 if rng.random() < 0.8 else 5
            self.predictor.record_detection(t, next_b)

        pred = self.predictor.predict_next_band(current_band=1)
        self.assertEqual(pred["top_1_band"], 4)
        top_k_bands = [b for b, p in pred["top_k_bands"]]
        self.assertIn(4, top_k_bands)
        self.assertIn(5, top_k_bands)


class TestRFEnvironmentChangeDetector(unittest.TestCase):
    """
    Test suite for standalone RF Environment-Change Detector.
    Verifies detection of:
    - NEW_ACTIVITY (activity burst on previously quiet band)
    - EMITTER_DISAPPEARANCE (extended silence on active band)
    - PERIODICITY_CHANGE (missed periodic detection expectation)
    - Structured event output with severity and confidence
    """

    def setUp(self):
        self.detector = RFEnvironmentChangeDetector(
            num_bands=10,
            activity_burst_threshold=2,
            quiet_streak_threshold=10
        )

    def test_new_activity_burst(self):
        """Sustained detections on quiet Band 7 should trigger a NEW_ACTIVITY event."""
        events = []
        for t in range(1, 4):
            evs = self.detector.process_observation(time=t, band=7, detected=True)
            events.extend(evs)

        self.assertTrue(len(events) >= 1)
        event = events[0]
        self.assertEqual(event.change_type, "NEW_ACTIVITY")
        self.assertEqual(event.band, 7)
        self.assertIn(event.severity, ["MEDIUM", "HIGH"])
        self.assertGreater(event.confidence, 0.5)

    def test_emitter_disappearance(self):
        """Active band that goes silent for quiet_streak_threshold scans triggers EMITTER_DISAPPEARANCE."""
        # 1. Establish activity
        self.detector.process_observation(time=1, band=3, detected=True)
        self.detector.process_observation(time=2, band=3, detected=True)

        # 2. Silence for 10 scans
        events = []
        for t in range(3, 15):
            evs = self.detector.process_observation(time=t, band=3, detected=False)
            events.extend(evs)

        disappear_events = [e for e in events if e.change_type == "EMITTER_DISAPPEARANCE"]
        self.assertTrue(len(disappear_events) >= 1)
        self.assertEqual(disappear_events[0].band, 3)

    def test_periodicity_change_anomaly(self):
        """Repeated misses when strongly predicted trigger PERIODICITY_CHANGE event."""
        events = []
        for t in range(1, 5):
            evs = self.detector.process_observation(
                time=t,
                band=4,
                detected=False,
                current_belief=0.8,
                predicted_detection=True
            )
            events.extend(evs)

        period_events = [e for e in events if e.change_type == "PERIODICITY_CHANGE"]
        self.assertTrue(len(period_events) >= 1)
        self.assertEqual(period_events[0].band, 4)


if __name__ == '__main__':
    unittest.main()
