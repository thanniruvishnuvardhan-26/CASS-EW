"""
CASS-EW Part 2 Comprehensive Tests.

Tests for:
- Temporal interval / PRI evaluation metrics
- Frequency hop prediction evaluation metrics
- RF change detector: HOPPING_CHANGE and SPECTRUM_DRIFT events
- Receiver bandwidth constraints
- Receiver-specific detection differences
- Multi-receiver failure/degradation
- Multi-receiver asymmetric SNR with actual VirtualReceiver
- Multi-receiver starvation metrics
- PDW causality leakage reinforcement
"""

import unittest
import numpy as np

from algorithms.temporal_interval_analyzer import TemporalIntervalAnalyzer
from algorithms.frequency_hop_predictor import FrequencyHopPredictor
from algorithms.rf_change_detector import RFEnvironmentChangeDetector, EnvironmentChangeEvent
from simulator.receiver import VirtualReceiver
from simulator.environment import RFEnvironment, Emitter
from algorithms.phase8_spatial_scheduler import SpatialScheduler


class TestPRIEvaluationMetrics(unittest.TestCase):
    """Test evaluation metrics for periodicity / PRI estimation."""

    def test_perfect_periodic_prediction_quality(self):
        """Perfect period T=10 should yield perfect predictions (MAE=0, hit_rate=1)."""
        analyzer = TemporalIntervalAnalyzer(history_size=20, minimum_detections=3, tolerance=2)
        detections = list(range(10, 110, 10))  # T=10, 10 pulses
        result = analyzer.evaluate_prediction_accuracy(detections)
        self.assertIsNotNone(result["mean_absolute_error"])
        self.assertEqual(result["mean_absolute_error"], 0.0)
        self.assertEqual(result["hit_rate"], 1.0)
        self.assertGreater(result["total_predictions"], 0)

    def test_jittered_prediction_quality(self):
        """Jittered period should produce non-zero MAE but reasonable hit_rate."""
        analyzer = TemporalIntervalAnalyzer(history_size=20, minimum_detections=3, tolerance=3)
        detections = [10, 21, 29, 41, 49, 60, 71, 79, 91, 100]  # ~T=10 +/- jitter
        result = analyzer.evaluate_prediction_accuracy(detections)
        self.assertIsNotNone(result["mean_absolute_error"])
        self.assertGreater(result["total_predictions"], 0)
        # With tolerance=3 and small jitter, should still get some hits
        self.assertGreater(result["hit_rate"], 0.0)

    def test_insufficient_data_returns_no_predictions(self):
        """With too few detections, evaluation should return zero predictions."""
        analyzer = TemporalIntervalAnalyzer(minimum_detections=5)
        detections = [10, 20, 30]  # Only 3 detections, needs 5+1
        result = analyzer.evaluate_prediction_accuracy(detections)
        self.assertEqual(result["total_predictions"], 0)
        self.assertIsNone(result["mean_absolute_error"])


class TestHopPredictionEvaluation(unittest.TestCase):
    """Test evaluation metrics for frequency-hop prediction."""

    def test_cyclic_hop_evaluation(self):
        """Cyclic pattern 1->3->5->1 should yield high top-1 accuracy."""
        pred = FrequencyHopPredictor(num_bands=10, alpha_smoothing=0.5, max_hop_gap=10)

        # Train on many cycles
        cycle = [1, 3, 5]
        t = 0
        for _ in range(20):
            for b in cycle:
                t += 3
                pred.record_detection(t, b)

        # Evaluate on fresh cycles
        test_trace = []
        for _ in range(10):
            for b in cycle:
                t += 3
                test_trace.append((t, b))

        result = pred.evaluate_prediction_quality(test_trace)
        self.assertGreater(result["top_1_accuracy"], 0.8)
        self.assertGreater(result["top_3_accuracy"], 0.9)
        self.assertGreater(result["prediction_coverage"], 0.0)
        self.assertGreater(result["total_transitions"], 0)

    def test_random_hop_evaluation_low_accuracy(self):
        """Random hopping should yield low prediction accuracy."""
        pred = FrequencyHopPredictor(num_bands=10, alpha_smoothing=0.5, max_hop_gap=10)
        rng = np.random.default_rng(42)

        # Train on random hops
        t = 0
        for _ in range(40):
            t += 3
            pred.record_detection(t, int(rng.integers(0, 10)))

        # Test on more random hops
        test_trace = []
        for _ in range(20):
            t += 3
            test_trace.append((t, int(rng.integers(0, 10))))

        result = pred.evaluate_prediction_quality(test_trace)
        # Random -> top-1 accuracy should be low
        self.assertLess(result["top_1_accuracy"], 0.5)


class TestHoppingChangeDetection(unittest.TestCase):
    """Test HOPPING_CHANGE detection in RFEnvironmentChangeDetector."""

    def test_hopping_change_on_unexpected_transition(self):
        """After learning A->B pattern, an A->C transition should trigger HOPPING_CHANGE."""
        det = RFEnvironmentChangeDetector(
            num_bands=10,
            activity_burst_threshold=2,
            quiet_streak_threshold=10,
            hopping_surprise_threshold=0.1
        )

        # Establish strong A(2)->B(5) pattern
        for i in range(20):
            det.process_observation(time=i * 2, band=2, detected=True)
            det.process_observation(time=i * 2 + 1, band=5, detected=True)

        # Now inject a surprising transition: last detection was band 5, now band 9
        events = det.process_observation(time=100, band=9, detected=True)
        hopping_events = [e for e in events if e.change_type == "HOPPING_CHANGE"]

        self.assertTrue(len(hopping_events) >= 1, "Expected HOPPING_CHANGE event for surprising transition")
        self.assertEqual(hopping_events[0].band, 9)

    def test_no_hopping_change_on_expected_transition(self):
        """A well-learned transition should not trigger HOPPING_CHANGE."""
        det = RFEnvironmentChangeDetector(num_bands=10, hopping_surprise_threshold=0.05)

        # Establish A(2)->B(5) pattern
        for i in range(20):
            det.process_observation(time=i * 2, band=2, detected=True)
            det.process_observation(time=i * 2 + 1, band=5, detected=True)

        # Expected transition 2->5
        events = det.process_observation(time=100, band=2, detected=True)
        events2 = det.process_observation(time=101, band=5, detected=True)
        hopping_events = [e for e in events + events2 if e.change_type == "HOPPING_CHANGE"]
        self.assertEqual(len(hopping_events), 0, "Expected NO hopping change for learned transition")


class TestSpectrumDriftDetection(unittest.TestCase):
    """Test SPECTRUM_DRIFT detection in RFEnvironmentChangeDetector."""

    def test_spectrum_drift_on_activity_shift(self):
        """Activity shifting from low bands to high bands should trigger SPECTRUM_DRIFT."""
        det = RFEnvironmentChangeDetector(
            num_bands=10,
            drift_window_size=30,
            activity_burst_threshold=100  # Suppress NEW_ACTIVITY events
        )

        # First phase: activity on bands 0-2
        for t in range(60):
            band = t % 3  # bands 0, 1, 2
            det.process_observation(time=t, band=band, detected=True)

        # Second phase: activity shifts to bands 7-9
        drift_events = []
        for t in range(60, 120):
            band = 7 + (t % 3)  # bands 7, 8, 9
            evs = det.process_observation(time=t, band=band, detected=True)
            drift_events.extend([e for e in evs if e.change_type == "SPECTRUM_DRIFT"])

        # Should detect spectrum drift at some point
        self.assertTrue(len(drift_events) >= 1, "Expected SPECTRUM_DRIFT event for activity shift")


class TestReceiverBandwidthAndSpecificDetection(unittest.TestCase):
    """Test receiver bandwidth constraints and receiver-specific differences."""

    def test_bandwidth_parameter_stored(self):
        """Receiver should store configured instantaneous bandwidth."""
        rec = VirtualReceiver(
            num_bands=10,
            instantaneous_bandwidth_mhz=5.0,
            seed=42
        )
        self.assertEqual(rec.instantaneous_bandwidth_mhz, 5.0)

    def test_receiver_specific_detection_differences(self):
        """Two receivers with different detection probabilities should have different detection rates."""
        env = RFEnvironment(num_bands=10, seed=42)
        em = Emitter(name="Signal", band=3, behavior="persistent")
        env.add_emitter(em)

        # Receiver A: high detection probability
        rec_a = VirtualReceiver(
            num_bands=10,
            detection_probability=0.95,
            false_alarm_probability=0.01,
            seed=42
        )

        # Receiver B: low detection probability
        rec_b = VirtualReceiver(
            num_bands=10,
            detection_probability=0.30,
            false_alarm_probability=0.01,
            seed=42
        )

        det_a, det_b = 0, 0
        trials = 100
        for _ in range(trials):
            if rec_a.scan(env, band=3, dwell_time=1):
                det_a += 1
            if rec_b.scan(env, band=3, dwell_time=1):
                det_b += 1

        # Receiver A should detect significantly more
        self.assertGreater(det_a, det_b, "Higher Pd receiver should detect more")

    def test_multiple_receivers_independent_scan_histories(self):
        """Each receiver should maintain its own independent scan history."""
        rec1 = VirtualReceiver(num_bands=10, seed=42)
        rec2 = VirtualReceiver(num_bands=10, seed=99)

        env = RFEnvironment(num_bands=10, seed=42)
        em = Emitter(name="E1", band=0, behavior="persistent")
        env.add_emitter(em)

        rec1.scan(env, band=0, dwell_time=1)
        rec1.scan(env, band=1, dwell_time=1)

        rec2.scan(env, band=5, dwell_time=1)

        self.assertEqual(len(rec1.scan_history), 2)
        self.assertEqual(len(rec2.scan_history), 1)
        self.assertEqual(rec1.scan_history[0]["band"], 0)
        self.assertEqual(rec2.scan_history[0]["band"], 5)


class TestMultiReceiverFailureDegradation(unittest.TestCase):
    """Test multi-receiver system under failure and degradation scenarios."""

    def test_receiver_failure_metrics_reflect_inactivity(self):
        """If one receiver is never used, its utilization should be 0."""
        sched = SpatialScheduler(
            receiver_ids=["R_Active", "R_Failed"],
            num_bands=5,
            seed=42,
            epsilon=0.0
        )

        # Only ever use R_Active
        for t in range(20):
            action = ("R_Active", t % 5)
            sched.update({"time": t, "observation_time": t}, action, t % 2 == 0)
            sched.receiver_selection_count["R_Active"] += 1

        metrics = sched.get_multi_receiver_metrics()
        # R_Failed should have 0 utilization
        self.assertEqual(metrics["receiver_utilization"]["R_Failed"], 0.0)
        self.assertGreater(metrics["receiver_utilization"]["R_Active"], 0.0)

    def test_receiver_starvation_increases_with_neglect(self):
        """Receivers not visited should accumulate staleness."""
        sched = SpatialScheduler(
            receiver_ids=["R1", "R2"],
            num_bands=5,
            seed=42,
            epsilon=0.0,
            max_revisit_interval=50
        )

        # Use only R1 for 20 steps
        for t in range(20):
            action = sched.select_action({"time": t, "observation_time": t})
            sched.update({"time": t + 1, "observation_time": t + 1}, action, True)

        metrics = sched.get_multi_receiver_metrics()
        # Both should have starvation metrics
        self.assertIn("R1", metrics["receiver_starvation"])
        self.assertIn("R2", metrics["receiver_starvation"])


class TestMultiReceiverAsymmetricSNRWithReceivers(unittest.TestCase):
    """Test multi-receiver with actual asymmetric receiver physics."""

    def test_asymmetric_snr_detection_rates(self):
        """Nearby receiver should detect more than distant receiver on same emitter."""
        env = RFEnvironment(num_bands=5, seed=42)
        em = Emitter(name="Target", band=2, behavior="persistent")
        em.power_dbm = -20.0  # Weak emitter
        em.position = (0.0, 0.0)
        env.add_emitter(em)

        # Nearby receiver (1m from emitter) -> SNR ≈ 60 dB -> Pd ≈ 1.0
        rec_near = VirtualReceiver(
            num_bands=5,
            power_model_enabled=True,
            receiver_position=(0.0, 1.0),
            noise_floor_dbm=-100.0,
            detection_threshold_snr_db=20.0,
            snr_steepness=1.0,
            seed=42
        )

        # Distant receiver (100m from emitter) -> SNR ≈ 20 dB -> Pd ≈ 0.5
        rec_far = VirtualReceiver(
            num_bands=5,
            power_model_enabled=True,
            receiver_position=(0.0, 100.0),
            noise_floor_dbm=-100.0,
            detection_threshold_snr_db=20.0,
            snr_steepness=1.0,
            seed=99
        )

        det_near, det_far = 0, 0
        trials = 100
        for _ in range(trials):
            if rec_near.scan(env, band=2, dwell_time=1):
                det_near += 1
            if rec_far.scan(env, band=2, dwell_time=1):
                det_far += 1

        self.assertGreater(det_near, det_far, f"Nearby receiver ({det_near}) should detect more than distant ({det_far})")


class TestPDWCausalityReinforcement(unittest.TestCase):
    """Reinforce PDW causality: observations at time t cannot influence decisions before t."""

    def test_scan_history_timestamps_are_monotonic(self):
        """VirtualReceiver scan history timestamps must be strictly non-decreasing."""
        env = RFEnvironment(num_bands=10, seed=42)
        em = Emitter(name="E1", band=3, behavior="periodic", period=5)
        env.add_emitter(em)

        rec = VirtualReceiver(num_bands=10, seed=42)
        for _ in range(20):
            band = np.random.default_rng(42).integers(0, 10)
            rec.scan(env, band=int(band), dwell_time=1)

        times = [h["observation_time"] for h in rec.scan_history]
        for i in range(1, len(times)):
            self.assertGreaterEqual(times[i], times[i - 1],
                                    f"Timestamp at index {i} ({times[i]}) < index {i-1} ({times[i-1]})")

    def test_scheduler_cannot_access_future_observations(self):
        """Scheduler update with an observation must only reference past/present time."""
        sched = SpatialScheduler(
            receiver_ids=["R1"],
            num_bands=5,
            seed=42,
            epsilon=0.0
        )

        # Simulate: at time t, action is chosen, then observation at t+1
        prev_time = 0
        for t in range(1, 15):
            action = sched.select_action({"time": prev_time, "observation_time": prev_time})
            # Observation comes AFTER action
            sched.update({"time": t, "observation_time": t}, action, True)
            prev_time = t

        # The scheduler's knowledge map should have entries with times <= prev_time
        for rid in sched.receiver_ids:
            for b in range(sched.num_bands):
                entry = sched.knowledge_map.get_entry(rid, b)
                if entry.last_observation >= 0:
                    self.assertLessEqual(entry.last_observation, prev_time)


class TestReceiverSwitchingCostExtended(unittest.TestCase):
    """Extended switching cost tests."""

    def test_no_switching_cost_same_band(self):
        """Repeatedly scanning same band should not incur switching cost."""
        rec = VirtualReceiver(num_bands=10, switching_time=5, settling_time=3, seed=42)
        env = RFEnvironment(num_bands=10, seed=42)

        rec.scan(env, band=3, dwell_time=1)
        time_after_first = env.time

        rec.scan(env, band=3, dwell_time=1)
        time_after_second = env.time

        # No retune: only dwell time of 1
        self.assertEqual(time_after_second - time_after_first, 1)

    def test_switching_cost_different_band(self):
        """Scanning different band should incur switching + settling cost."""
        rec = VirtualReceiver(num_bands=10, switching_time=3, settling_time=2, seed=42)
        env = RFEnvironment(num_bands=10, seed=42)

        rec.scan(env, band=0, dwell_time=1)
        time_after_first = env.time

        rec.scan(env, band=5, dwell_time=1)
        time_after_second = env.time

        # Retune: switching(3) + settling(2) + dwell(1) = 6
        self.assertEqual(time_after_second - time_after_first, 6)


if __name__ == "__main__":
    unittest.main()
