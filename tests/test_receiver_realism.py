"""
Unit tests for Realistic Receiver Model and Multi-Receiver Validation:
- High SNR detection
- Low SNR misses
- Noise-only false alarms
- Dwell effects on detection probability
- Receiver switching and settling costs
- Multi-receiver asymmetric SNR, starvation, coverage, and utilization metrics
"""

import unittest
import numpy as np

from simulator.receiver import VirtualReceiver
from simulator.environment import RFEnvironment, Emitter
from algorithms.phase8_spatial_scheduler import SpatialScheduler


class TestRealisticReceiverModel(unittest.TestCase):
    """
    Test suite for VirtualReceiver SNR, dwell, switching, and noise physics.
    """

    def test_high_snr_detection_reliability(self):
        """High SNR signal (> 25 dB above threshold) must yield near 1.0 detection rate."""
        rec = VirtualReceiver(
            num_bands=10,
            noise_floor_dbm=-100.0,
            detection_threshold_snr_db=10.0,
            power_model_enabled=True,
            receiver_position=(0.0, 0.0),
            seed=42
        )
        env = RFEnvironment(num_bands=10, seed=42)
        # Strong emitter nearby: P_tx = 30 dBm, position (0, 2)
        em = Emitter(name="StrongRadar", band=3, behavior="persistent")
        em.power_dbm = 30.0
        em.position = (0.0, 2.0)
        env.add_emitter(em)

        detections = 0
        trials = 50
        for _ in range(trials):
            det = rec.scan(env, band=3, dwell_time=1)
            if det:
                detections += 1

        det_rate = detections / trials
        self.assertGreater(det_rate, 0.90, f"Expected high SNR detection rate > 0.90, got {det_rate}")

    def test_low_snr_misses(self):
        """Low SNR signal (well below detection threshold) must exhibit high miss rate."""
        rec = VirtualReceiver(
            num_bands=10,
            noise_floor_dbm=-100.0,
            detection_threshold_snr_db=15.0,
            snr_steepness=0.5,
            power_model_enabled=True,
            receiver_position=(0.0, 0.0),
            seed=123
        )
        env = RFEnvironment(num_bands=10, seed=123)
        # Very weak emitter far away: P_tx = -50 dBm, position (0, 100)
        # SNR = (-50 - (20*log10(100)+20)) - (-100) = (-50 - 60) - (-100) = -10 dB
        # Well below 15 dB threshold -> Pd ≈ 0
        em = Emitter(name="WeakEmitter", band=5, behavior="persistent")
        em.power_dbm = -50.0
        em.position = (0.0, 100.0)
        env.add_emitter(em)

        detections = 0
        trials = 50
        for _ in range(trials):
            det = rec.scan(env, band=5, dwell_time=1)
            if det:
                detections += 1

        det_rate = detections / trials
        self.assertLess(det_rate, 0.25, f"Expected weak SNR detection rate < 0.25, got {det_rate}")

    def test_noise_only_false_alarms(self):
        """Scanning a quiet band with no emitter should trigger false alarms at rate ~ P_fa."""
        rec = VirtualReceiver(
            num_bands=10,
            false_alarm_probability=0.04,
            power_model_enabled=True,
            seed=999
        )
        env = RFEnvironment(num_bands=10, seed=999)
        # Empty environment

        detections = 0
        trials = 200
        for _ in range(trials):
            det = rec.scan(env, band=2, dwell_time=1)
            if det:
                detections += 1

        empirical_pfa = detections / trials
        # Empirical rate should be roughly around 0.04 (e.g. 0.01 - 0.10)
        self.assertLessEqual(empirical_pfa, 0.12)

    def test_dwell_effect_improves_marginal_snr_detection(self):
        """Longer dwell time should integrate energy and increase detection probability for marginal SNR."""
        env = RFEnvironment(num_bands=10, seed=77)
        em = Emitter(name="MarginalEmitter", band=4, behavior="persistent")
        em.power_dbm = -5.0
        em.position = (0.0, 10.0)
        env.add_emitter(em)

        # Receiver with dwell = 1
        rec1 = VirtualReceiver(
            num_bands=10,
            detection_threshold_snr_db=12.0,
            power_model_enabled=True,
            seed=77
        )
        # Receiver with dwell = 4
        rec4 = VirtualReceiver(
            num_bands=10,
            detection_threshold_snr_db=12.0,
            power_model_enabled=True,
            seed=77
        )

        det1_count = 0
        det4_count = 0
        trials = 60
        for _ in range(trials):
            if rec1.scan(env, band=4, dwell_time=1):
                det1_count += 1
            if rec4.scan(env, band=4, dwell_time=4):
                det4_count += 1

        self.assertGreaterEqual(det4_count, det1_count, "Dwell=4 should achieve >= detections than Dwell=1")

    def test_switching_and_settling_costs(self):
        """Switching and settling times must advance environment clock proportionally."""
        rec = VirtualReceiver(
            num_bands=10,
            switching_time=2,
            settling_time=1,
            seed=10
        )
        env = RFEnvironment(num_bands=10, seed=10)
        self.assertEqual(env.time, 0)

        # Initial scan on band 1 -> switching time applies only if changing bands
        rec.scan(env, band=1, dwell_time=1)
        # initial band was None -> switching delay 0, dwell 1 -> time = 1
        self.assertEqual(env.time, 1)

        # Retuning to band 5 -> switching_time(2) + settling_time(1) + dwell(1) = 4 steps
        rec.scan(env, band=5, dwell_time=1)
        self.assertEqual(env.time, 5)

        # Same band scan -> no retune delay, only dwell(1) = 1 step
        rec.scan(env, band=5, dwell_time=1)
        self.assertEqual(env.time, 6)


class TestMultiReceiverValidationAndMetrics(unittest.TestCase):
    """
    Test suite for multi-receiver scheduling metrics and validation:
    - Asymmetric receiver SNR / positions
    - Receiver starvation prevention
    - Utilization, coverage, and per-receiver detection rate metrics
    """

    def test_multi_receiver_metrics_reporting(self):
        """SpatialScheduler must calculate and expose multi-receiver utilization and coverage."""
        sched = SpatialScheduler(
            receiver_ids=['R_Alpha', 'R_Bravo'],
            num_bands=6,
            seed=42,
            epsilon=0.0
        )

        # Simulate 20 scheduling steps
        t = 0
        for i in range(20):
            action = sched.select_action({'time': t, 'observation_time': t})
            rid, band = action
            # Feed alternating detection
            detected = (i % 2 == 0)
            sched.update({'time': t + 1, 'observation_time': t + 1}, action, detected)
            t += 1

        metrics = sched.get_multi_receiver_metrics()
        self.assertEqual(metrics["total_scans"], 20)
        self.assertIn("R_Alpha", metrics["receiver_utilization"])
        self.assertIn("R_Bravo", metrics["receiver_utilization"])
        self.assertAlmostEqual(
            metrics["receiver_utilization"]["R_Alpha"] + metrics["receiver_utilization"]["R_Bravo"],
            1.0,
            places=4
        )
        self.assertIn("receiver_coverage", metrics)
        self.assertIn("receiver_starvation", metrics)
        self.assertIn("per_receiver_detection_rate", metrics)

    def test_asymmetric_multi_receiver_spatial_selection(self):
        """When R1 consistently sees an emitter on Band 2 but R2 does not, R1 should be prioritized for Band 2."""
        sched = SpatialScheduler(
            receiver_ids=['R1', 'R2'],
            num_bands=5,
            spatial_weight=0.5,
            seed=101,
            epsilon=0.0
        )

        # Train: R1 detects Band 2 repeatedly, R2 misses Band 2
        for t in range(5):
            sched.update({'time': t, 'observation_time': t}, ('R1', 2), True)
            sched.update({'time': t, 'observation_time': t}, ('R2', 2), False)

        expl = sched.explain_last_decision()
        self.assertIsNotNone(expl)
        r1_ev = sched.spatial_profiles['R1'][2].spatial_evidence(5)
        r2_ev = sched.spatial_profiles['R2'][2].spatial_evidence(5)
        self.assertGreater(r1_ev, r2_ev, "R1 spatial evidence on Band 2 must exceed R2")


if __name__ == '__main__':
    unittest.main()
