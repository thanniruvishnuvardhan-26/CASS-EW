"""
Test Suite: Virtual Receiver and Detection Behavior.
Tests requirements 6, 7:
- receiver behavior
- detection probability boundaries
- false alarm probability boundaries
- dwell stepping semantics
"""

import unittest
from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver


class TestVirtualReceiver(unittest.TestCase):

    def test_receiver_initialization_and_reset(self):
        """Verify receiver initializes properly and reset clears scan history."""
        receiver = VirtualReceiver(num_bands=10)
        self.assertEqual(receiver.num_bands, 10)
        self.assertEqual(len(receiver.scan_history), 0)
        self.assertIsNone(receiver.get_last_result())

    def test_detection_probability_unity(self):
        """Sanity test: If Pd = 1.0, active signal must ALWAYS be detected."""
        env = RFEnvironment(num_bands=10, seed=123)
        # Always active emitter on band 3
        emitter = Emitter("Constant", band=3, behavior="intermittent", activity_probability=1.0, seed=123)
        env.add_emitter(emitter)

        receiver = VirtualReceiver(
            num_bands=10,
            detection_probability=1.0,
            false_alarm_probability=0.0,
            seed=42
        )

        for _ in range(20):
            detected = receiver.scan(env, band=3, dwell_time=1)
            self.assertTrue(detected)
            result = receiver.get_last_result()
            self.assertTrue(result["signal_present"])
            self.assertTrue(result["detected"])

    def test_false_alarm_probability_zero(self):
        """Sanity test: If Pfa = 0.0, inactive bands must NEVER produce detections."""
        env = RFEnvironment(num_bands=10, seed=456)
        # Emitter on band 2 only
        emitter = Emitter("OnBand2", band=2, behavior="intermittent", activity_probability=1.0, seed=456)
        env.add_emitter(emitter)

        receiver = VirtualReceiver(
            num_bands=10,
            detection_probability=0.90,
            false_alarm_probability=0.0,
            seed=42
        )

        # Scan inactive bands (e.g., band 0, 1, 4, 7)
        for band in [0, 1, 4, 7]:
            detected = receiver.scan(env, band=band, dwell_time=1)
            self.assertFalse(detected)
            result = receiver.get_last_result()
            self.assertFalse(result["signal_present"])
            self.assertFalse(result["detected"])

    def test_dwell_stepping_semantics(self):
        """Verify dwell time advances environment by exactly dwell_time steps."""
        env = RFEnvironment(num_bands=10, seed=789)
        receiver = VirtualReceiver(num_bands=10, seed=42)

        self.assertEqual(env.time, 0)

        # Scan with dwell=3
        receiver.scan(env, band=2, dwell_time=3)
        self.assertEqual(env.time, 3)
        self.assertEqual(len(receiver.scan_history), 1)

        # Scan with dwell=2
        receiver.scan(env, band=5, dwell_time=2)
        self.assertEqual(env.time, 5)
        self.assertEqual(len(receiver.scan_history), 2)

    def test_scan_history_semantic_contract(self):
        """Verify Option B semantic contract for scan history."""
        env = RFEnvironment(num_bands=10, seed=789)
        receiver = VirtualReceiver(num_bands=10, switching_time=2, seed=42)

        self.assertEqual(env.time, 0)
        
        receiver.scan(env, band=2, dwell_time=3)
        
        self.assertEqual(len(receiver.scan_history), 1)
        obs = receiver.scan_history[0]
        
        self.assertEqual(obs["dwell_start_time"], 0)
        self.assertEqual(obs["dwell_end_time"], 3)
        self.assertEqual(obs["observation_time"], 3)
        self.assertEqual(obs["effective_duration"], 3)
        self.assertEqual(obs["tuned_band"], 2)
        
        receiver.scan(env, band=5, dwell_time=2)
        
        self.assertEqual(len(receiver.scan_history), 2)
        obs2 = receiver.scan_history[1]
        
        self.assertEqual(obs2["dwell_start_time"], 5) 
        self.assertEqual(obs2["dwell_end_time"], 7)
        self.assertEqual(obs2["observation_time"], 7)
        self.assertEqual(obs2["effective_duration"], 2)
        self.assertEqual(obs2["tuned_band"], 5)


if __name__ == "__main__":
    unittest.main()
