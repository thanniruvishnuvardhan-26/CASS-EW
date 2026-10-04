"""
Test Suite: Random Seed Handling and Experiment Reproducibility.
Tests requirement 10: random seed reproducibility.
"""

import unittest
import numpy as np

from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver
from algorithms.rl_scheduler import RLScheduler
from evaluation.final_benchmark import generate_trace, generate_detector_trace


class TestReproducibility(unittest.TestCase):

    def test_trace_generation_seed_reproducibility(self):
        """Verify identical seed produces byte-for-byte identical traces."""
        trace1 = generate_trace(seed=42, total_time=100)
        trace2 = generate_trace(seed=42, total_time=100)
        self.assertEqual(trace1, trace2)

        # Different seed must produce different traces
        trace3 = generate_trace(seed=999, total_time=100)
        self.assertNotEqual(trace1, trace3)

    def test_detector_trace_seed_reproducibility(self):
        """Verify detector randomness matrix is strictly reproducible by seed."""
        det1, fa1 = generate_detector_trace(seed=542)
        det2, fa2 = generate_detector_trace(seed=542)
        np.testing.assert_array_equal(det1, det2)
        np.testing.assert_array_equal(fa1, fa2)

        det3, fa3 = generate_detector_trace(seed=888)
        self.assertFalse(np.array_equal(det1, det3))

    def test_environment_and_receiver_seed_reproducibility(self):
        """Verify environment stepping with seeded emitters is deterministic."""
        def run_sim(seed):
            env = RFEnvironment(num_bands=10, seed=seed)
            env.add_emitter(Emitter("E1", band=2, behavior="intermittent", activity_probability=0.7, seed=seed + 1))
            env.add_emitter(Emitter("E2", band=7, behavior="hopping", activity_probability=0.8, hop_bands=[6, 7, 8], seed=seed + 2))
            rec = VirtualReceiver(num_bands=10, detection_probability=0.90, false_alarm_probability=0.05, seed=seed + 3)

            history = []
            for b in [2, 7, 2, 7, 6]:
                rec.scan(env, b, dwell_time=1)
                history.append(rec.get_last_result())
            return history

        run_a = run_sim(seed=100)
        run_b = run_sim(seed=100)
        self.assertEqual(run_a, run_b)

        run_c = run_sim(seed=200)
        self.assertNotEqual(run_a, run_c)

    def test_rl_scheduler_exploration_determinism(self):
        """Verify RL scheduler choose_action with seeded RNG is deterministic."""
        s1 = RLScheduler(num_bands=10, epsilon=0.5, seed=42)
        s2 = RLScheduler(num_bands=10, epsilon=0.5, seed=42)

        state = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        actions1 = [s1.choose_action(state, training=True) for _ in range(50)]
        actions2 = [s2.choose_action(state, training=True) for _ in range(50)]

        self.assertEqual(actions1, actions2)


if __name__ == "__main__":
    unittest.main()
