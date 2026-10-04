"""
Test Suite: Scheduler Action Validity and Algorithms.
Tests requirements 4, 9:
- action validation
- sequential scanner action progression
- bayesian scheduler belief & dwell selection
- RL scheduler action decoding & Q-update
"""

import unittest
import numpy as np

from algorithms.baseline_scanner import SequentialScanner
from algorithms.bayesian_scheduler import BayesianScheduler, BayesianBelief
from algorithms.rl_scheduler import RLScheduler, calculate_reward


class TestSchedulers(unittest.TestCase):

    def test_sequential_scanner_cycle(self):
        """Verify sequential scanner deterministically cycles through all bands in order."""
        scanner = SequentialScanner(num_bands=10)
        expected = list(range(10)) + list(range(10))
        actual = [scanner.get_action() for _ in range(20)]
        self.assertEqual(actual, expected)

        scanner.reset()
        self.assertEqual(scanner.get_action(), 0)

    def test_bayesian_belief_updates(self):
        """Verify Bayesian belief updates increase on detection and decrease on miss."""
        belief_model = BayesianBelief(num_bands=10, initial_probability=0.1)
        initial_beliefs = belief_model.get_beliefs()
        np.testing.assert_array_almost_equal(initial_beliefs, np.full(10, 0.1))

        # Update band 3 with detection
        belief_model.update(band=3, detected=True)
        self.assertAlmostEqual(belief_model.belief[3], 0.35)  # 0.1 + 0.25

        # Update band 3 with miss
        belief_model.update(band=3, detected=False)
        self.assertAlmostEqual(belief_model.belief[3], 0.30)  # 0.35 - 0.05

        # Bounds clipping checks
        for _ in range(10):
            belief_model.update(band=3, detected=True)
        self.assertAlmostEqual(belief_model.belief[3], 0.99)

        for _ in range(30):
            belief_model.update(band=3, detected=False)
        self.assertAlmostEqual(belief_model.belief[3], 0.01)

    def test_bayesian_scheduler_actions_and_dwell(self):
        """Verify Bayesian scheduler visits all bands initially and scales dwell with probability."""
        scheduler = BayesianScheduler(num_bands=10)

        # First 10 actions must explore bands 0 through 9
        first_10_bands = []
        for _ in range(10):
            band, dwell = scheduler.get_action()
            first_10_bands.append(band)
            self.assertIn(dwell, [1, 2, 3])
            scheduler.update(band, detected=False)

        self.assertEqual(first_10_bands, list(range(10)))

        # Now boost band 4 belief to high probability
        for _ in range(5):
            scheduler.update(band=4, detected=True)

        best_band, chosen_dwell = scheduler.get_action()
        self.assertEqual(best_band, 4)
        self.assertEqual(chosen_dwell, 3)  # Since belief >= 0.6

    def test_rl_scheduler_action_space_mapping(self):
        """Verify 30 discrete actions correctly map to (band, dwell_time) pairs."""
        scheduler = RLScheduler(num_bands=10, dwell_times=(1, 2, 3))
        self.assertEqual(scheduler.num_actions, 30)

        for action in range(30):
            band, dwell = scheduler.action_to_parameters(action)
            self.assertIn(band, list(range(10)))
            self.assertIn(dwell, [1, 2, 3])
            # Check inverse encoding
            expected_band = action // 3
            expected_dwell = [1, 2, 3][action % 3]
            self.assertEqual(band, expected_band)
            self.assertEqual(dwell, expected_dwell)

    def test_rl_scheduler_q_learning_update(self):
        """Verify Bellman update correctly updates Q-values."""
        scheduler = RLScheduler(
            num_bands=10,
            dwell_times=(1, 2, 3),
            learning_rate=0.5,
            discount_factor=0.9,
            seed=42
        )

        state = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        next_state = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
        action = 5
        reward = 10.0

        # Initially Q is 0
        scheduler.update(state, action, reward, next_state)

        # Q(s, a) should be 0 + 0.5 * (10.0 + 0.9 * 0 - 0) = 5.0
        self.assertAlmostEqual(scheduler.q_table[state][action], 5.0)

    def test_reward_function(self):
        """Verify reward function components."""
        # Hit
        r_hit = calculate_reward(detected=True, signal_present=True, dwell_time=1, false_alarm=False)
        self.assertEqual(r_hit, 10.0 - 0.5)

        # Miss
        r_miss = calculate_reward(detected=False, signal_present=True, dwell_time=1, false_alarm=False)
        self.assertEqual(r_miss, -3.0 - 0.5)

        # False alarm
        r_fa = calculate_reward(detected=True, signal_present=False, dwell_time=1, false_alarm=True)
        self.assertEqual(r_fa, -4.0 - 0.5)

        # Longer dwell penalty
        r_dwell3 = calculate_reward(detected=True, signal_present=True, dwell_time=3, false_alarm=False)
        self.assertEqual(r_dwell3, 10.0 - 1.5)


if __name__ == "__main__":
    unittest.main()
