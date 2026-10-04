"""
Test Suite: RF Environment and Emitter Behavior.
Tests requirements 2, 3, 5:
- environment initialization
- environment reset
- emitter generation & stepping
"""

import unittest
import numpy as np

from simulator.environment import RFEnvironment, Emitter


class TestRFEnvironment(unittest.TestCase):

    def test_environment_initialization(self):
        """Verify environment initializes with correct default state."""
        env = RFEnvironment(num_bands=10)
        self.assertEqual(env.num_bands, 10)
        self.assertEqual(env.time, 0)
        self.assertEqual(len(env.emitters), 0)

        state = env.get_state()
        self.assertEqual(state["time"], 0)
        self.assertEqual(state["active_bands"], [])

    def test_environment_reset(self):
        """Verify environment reset properly resets time and emitter states."""
        env = RFEnvironment(num_bands=10, seed=42)
        emitter = Emitter("Test_Emitter", band=3, behavior="intermittent", activity_probability=1.0)
        env.add_emitter(emitter)

        # Advance 5 steps
        for _ in range(5):
            env.step()

        self.assertEqual(env.time, 5)
        self.assertTrue(emitter.active)

        # Reset
        env.reset(seed=42)
        self.assertEqual(env.time, 0)
        self.assertFalse(emitter.active)
        self.assertEqual(emitter.band, 3)

    def test_intermittent_emitter_probability_boundaries(self):
        """Verify intermittent emitter behavior at boundary probabilities 0.0 and 1.0."""
        # 100% active emitter
        always_on = Emitter("AlwaysOn", band=1, behavior="intermittent", activity_probability=1.0, seed=42)
        for _ in range(20):
            self.assertTrue(always_on.step())
            self.assertTrue(always_on.active)

        # 0% active emitter
        always_off = Emitter("AlwaysOff", band=2, behavior="intermittent", activity_probability=0.0, seed=42)
        for _ in range(20):
            self.assertFalse(always_off.step())
            self.assertFalse(always_off.active)

    def test_periodic_emitter_toggling(self):
        """Verify periodic emitter strictly toggles active state every timestep."""
        periodic = Emitter("Periodic", band=5, behavior="periodic")
        self.assertFalse(periodic.active)

        for step in range(10):
            active = periodic.step()
            # On odd steps (1, 3, 5...) active is True, on even steps False
            expected = bool((step + 1) % 2 == 1)
            self.assertEqual(active, expected)

    def test_frequency_hopping_sequence(self):
        """Verify frequency hopping emitter hops through its sequence correctly."""
        hop_bands = [6, 7, 8, 9]
        hopper = Emitter(
            "Hopper",
            band=7,
            behavior="hopping",
            activity_probability=1.0,  # Always active to test deterministic hopping
            hop_bands=hop_bands,
            seed=42
        )

        visited_bands = []
        for _ in range(8):
            hopper.step()
            visited_bands.append(hopper.band)

        # Should cycle through hop_bands twice: 6, 7, 8, 9, 6, 7, 8, 9
        self.assertEqual(visited_bands, [6, 7, 8, 9, 6, 7, 8, 9])

    def test_empty_environment_step(self):
        """Verify environment with no emitters returns empty active bands."""
        env = RFEnvironment(num_bands=10)
        for _ in range(5):
            active = env.step()
            self.assertEqual(active, set())


if __name__ == "__main__":
    unittest.main()
