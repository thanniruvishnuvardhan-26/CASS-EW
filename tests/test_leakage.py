import unittest
import numpy as np

from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver
from algorithms.rl_scheduler import RLScheduler
from algorithms.bayesian_scheduler import BayesianScheduler


class TestLeakage(unittest.TestCase):

    def test_receiver_hides_ground_truth(self):
        env = RFEnvironment(num_bands=10, seed=42)
        env.add_emitter(Emitter("E1", band=2, behavior="intermittent", activity_probability=1.0, seed=42))
        rec = VirtualReceiver(num_bands=10, detection_probability=0.9, false_alarm_probability=0.05, seed=42)
        
        # Action only returns a boolean (detected or not)
        result = rec.scan(env, 2, dwell_time=1)
        self.assertIsInstance(result, bool)

        # Ensure the environment's actual state is not leaked in the return value
        self.assertNotIsInstance(result, dict)
        self.assertNotIsInstance(result, list)

    def test_rl_scheduler_input_isolation(self):
        s = RLScheduler(num_bands=10, seed=42)
        
        # Verify state generation only takes beliefs (which are receiver-derived), not ground truth
        belief = np.zeros(10)
        state = s.get_state(belief)
        self.assertIsInstance(state, tuple)
        
        # Verify action only relies on state
        action = s.choose_action(state, training=False)
        self.assertIsInstance(action, int)

    def test_bayesian_scheduler_input_isolation(self):
        s = BayesianScheduler(num_bands=10)
        
        # Verify update only takes band and boolean detection
        # Attempting to pass a dict with ground truth would fail or be ignored
        try:
            s.update(band=2, detected=True)
        except Exception as e:
            self.fail(f"BayesianScheduler update failed with valid observable input: {e}")

if __name__ == "__main__":
    unittest.main()
