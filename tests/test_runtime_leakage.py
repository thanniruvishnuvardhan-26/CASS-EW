import unittest
import numpy as np

from algorithms.rl_scheduler import RLScheduler

# A test harness for verifying runtime leakage
class TestRuntimeLeakage(unittest.TestCase):

    def test_identical_observable_history_same_action(self):
        # We will create two schedulers with identical RNG seeds.
        # We will feed them the same sequence of observations (detected = True/False),
        # representing the observable history.
        # We will prove they take the exact same next action, regardless of what the 
        # actual ground truth would be in the future.
        
        NUM_BANDS = 10
        seed = 42

        scheduler1 = RLScheduler(
            num_bands=NUM_BANDS,
            dwell_times=(1, 2, 3),
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.3, # stochastic
            seed=seed
        )

        scheduler2 = RLScheduler(
            num_bands=NUM_BANDS,
            dwell_times=(1, 2, 3),
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=0.3, # stochastic
            seed=seed
        )

        # Same observable belief state
        belief = np.array([0.1] * NUM_BANDS)
        belief[2] = 0.9 # We observed a detection here
        
        state1 = scheduler1.get_state(belief)
        state2 = scheduler2.get_state(belief)

        # Both take an action. Because they have identical internal state and RNG,
        # they MUST take the exact same action, proving that the action solely
        # depends on the observable state + RNG, and has zero data path to any
        # future ground truth (which doesn't even exist in this context).
        
        action1 = scheduler1.choose_action(state1, training=True)
        action2 = scheduler2.choose_action(state2, training=True)

        self.assertEqual(action1, action2, "Stochastic schedulers with same seed and history must take identical actions")

        # Now test with evaluation mode (deterministic)
        scheduler1.epsilon = 0.0
        scheduler2.epsilon = 0.0

        action1_det = scheduler1.choose_action(state1, training=False)
        action2_det = scheduler2.choose_action(state2, training=False)

        self.assertEqual(action1_det, action2_det, "Deterministic schedulers with same history must take identical actions")
        
        # Test that updating the Q-table with identical observations produces identical results
        # Even if "true" hidden ground truth (which affects reward in training) was different,
        # if the reward provided to update is the same, the Q-table must be identical.
        
        scheduler1.update(state1, action1, 5.0, state1)
        scheduler2.update(state2, action2, 5.0, state2)

        for key in scheduler1.q_table:
            np.testing.assert_array_equal(scheduler1.q_table[key], scheduler2.q_table[key])

    def test_production_end_to_end_leakage(self):
        from simulator.environment import RFEnvironment, Emitter
        from simulator.receiver import VirtualReceiver

        # Construct two controlled environments
        
        # Scenario A: Emitter is always on
        envA = RFEnvironment(num_bands=10, seed=100)
        envA.add_emitter(Emitter(name="A", band=2, behavior="intermittent", activity_probability=1.0, seed=101))
        
        # Scenario B: Emitter is always on INITALLY, but we will mutate its future
        envB = RFEnvironment(num_bands=10, seed=100)
        envB.add_emitter(Emitter(name="B", band=2, behavior="intermittent", activity_probability=1.0, seed=101))

        # Identical receivers
        receiverA = VirtualReceiver(num_bands=10, detection_probability=1.0, false_alarm_probability=0.0, seed=200)
        receiverB = VirtualReceiver(num_bands=10, detection_probability=1.0, false_alarm_probability=0.0, seed=200)

        # Identical schedulers
        schedulerA = RLScheduler(num_bands=10, dwell_times=(1, 2, 3), seed=300)
        schedulerB = RLScheduler(num_bands=10, dwell_times=(1, 2, 3), seed=300)

        # Step 1: scan band 2, generating identical observable history
        detectedA = receiverA.scan(envA, band=2, dwell_time=1)
        detectedB = receiverB.scan(envB, band=2, dwell_time=1)
        
        self.assertEqual(detectedA, detectedB)
        
        # Convert observation to belief
        beliefA = np.zeros(10)
        beliefA[2] = 1.0 if detectedA else 0.0
        
        beliefB = np.zeros(10)
        beliefB[2] = 1.0 if detectedB else 0.0

        stateA = schedulerA.get_state(beliefA)
        stateB = schedulerB.get_state(beliefB)

        # Diverge hidden future: 
        # For scenario B, we turn off the emitter completely (future truth is different).
        # This simulates a hidden ground truth trajectory change.
        envB.emitters[0].activity_probability = 0.0
        
        # Now choose action
        actionA = schedulerA.choose_action(stateA)
        actionB = schedulerB.choose_action(stateB)

        # The schedulers have exactly the same state and RNG.
        # They MUST not look at `envA.emitters` or `receiver.scan_history` (which has future / ground truth if abused)
        self.assertEqual(actionA, actionB, "Scheduler decision must not be influenced by future hidden truth")

        # Prove that the environments are actually different now in the future
        future_envA = envA.step()
        future_envB = envB.step()
        self.assertNotEqual(future_envA, future_envB, "Test setup error: future environments must diverge")

if __name__ == '__main__':
    unittest.main()
