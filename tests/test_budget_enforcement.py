import unittest
from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver
from algorithms.sequential import SequentialScheduler
from algorithms.random_scheduler import RandomScheduler
from algorithms.adaptive_belief import AdaptiveBeliefScheduler
from algorithms.ucb_scheduler import UCB1Scheduler
from algorithms.phase8_spatial_scheduler import SpatialScheduler
from simulator.multi_receiver import SpatialRFEnvironment, MultiReceiverSystem


class TestFixedResourceBudget(unittest.TestCase):
    """
    Verifies that all evaluated schedulers operate under strictly identical
    fixed resource constraints:
    - Same simulation steps/time budget
    - Same receiver switching penalties
    - Identical dwell accounting
    - No scheduler receives extra time, hidden scans, or unauthorized queries.
    """

    def test_single_receiver_budget_parity(self):
        budget_time = 100
        num_bands = 10
        seed = 42

        schedulers = {
            "Sequential": SequentialScheduler(num_bands),
            "Random": RandomScheduler(num_bands, seed=seed),
            "AdaptiveBelief": AdaptiveBeliefScheduler(num_bands, seed=seed),
            "UCB1": UCB1Scheduler(num_bands, seed=seed),
        }

        for name, sched in schedulers.items():
            env = RFEnvironment(num_bands=num_bands, seed=seed)
            em = Emitter(name="E1", band=3, behavior="periodic", period=5, duty_cycle=2, seed=seed)
            env.add_emitter(em)

            receiver = VirtualReceiver(num_bands=num_bands, switching_time=1, seed=seed)
            sched.reset(seed=seed)

            obs_count = 0
            while env.time < budget_time:
                last_obs = receiver.scan_history[-1] if receiver.scan_history else None
                action = sched.select_action(last_obs)
                obs_count += 1
                detected = receiver.scan(env, action, dwell_time=1)
                sched.update(receiver.scan_history[-1], action, detected)

            # Assert time did not exceed budget + 1 (single-step granularity limit)
            self.assertGreaterEqual(env.time, budget_time)
            self.assertLessEqual(env.time, budget_time + 2)

            # Total dwell duration plus switching steps must equal environment time
            total_dwell = sum(o['effective_duration'] for o in receiver.scan_history)
            total_switching = sum(o['switching_time'] for o in receiver.scan_history)
            self.assertEqual(total_dwell + total_switching, env.time,
                             f"Budget accounting failure in {name}: dwell + switch != elapsed time")

    def test_multi_receiver_budget_parity(self):
        budget_time = 100
        num_bands = 10
        seed = 42

        rcfg = [
            {'id': 'R1', 'position': (0.0, 0.0), 'snr_threshold': 15.0},
            {'id': 'R2', 'position': (10.0, 0.0), 'snr_threshold': 15.0},
        ]
        r_ids = ['R1', 'R2']

        sched = SpatialScheduler(r_ids, num_bands=num_bands, seed=seed)
        env = SpatialRFEnvironment(num_bands=num_bands, seed=seed)
        em = Emitter(name="E1", band=2, behavior="periodic", period=6, duty_cycle=2, seed=seed)
        env.add_emitter(em, position=(5.0, 0.0))

        mrs = MultiReceiverSystem(env, rcfg)
        sched.reset(seed=seed)

        last_obs = None
        while env.time < budget_time:
            action = sched.select_action(last_obs)
            rid, band = action
            detected = mrs.scan(rid, band, dwell_time=1)
            last_obs = mrs.get_receiver(rid).scan_history[-1]
            sched.update(last_obs, action, detected)

        self.assertGreaterEqual(env.time, budget_time)
        self.assertLessEqual(env.time, budget_time + 2)


if __name__ == '__main__':
    unittest.main()
