import unittest
import numpy as np
from algorithms.ucb_scheduler import UCB1Scheduler


class TestUCB1Scheduler(unittest.TestCase):
    def setUp(self):
        self.num_bands = 10
        self.scheduler = UCB1Scheduler(num_bands=self.num_bands, c=1.414, seed=42)

    def test_initial_exploration_covers_all_bands(self):
        """UCB1 must explore all arms once before exploiting."""
        actions = []
        for _ in range(self.num_bands):
            action = self.scheduler.select_action()
            actions.append(action)
            self.scheduler.update(None, action, False)
        self.assertEqual(actions, list(range(self.num_bands)))

    def test_exploitation_favors_rewarded_band(self):
        """Band receiving consistent rewards should be pulled more frequently than unrewarded bands."""
        # Initial 10 pulls: band 3 gets reward 1, all others get 0
        for b in range(self.num_bands):
            a = self.scheduler.select_action()
            self.scheduler.update(None, a, (a == 3))

        # Next 30 pulls
        pull_counts = {b: 0 for b in range(self.num_bands)}
        for _ in range(30):
            a = self.scheduler.select_action()
            pull_counts[a] += 1
            # Band 3 continues to give rewards with 80% prob
            self.scheduler.update(None, a, (a == 3))

        # Band 3 should be pulled significantly more than any other individual band
        self.assertGreater(pull_counts[3], 5)
        for b in range(self.num_bands):
            if b != 3:
                self.assertLess(pull_counts[b], pull_counts[3])

    def test_reset_clears_counts_and_rewards(self):
        """Resetting must re-initialize counters and RNG."""
        for b in range(self.num_bands):
            a = self.scheduler.select_action()
            self.scheduler.update(None, a, True)
        self.assertEqual(self.scheduler.total_steps, self.num_bands)

        self.scheduler.reset(seed=99)
        self.assertEqual(self.scheduler.total_steps, 0)
        self.assertEqual(sum(self.scheduler.counts), 0)
        self.assertEqual(sum(self.scheduler.rewards), 0.0)
        self.assertEqual(self.scheduler.select_action(), 0)


if __name__ == '__main__':
    unittest.main()
