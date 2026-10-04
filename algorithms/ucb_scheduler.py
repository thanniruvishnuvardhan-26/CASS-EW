import numpy as np
from typing import Any, Dict, Optional
from algorithms.base import BaseScheduler


class UCB1Scheduler(BaseScheduler):
    """
    UCB1 (Upper Confidence Bound) RF Scan Scheduler Baseline.
    
    Treats each RF band as an arm in a multi-armed bandit problem.
    Reward is 1.0 for detection, 0.0 for non-detection.
    
    Formula:
        UCB_i = Q_i + c * sqrt(2 * ln(N) / N_i)
    
    Where:
        Q_i = average reward (empirical detection rate) for band i
        N = total observation steps taken
        N_i = number of times band i has been observed
        c = exploration bonus coefficient (default sqrt(2) approx 1.414)
    """

    def __init__(
        self,
        num_bands: int = 10,
        c: float = 1.414,
        seed: Optional[int] = None
    ):
        self.num_bands = num_bands
        self.c = c
        self.rng = np.random.default_rng(seed)
        
        self.counts = np.zeros(num_bands, dtype=int)
        self.rewards = np.zeros(num_bands, dtype=float)
        self.total_steps = 0
        self.last_action = 0

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(self.num_bands, dtype=int)
        self.rewards = np.zeros(self.num_bands, dtype=float)
        self.total_steps = 0
        self.last_action = 0

    def select_action(self, observation: Optional[Any] = None) -> int:
        """
        Select band using UCB1 strategy.
        First observes each band at least once, then maximizes upper confidence bound.
        """
        # Step 1: Ensure each band is sampled at least once
        unvisited = np.where(self.counts == 0)[0]
        if len(unvisited) > 0:
            # Deterministic scan of unvisited in index order
            action = int(unvisited[0])
            self.last_action = action
            return action

        # Step 2: Compute UCB value for each arm
        total_ln = np.log(max(1, self.total_steps))
        avg_rewards = self.rewards / np.maximum(1, self.counts)
        exploration_bonuses = self.c * np.sqrt(2.0 * total_ln / self.counts)
        ucb_values = avg_rewards + exploration_bonuses

        # Pick arm with highest UCB; tie-break deterministically
        best_band = int(np.argmax(ucb_values))
        self.last_action = best_band
        return best_band

    def update(self, observation: Optional[Any], action: int, result: bool) -> None:
        """
        Update arm pull counts and cumulative detection rewards.
        """
        self.total_steps += 1
        self.counts[action] += 1
        if result:
            self.rewards[action] += 1.0

    def get_state(self) -> Dict[str, Any]:
        avg_rewards = self.rewards / np.maximum(1, self.counts)
        return {
            "counts": self.counts.tolist(),
            "rewards": self.rewards.tolist(),
            "mean_detection_rates": avg_rewards.tolist(),
            "total_steps": self.total_steps,
            "last_action": self.last_action
        }
