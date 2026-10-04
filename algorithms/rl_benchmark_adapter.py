import numpy as np
from typing import Any, Dict, Optional
from algorithms.base import BaseScheduler
from algorithms.rl_scheduler import RLScheduler, calculate_reward


class RLBenchmarkAdapter(BaseScheduler):
    """
    Standard BaseScheduler adapter for the experimental RLScheduler.
    Maintains observable Bayesian belief proxy to feed the Q-table discretizer.
    Enforces identical observation and action constraints as all other schedulers.
    """

    def __init__(self, num_bands: int = 10, seed: Optional[int] = None):
        self.num_bands = num_bands
        self.seed = seed
        self.rl = RLScheduler(num_bands=num_bands, dwell_times=(1,), seed=seed, epsilon=0.1)
        self.belief = np.full(num_bands, 0.5)
        self.current_state = self.rl.get_state(self.belief)
        self.last_action_idx = 0

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rl.reset(seed=self.seed)
        self.belief = np.full(self.num_bands, 0.5)
        self.current_state = self.rl.get_state(self.belief)
        self.last_action_idx = 0

    def select_action(self, observation: Optional[Any] = None) -> int:
        self.current_state = self.rl.get_state(self.belief)
        self.last_action_idx = self.rl.choose_action(self.current_state, training=True)
        band, _ = self.rl.action_to_parameters(self.last_action_idx)
        return int(band)

    def update(self, observation: Optional[Any], action: int, result: bool) -> None:
        # Update belief from observable detection
        if result:
            self.belief[action] = min(0.99, self.belief[action] + 0.25)
        else:
            self.belief[action] = max(0.01, self.belief[action] - 0.05)

        next_state = self.rl.get_state(self.belief)
        reward = 10.0 if result else -1.0
        self.rl.update(self.current_state, self.last_action_idx, reward, next_state)
        self.current_state = next_state

    def get_state(self) -> Dict[str, Any]:
        return {
            "q_table_size": len(self.rl.q_table),
            "belief": self.belief.tolist(),
            "epsilon": self.rl.epsilon
        }
