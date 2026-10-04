import numpy as np
from typing import Any, Dict, Optional
from algorithms.base import BaseScheduler

class AdaptiveBeliefScheduler(BaseScheduler):
    """
    Phase 3 Adaptive RF Scheduler using explicit beliefs.
    
    Operates solely on receiver observations. Maintains a belief for each band
    and updates it based on detection hits/misses. Selects bands using an 
    epsilon-greedy approach for exploration, with deterministic tie-breaking.
    """
    
    def __init__(
        self, 
        num_bands: int = 10,
        initial_belief: float = 0.5,
        belief_hit_update: float = 0.2,
        belief_miss_update: float = 0.1,
        epsilon: float = 0.1,
        seed: Optional[int] = None
    ):
        self.num_bands = num_bands
        self.initial_belief = initial_belief
        self.belief_hit_update = belief_hit_update
        self.belief_miss_update = belief_miss_update
        self.epsilon = epsilon
        
        self.rng = np.random.default_rng(seed)
        
        # Internal state
        self.beliefs = np.full(num_bands, initial_belief)
        self.observations = np.zeros(num_bands, dtype=int)
        self.detections = np.zeros(num_bands, dtype=int)
        self.last_observed = np.full(num_bands, -1, dtype=int)
        
        self.step_counter = 0

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
        self.beliefs = np.full(self.num_bands, self.initial_belief)
        self.observations = np.zeros(self.num_bands, dtype=int)
        self.detections = np.zeros(self.num_bands, dtype=int)
        self.last_observed = np.full(self.num_bands, -1, dtype=int)
        self.step_counter = 0

    def select_action(self, observation: Optional[Any] = None) -> int:
        """
        Selects next band using epsilon-greedy over beliefs.
        Ties are broken by choosing the lowest band index.
        """
        # Exploration
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_bands))
            
        # Exploitation
        # np.argmax returns the first occurrence of the maximum value, 
        # guaranteeing deterministic tie-breaking to the lowest index.
        return int(np.argmax(self.beliefs))

    def update(self, observation: Optional[Any], action: int, result: bool) -> None:
        """
        Updates belief state based ONLY on the observation result.
        """
        self.observations[action] += 1
        self.last_observed[action] = self.step_counter
        self.step_counter += 1
        
        if result:
            self.detections[action] += 1
            self.beliefs[action] = min(1.0, self.beliefs[action] + self.belief_hit_update)
        else:
            self.beliefs[action] = max(0.0, self.beliefs[action] - self.belief_miss_update)

    def get_state(self) -> Dict[str, Any]:
        """Returns the inspectable belief state."""
        return {
            "beliefs": self.beliefs.copy(),
            "observations": self.observations.copy(),
            "detections": self.detections.copy(),
            "last_observed": self.last_observed.copy(),
            "step_counter": self.step_counter
        }
