from typing import Optional, Dict, Any
import numpy as np

from algorithms.adaptive_belief import AdaptiveBeliefScheduler
from algorithms.temporal_model import TemporalModel

class TemporalBeliefScheduler(AdaptiveBeliefScheduler):
    """
    Hybrid scheduler conceptually combining:
        spatial/band belief + temporal activity score
    """
    def __init__(
        self,
        num_bands: int,
        initial_belief: float = 0.5,
        belief_hit_update: float = 0.2,
        belief_miss_update: float = 0.1,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
        temporal_history_size: int = 10,
        minimum_detections: int = 3,
        period_window_size: int = 5,
        temporal_tolerance: int = 2,
        belief_weight: float = 0.7,
        temporal_weight: float = 0.3
    ):
        super().__init__(
            num_bands=num_bands,
            initial_belief=initial_belief,
            belief_hit_update=belief_hit_update,
            belief_miss_update=belief_miss_update,
            epsilon=epsilon,
            seed=seed
        )
        self.belief_weight = belief_weight
        self.temporal_weight = temporal_weight
        
        self.temporal_model = TemporalModel(
            num_bands=num_bands,
            history_size=temporal_history_size,
            minimum_detections=minimum_detections,
            period_window_size=period_window_size,
            temporal_tolerance=temporal_tolerance
        )
        
    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        if hasattr(self, 'temporal_model'):
            self.temporal_model.reset()
            
    def _get_obs_time(self, observation: Optional[Any]) -> int:
        if isinstance(observation, dict) and 'observation_time' in observation:
            return observation['observation_time']
        elif hasattr(observation, 'observation_time'):
            return observation.observation_time
        elif isinstance(observation, (int, float)):
            return int(observation)
        return self.step_counter

    def update(self, observation: Optional[Any], action: int, result: bool) -> None:
        """
        The scheduler obtains observation_time from the legitimate receiver
        observation path if available.
        """
        obs_time = self._get_obs_time(observation)
        
        # Temporal observation record
        self.temporal_model.update(action, obs_time, result)
        
        # Phase 3 belief update
        super().update(observation, action, result)
        
    def select_action(self, observation: Optional[Any] = None) -> int:
        obs_time = self._get_obs_time(observation)
        
        # Exploration mechanism
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_bands))
            
        combined_scores = np.zeros(self.num_bands)
        
        # Combine belief and temporal score
        for b in range(self.num_bands):
            belief = self.beliefs[b]
            t_score = self.temporal_model.temporal_score(b, obs_time)
            combined_scores[b] = self.belief_weight * belief + self.temporal_weight * t_score
            
        # Deterministic tie-breaking (lowest index on tie)
        return int(np.argmax(combined_scores))
        
    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        valid_periods = 0
        sum_periods = 0.0
        sum_intervals = 0
        sum_mads = 0.0
        
        for b in range(self.num_bands):
            p, num_ints, mad = self.temporal_model.estimate_period(b)
            if p is not None:
                valid_periods += 1
                sum_periods += p
                sum_intervals += num_ints
                sum_mads += mad
                
        state.update({
            "temporal_valid_bands": valid_periods,
            "temporal_avg_period": sum_periods / valid_periods if valid_periods > 0 else 0.0,
            "temporal_avg_intervals": sum_intervals / valid_periods if valid_periods > 0 else 0,
            "temporal_avg_mad": sum_mads / valid_periods if valid_periods > 0 else 0.0
        })
        return state
