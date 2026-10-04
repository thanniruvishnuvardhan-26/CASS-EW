import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from algorithms.adaptive_belief import AdaptiveBeliefScheduler

class BandTemporalProfile:
    """
    Maintains an independent temporal profile for a single RF band.
    Learns strictly from receiver observations.
    """
    def __init__(
        self,
        history_size: int = 10,
        minimum_detections: int = 3,
        period_window_size: int = 5,
        temporal_tolerance: int = 2,
        freshness_decay_rate: float = 0.01
    ):
        self.history_size = history_size
        self.minimum_detections = minimum_detections
        self.period_window_size = period_window_size
        self.temporal_tolerance = temporal_tolerance
        self.freshness_decay_rate = freshness_decay_rate
        self.reset()
        
    def reset(self) -> None:
        self.detections: List[int] = []
        self.last_observation_time: Optional[int] = None
        
    def update(self, time: int, detection: bool) -> None:
        self.last_observation_time = time
        if detection:
            self.detections.append(time)
            if len(self.detections) > self.history_size:
                self.detections.pop(0)
                
    def get_period_stats(self) -> Tuple[Optional[float], int, Optional[float]]:
        if len(self.detections) < self.minimum_detections:
            return None, 0, None
            
        intervals = [self.detections[i] - self.detections[i-1] for i in range(1, len(self.detections))]
        recent_intervals = intervals[-self.period_window_size:]
        
        estimated_period = float(np.median(recent_intervals))
        mad = float(np.median(np.abs(np.array(recent_intervals) - estimated_period)))
        
        return estimated_period, len(recent_intervals), mad

    def temporal_prediction_score(self, current_time: int) -> float:
        period, _, _ = self.get_period_stats()
        
        if period is None or period <= 0:
            return 0.5  # Neutral score
            
        t_last = self.detections[-1]
        elapsed = current_time - t_last
        
        nearest_multiple = round(elapsed / period) * period
        distance = abs(elapsed - nearest_multiple)
        
        if distance <= self.temporal_tolerance:
            return 1.0
        else:
            decay = (distance - self.temporal_tolerance) / max(1.0, float(self.temporal_tolerance))
            return max(0.0, 1.0 - decay)

    def temporal_reliability(self) -> float:
        """
        Reliability in [0,1].
        Increases with more valid intervals (up to period_window_size).
        Decreases with higher rhythm variability (MAD).
        """
        period, num_intervals, mad = self.get_period_stats()
        if period is None or period <= 0 or num_intervals == 0:
            return 0.0
            
        interval_score = min(1.0, num_intervals / float(self.period_window_size))
        
        # Penalty is 0 if MAD is 0, reaches 1.0 if MAD is half the period or more.
        variability_penalty = min(1.0, mad / (period * 0.5 + 1e-9))
        
        reliability = interval_score * (1.0 - variability_penalty)
        return float(np.clip(reliability, 0.0, 1.0))

    def temporal_freshness(self, current_time: int) -> float:
        """
        Freshness in [0,1].
        Linearly decays with time since last detection.
        """
        if not self.detections:
            return 0.0
        age = current_time - self.detections[-1]
        if age < 0:
            age = 0
        freshness = max(0.0, 1.0 - self.freshness_decay_rate * age)
        return float(freshness)


class MultiBandTemporalProfileScheduler(AdaptiveBeliefScheduler):
    """
    Phase 5 Scheduler using structured Multi-Band Temporal Profiles.
    Allocates scans using a combination of belief, temporal prediction,
    reliability, and evidence freshness.
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
        freshness_decay_rate: float = 0.01,
        belief_weight: float = 0.5,
        temporal_weight: float = 0.5,
        disable_reliability: bool = False,
        disable_freshness: bool = False
    ):
        super().__init__(
            num_bands=num_bands,
            initial_belief=initial_belief,
            belief_hit_update=belief_hit_update,
            belief_miss_update=belief_miss_update,
            epsilon=epsilon,
            seed=seed
        )
        self.temporal_history_size = temporal_history_size
        self.minimum_detections = minimum_detections
        self.period_window_size = period_window_size
        self.temporal_tolerance = temporal_tolerance
        self.freshness_decay_rate = freshness_decay_rate
        
        self.belief_weight = belief_weight
        self.temporal_weight = temporal_weight
        
        self.disable_reliability = disable_reliability
        self.disable_freshness = disable_freshness
        
        self.reset(seed)
        
    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        if hasattr(self, 'temporal_history_size'):
            self.profiles = [
                BandTemporalProfile(
                    history_size=self.temporal_history_size,
                    minimum_detections=self.minimum_detections,
                    period_window_size=self.period_window_size,
                    temporal_tolerance=self.temporal_tolerance,
                    freshness_decay_rate=self.freshness_decay_rate
                ) for _ in range(self.num_bands)
            ]
            self.temporal_influenced_selections = 0
            self.exploration_selections = 0
            self.exploitation_selections = 0

    def _get_obs_time(self, observation: Optional[Any]) -> int:
        if isinstance(observation, dict) and 'observation_time' in observation:
            return observation['observation_time']
        elif hasattr(observation, 'observation_time'):
            return observation.observation_time
        elif isinstance(observation, (int, float)):
            return int(observation)
        return self.step_counter

    def update(self, observation: Optional[Any], action: int, result: bool) -> None:
        obs_time = self._get_obs_time(observation)
        self.profiles[action].update(obs_time, result)
        super().update(observation, action, result)

    def select_action(self, observation: Optional[Any] = None) -> int:
        obs_time = self._get_obs_time(observation)
        
        if self.rng.random() < self.epsilon:
            self.exploration_selections += 1
            return int(self.rng.integers(0, self.num_bands))
            
        self.exploitation_selections += 1
        
        combined_scores = np.zeros(self.num_bands)
        
        for b in range(self.num_bands):
            belief = self.beliefs[b]
            prof = self.profiles[b]
            
            t_score = prof.temporal_prediction_score(obs_time)
            t_rel = prof.temporal_reliability() if not self.disable_reliability else 1.0
            t_fresh = prof.temporal_freshness(obs_time) if not self.disable_freshness else 1.0
            
            temporal_strength = t_score * t_rel * t_fresh
            priority = self.belief_weight * belief + self.temporal_weight * temporal_strength
            combined_scores[b] = priority
            
        belief_only_action = int(np.argmax(self.beliefs))
        combined_action = int(np.argmax(combined_scores))
        
        if belief_only_action != combined_action:
            self.temporal_influenced_selections += 1
            
        return combined_action

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        valid_periods = 0
        sum_periods = 0.0
        sum_intervals = 0
        sum_mads = 0.0
        sum_rel = 0.0
        sum_fresh = 0.0
        
        current_time = self._get_obs_time(None)
        
        for b in range(self.num_bands):
            p, num_ints, mad = self.profiles[b].get_period_stats()
            rel = self.profiles[b].temporal_reliability()
            fresh = self.profiles[b].temporal_freshness(current_time)
            
            sum_rel += rel
            sum_fresh += fresh
            
            if p is not None:
                valid_periods += 1
                sum_periods += p
                sum_intervals += num_ints
                sum_mads += mad
                
        state.update({
            "temporal_valid_bands": valid_periods,
            "temporal_avg_period": sum_periods / valid_periods if valid_periods > 0 else 0.0,
            "temporal_avg_intervals": sum_intervals / valid_periods if valid_periods > 0 else 0,
            "temporal_avg_mad": sum_mads / valid_periods if valid_periods > 0 else 0.0,
            "temporal_avg_reliability": sum_rel / self.num_bands,
            "temporal_avg_freshness": sum_fresh / self.num_bands,
            "temporal_influenced_selections": self.temporal_influenced_selections,
            "exploration_selections": self.exploration_selections,
            "exploitation_selections": self.exploitation_selections
        })
        return state
