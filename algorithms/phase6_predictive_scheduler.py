import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from algorithms.phase5_temporal_profile import MultiBandTemporalProfileScheduler, BandTemporalProfile

class BandPredictiveProfile(BandTemporalProfile):
    """
    Phase 6 Predictive Profile.
    Extends BandTemporalProfile with explicit next-detection prediction
    and lifecycle tracking (hits, misses, errors).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_prediction_stats()
        
    def reset_prediction_stats(self):
        self.next_predicted_time: Optional[float] = None
        self.prediction_confidence: float = 0.0
        
        self.prediction_count = 0
        self.prediction_hits = 0
        self.prediction_misses = 0
        self.prediction_absolute_errors: List[float] = []
        self.prediction_history: List[Tuple[float, float]] = []
        self.active_prediction: Optional[Tuple[float, int]] = None

    def reset(self) -> None:
        super().reset()
        self.reset_prediction_stats()

    def update(self, time: int, detection: bool) -> None:
        # Evaluate previous prediction before updating with the new detection
        if detection and self.active_prediction is not None:
            predicted_time, made_at_time = self.active_prediction
            if time > made_at_time:
                error = time - predicted_time
                abs_error = abs(error)
                
                self.prediction_count += 1
                self.prediction_absolute_errors.append(abs_error)
                
                if abs_error <= self.temporal_tolerance:
                    self.prediction_hits += 1
                else:
                    self.prediction_misses += 1
                    
                self.active_prediction = None
                
        # Super update handles appending detection and tracking intervals
        super().update(time, detection)
        
        # Formulate new prediction
        period, num_ints, mad = self.get_period_stats()
        if period is not None and len(self.detections) >= self.minimum_detections:
            last_det = self.detections[-1]
            self.next_predicted_time = last_det + period
            
            # Confidence is bounded and derived from Phase 5 reliability/freshness
            rel = self.temporal_reliability()
            fresh = self.temporal_freshness(time)
            self.prediction_confidence = rel * fresh
            
            # Start tracking a new prediction opportunity if we just detected something
            if detection or self.active_prediction is None:
                self.active_prediction = (self.next_predicted_time, time)
                self.prediction_history.append((self.next_predicted_time, self.prediction_confidence))
        else:
            self.next_predicted_time = None
            self.prediction_confidence = 0.0

    def predictive_score(self, current_time: int) -> float:
        """
        Score in [0, 1] based on proximity to next predicted time.
        Takes temporal_tolerance into account.
        """
        if self.next_predicted_time is None:
            return 0.5
            
        distance = abs(current_time - self.next_predicted_time)
        if distance <= self.temporal_tolerance:
            return 1.0
        else:
            # Linearly decay to 0 at distance == 2 * tolerance
            decay = (distance - self.temporal_tolerance) / max(1.0, float(self.temporal_tolerance))
            return max(0.0, 1.0 - decay)


class PredictiveScheduler(MultiBandTemporalProfileScheduler):
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
        belief_weight: float = 0.4,
        temporal_weight: float = 0.3,
        prediction_weight: float = 0.3,
        disable_reliability: bool = False,
        disable_freshness: bool = False,
        disable_prediction: bool = False
    ):
        self.prediction_weight = prediction_weight
        self.disable_prediction = disable_prediction
        # Setting prediction_weight early so we can override super's behavior
        
        super().__init__(
            num_bands=num_bands,
            initial_belief=initial_belief,
            belief_hit_update=belief_hit_update,
            belief_miss_update=belief_miss_update,
            epsilon=epsilon,
            seed=seed,
            temporal_history_size=temporal_history_size,
            minimum_detections=minimum_detections,
            period_window_size=period_window_size,
            temporal_tolerance=temporal_tolerance,
            freshness_decay_rate=freshness_decay_rate,
            belief_weight=belief_weight,
            temporal_weight=temporal_weight,
            disable_reliability=disable_reliability,
            disable_freshness=disable_freshness
        )

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        if hasattr(self, 'temporal_history_size'):
            # Override Phase 5 profiles with Phase 6 Predictive profiles
            self.profiles = [
                BandPredictiveProfile(
                    history_size=self.temporal_history_size,
                    minimum_detections=self.minimum_detections,
                    period_window_size=self.period_window_size,
                    temporal_tolerance=self.temporal_tolerance,
                    freshness_decay_rate=self.freshness_decay_rate
                ) for _ in range(self.num_bands)
            ]
            self.predictive_influenced_selections = 0

    def select_action(self, observation: Optional[Any] = None) -> int:
        obs_time = self._get_obs_time(observation)
        
        if self.rng.random() < self.epsilon:
            self.exploration_selections += 1
            return int(self.rng.integers(0, self.num_bands))
            
        self.exploitation_selections += 1
        
        combined_scores = np.zeros(self.num_bands)
        temporal_only_scores = np.zeros(self.num_bands)
        belief_only_scores = np.zeros(self.num_bands)
        
        for b in range(self.num_bands):
            belief = self.beliefs[b]
            prof = self.profiles[b]
            
            # Phase 5 temporal strength
            t_score = prof.temporal_prediction_score(obs_time)
            t_rel = prof.temporal_reliability() if not self.disable_reliability else 1.0
            t_fresh = prof.temporal_freshness(obs_time) if not self.disable_freshness else 1.0
            temporal_strength = t_score * t_rel * t_fresh
            
            # Phase 6 prediction strength
            p_score = prof.predictive_score(obs_time)
            prediction_strength = p_score * prof.prediction_confidence if not self.disable_prediction else 0.0
            
            priority = (self.belief_weight * belief + 
                        self.temporal_weight * temporal_strength + 
                        self.prediction_weight * prediction_strength)
                        
            combined_scores[b] = priority
            temporal_only_scores[b] = self.belief_weight * belief + self.temporal_weight * temporal_strength
            belief_only_scores[b] = belief
            
        belief_only_action = int(np.argmax(belief_only_scores))
        combined_action = int(np.argmax(combined_scores))
        temporal_only_action = int(np.argmax(temporal_only_scores))
        
        if belief_only_action != temporal_only_action and self.temporal_weight > 0:
            self.temporal_influenced_selections += 1
            
        if belief_only_action != combined_action and temporal_only_action != combined_action and not self.disable_prediction and self.prediction_weight > 0:
            self.predictive_influenced_selections += 1
            
        return combined_action

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        
        total_opportunities = 0
        total_hits = 0
        total_misses = 0
        total_abs_errors = 0.0
        sum_confidence = 0.0
        
        for b in range(self.num_bands):
            prof = self.profiles[b]
            total_opportunities += prof.prediction_count
            total_hits += prof.prediction_hits
            total_misses += prof.prediction_misses
            total_abs_errors += sum(prof.prediction_absolute_errors)
            sum_confidence += prof.prediction_confidence
            
        state.update({
            "prediction_opportunities": total_opportunities,
            "prediction_hits": total_hits,
            "prediction_misses": total_misses,
            "prediction_hit_rate": total_hits / total_opportunities if total_opportunities > 0 else 0.0,
            "mean_absolute_prediction_error": total_abs_errors / total_opportunities if total_opportunities > 0 else 0.0,
            "prediction_coverage": total_opportunities / max(1, self.step_counter),
            "prediction_confidence_mean": sum_confidence / self.num_bands,
            "predictive_influenced_selections": self.predictive_influenced_selections
        })
        return state
