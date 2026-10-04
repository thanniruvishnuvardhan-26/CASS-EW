import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Set
from collections import defaultdict

from algorithms.phase6_predictive_scheduler import PredictiveScheduler, BandPredictiveProfile

class PatternRelationship:
    def __init__(self, band_a: int, band_b: int):
        self.band_a = band_a
        self.band_b = band_b
        self.coincidences = 0
        self.observations = 0
        self.relative_timings = []
        self.last_observed_time = -1
        
    def update(self, time: int, coincidence: bool, timing_diff: Optional[float] = None):
        self.observations += 1
        if coincidence:
            self.coincidences += 1
            self.last_observed_time = time
            if timing_diff is not None:
                self.relative_timings.append(timing_diff)
                
    def confidence(self) -> float:
        if self.observations < 3:
            return 0.0
        return self.coincidences / self.observations
        
    def freshness(self, current_time: int, decay_rate: float = 0.01) -> float:
        if self.last_observed_time < 0:
            return 0.0
        return np.exp(-decay_rate * (current_time - self.last_observed_time))


class BandPatternProfile(BandPredictiveProfile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_count = 0
        self.detection_count = 0
        
    def update(self, time: int, detection: bool) -> None:
        self.observation_count += 1
        if detection:
            self.detection_count += 1
        super().update(time, detection)


class PatternAwareScheduler(PredictiveScheduler):
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
        belief_weight: float = 0.3,
        temporal_weight: float = 0.2,
        prediction_weight: float = 0.3,
        pattern_weight: float = 0.2,
        disable_reliability: bool = False,
        disable_freshness: bool = False,
        disable_prediction: bool = False,
        disable_pattern: bool = False
    ):
        self.pattern_weight = pattern_weight
        self.disable_pattern = disable_pattern
        
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
            prediction_weight=prediction_weight,
            disable_reliability=disable_reliability,
            disable_freshness=disable_freshness,
            disable_prediction=disable_prediction
        )

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        if hasattr(self, 'temporal_history_size'):
            self.profiles = [
                BandPatternProfile(
                    history_size=self.temporal_history_size,
                    minimum_detections=self.minimum_detections,
                    period_window_size=self.period_window_size,
                    temporal_tolerance=self.temporal_tolerance,
                    freshness_decay_rate=self.freshness_decay_rate
                ) for _ in range(self.num_bands)
            ]
        self.relationships: Dict[Tuple[int, int], PatternRelationship] = {}
        self.pattern_influenced_selections = 0
        self.pattern_candidates_total = 0
        self.competing_candidates_count = 0
        self.last_selected_band = None

    def _update_relationships(self, time: int):
        # Examine pairs of bands that have predictions near the current time
        for i in range(self.num_bands):
            for j in range(i + 1, self.num_bands):
                if (i, j) not in self.relationships:
                    self.relationships[(i, j)] = PatternRelationship(i, j)
                
                pi = self.profiles[i].next_predicted_time
                pj = self.profiles[j].next_predicted_time
                
                if pi is not None and pj is not None:
                    # Are they predicted to happen close to each other?
                    diff = abs(pi - pj)
                    if abs(time - pi) <= self.temporal_tolerance or abs(time - pj) <= self.temporal_tolerance:
                        coincidence = (diff <= self.temporal_tolerance * 2)
                        self.relationships[(i, j)].update(time, coincidence, diff if coincidence else None)

    def update(self, observation: Dict[str, Any], action: int, result: bool) -> None:
        super().update(observation, action, result)
        obs_time = self._get_obs_time(observation)
        self._update_relationships(obs_time)

    def select_action(self, observation: Optional[Any] = None) -> int:
        obs_time = self._get_obs_time(observation)
        
        if self.rng.random() < self.epsilon:
            self.exploration_selections += 1
            action = int(self.rng.integers(0, self.num_bands))
            self.last_selected_band = action
            return action
            
        self.exploitation_selections += 1
        
    def get_priorities(self, obs_time: int) -> Tuple[np.ndarray, np.ndarray]:
        combined_scores = np.zeros(self.num_bands)
        without_pattern_scores = np.zeros(self.num_bands)
        
        for b in range(self.num_bands):
            belief = self.beliefs[b]
            prof = self.profiles[b]
            
            t_score = prof.temporal_prediction_score(obs_time)
            t_rel = prof.temporal_reliability() if not self.disable_reliability else 1.0
            t_fresh = prof.temporal_freshness(obs_time) if not self.disable_freshness else 1.0
            temporal_strength = t_score * t_rel * t_fresh
            
            p_score = prof.predictive_score(obs_time)
            prediction_strength = p_score * prof.prediction_confidence if not self.disable_prediction else 0.0
            
            # Pattern strength from relationships
            pattern_strength = 0.0
            if not self.disable_pattern:
                max_rel_score = 0.0
                for other_b in range(self.num_bands):
                    if other_b != b:
                        pair = (min(b, other_b), max(b, other_b))
                        if pair in self.relationships:
                            rel = self.relationships[pair]
                            conf = rel.confidence() * rel.freshness(obs_time, self.freshness_decay_rate)
                            other_prof = self.profiles[other_b]
                            other_p_score = other_prof.predictive_score(obs_time)
                            score = conf * other_p_score
                            if score > max_rel_score:
                                max_rel_score = score
                pattern_strength = max_rel_score
            
            priority_without = (self.belief_weight * belief + 
                                self.temporal_weight * temporal_strength + 
                                self.prediction_weight * prediction_strength)
                                
            priority_with = priority_without + self.pattern_weight * pattern_strength
            
            combined_scores[b] = priority_with
            without_pattern_scores[b] = priority_without
            
        return combined_scores, without_pattern_scores

    def select_action(self, observation: Optional[Any] = None) -> int:
        obs_time = self._get_obs_time(observation)
        
        if self.rng.random() < self.epsilon:
            self.exploration_selections += 1
            action = int(self.rng.integers(0, self.num_bands))
            self.last_selected_band = action
            return action
            
        self.exploitation_selections += 1
        
        combined_scores, without_pattern_scores = self.get_priorities(obs_time)

        
        # Find candidates based on combined_scores
        candidates = [b for b, score in enumerate(combined_scores) if score > 0.5]

        self.competing_candidates_count += len(candidates)
        if len(candidates) > 0:
            self.pattern_candidates_total += 1
            
        action_without = int(np.argmax(without_pattern_scores))
        action_with = int(np.argmax(combined_scores))
        
        if action_with != action_without and not self.disable_pattern and self.pattern_weight > 0:
            self.pattern_influenced_selections += 1
            
        # We also need to preserve Phase 6 and Phase 5 diagnostic tracking conceptually,
        # but the Phase 6 implementation already tracks it in super() if we call it.
        # However, super().select_action handles epsilon and increments selections.
        # To avoid duplicating logic or calling super and discarding its result,
        # we duplicate the Phase 5/6 influence tracking here.
        belief_only_scores = self.beliefs
        temporal_only_scores = np.zeros(self.num_bands)
        for b in range(self.num_bands):
            prof = self.profiles[b]
            t_score = prof.temporal_prediction_score(obs_time)
            t_rel = prof.temporal_reliability() if not self.disable_reliability else 1.0
            t_fresh = prof.temporal_freshness(obs_time) if not self.disable_freshness else 1.0
            temporal_only_scores[b] = self.belief_weight * self.beliefs[b] + self.temporal_weight * (t_score * t_rel * t_fresh)

        belief_only_action = int(np.argmax(belief_only_scores))
        temporal_only_action = int(np.argmax(temporal_only_scores))
        
        if belief_only_action != temporal_only_action and self.temporal_weight > 0:
            self.temporal_influenced_selections += 1
            
        if (belief_only_action != action_without and 
            temporal_only_action != action_without and 
            not self.disable_prediction and 
            self.prediction_weight > 0):
            self.predictive_influenced_selections += 1

        self.last_selected_band = action_with
        return action_with

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        
        valid_relationships = sum(1 for r in self.relationships.values() if r.confidence() > 0.5)
        sum_rel_conf = sum(r.confidence() for r in self.relationships.values())
        
        state.update({
            "pattern_influenced_selections": self.pattern_influenced_selections,
            "valid_pattern_relationships": valid_relationships,
            "pattern_candidates_total": self.pattern_candidates_total,
            "competing_candidates_count": self.competing_candidates_count,
            "pattern_confidence_mean": sum_rel_conf / max(1, len(self.relationships))
        })
        return state
