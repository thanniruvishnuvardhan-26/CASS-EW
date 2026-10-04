import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
from algorithms.phase7_signal_pattern import PatternAwareScheduler
from algorithms.rf_knowledge_map import RFKnowledgeMap, BandKnowledgeEntry
from algorithms.temporal_interval_analyzer import TemporalIntervalAnalyzer
from algorithms.frequency_hop_predictor import FrequencyHopPredictor
from algorithms.rf_change_detector import RFEnvironmentChangeDetector, EnvironmentChangeEvent


class ActionExplanation:
    """Detailed mathematical explanation of a scheduler decision."""
    def __init__(
        self,
        receiver_id: str,
        band: int,
        dwell: int,
        belief_score: float,
        temporal_score: float,
        prediction_score: float,
        pattern_score: float,
        spatial_evidence: float,
        uncertainty_score: float,
        staleness_score: float,
        hop_score: float = 0.0,
        env_change_penalty: float = 0.0,
        total_priority: float = 0.0,
        is_exploration: bool = False,
        is_anti_starvation: bool = False,
        reason: str = "",
        priority_list: Optional[List[Dict[str, Any]]] = None
    ):
        self.receiver_id = receiver_id
        self.band = band
        self.dwell = dwell
        self.belief_score = belief_score
        self.temporal_score = temporal_score
        self.prediction_score = prediction_score
        self.pattern_score = pattern_score
        self.spatial_evidence = spatial_evidence
        self.uncertainty_score = uncertainty_score
        self.staleness_score = staleness_score
        self.hop_score = hop_score
        self.env_change_penalty = env_change_penalty
        self.total_priority = total_priority
        self.is_exploration = is_exploration
        self.is_anti_starvation = is_anti_starvation
        self.reason = reason
        self.priority_list = priority_list if priority_list is not None else []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receiver": self.receiver_id,
            "band": self.band,
            "dwell": self.dwell,
            "belief_contribution": float(self.belief_score),
            "temporal_contribution": float(self.temporal_score),
            "prediction_contribution": float(self.prediction_score),
            "pattern_contribution": float(self.pattern_score),
            "spatial_contribution": float(self.spatial_evidence),
            "uncertainty_contribution": float(self.uncertainty_score),
            "staleness_contribution": float(self.staleness_score),
            "hop_contribution": float(self.hop_score),
            "env_change_penalty": float(self.env_change_penalty),
            "total_priority": float(self.total_priority),
            "is_exploration": bool(self.is_exploration),
            "is_anti_starvation": bool(self.is_anti_starvation),
            "reason": str(self.reason),
            "priority_list": self.priority_list
        }

    def summary(self) -> str:
        if self.is_exploration:
            return f"Receiver: {self.receiver_id}, Band: {self.band}, Dwell: {self.dwell} | Reason: Epsilon exploration."
        if self.is_anti_starvation:
            return f"Receiver: {self.receiver_id}, Band: {self.band}, Dwell: {self.dwell} | Reason: Anti-starvation deadline triggered (staleness: {self.staleness_score:.0f})."
        return (
            f"Receiver: {self.receiver_id}, Band: {self.band}, Dwell: {self.dwell} | "
            f"Reason: {self.reason} (Total score: {self.total_priority:.3f})"
        )


class ReceiverSpatialProfile:
    def __init__(self):
        self.observation_count = 0
        self.detection_count = 0
        self.recent_signal_strength = 0.0
        self.last_observed_time = -1
        
    def update(self, time: int, detection: bool, strength: float = 0.0):
        self.observation_count += 1
        if detection:
            self.detection_count += 1
        if strength > 0:
            self.recent_signal_strength = strength
        self.last_observed_time = time
        
    def detection_rate(self) -> float:
        if self.observation_count == 0:
            return 0.5
        return self.detection_count / self.observation_count
        
    def freshness(self, current_time: int, decay_rate: float = 0.01) -> float:
        if self.last_observed_time < 0:
            return 0.0
        return np.exp(-decay_rate * (current_time - self.last_observed_time))
        
    def spatial_evidence(self, current_time: int) -> float:
        return self.detection_rate() * self.freshness(current_time)


class SpatialScheduler:
    """
    CASS-EW Cognitive Adaptive Multi-Receiver Spatial Scheduler with
    RF Knowledge Map, Principled Uncertainty, Anti-Starvation,
    Frequency-Hop Predictor, Interval PRI Analyzer, RF Environment-Change Detector,
    and Explainable Evidence Fusion.
    """
    def __init__(
        self,
        receiver_ids: List[Any],
        num_bands: int,
        spatial_weight: float = 0.2,
        seed: Optional[int] = None,
        epsilon: float = 0.1,
        base_scheduler_class: Optional[Any] = None,
        uncertainty_weight: float = 0.0,
        staleness_weight: float = 0.0,
        hop_weight: float = 0.25,
        max_revisit_interval: Optional[int] = None,
        **kwargs
    ):
        self.receiver_ids = [str(r) for r in receiver_ids]
        self.num_bands = num_bands
        self.spatial_weight = spatial_weight
        self.epsilon = epsilon
        self.seed = seed
        self.uncertainty_weight = uncertainty_weight
        self.staleness_weight = staleness_weight
        self.hop_weight = hop_weight
        self.max_revisit_interval = max_revisit_interval
        self.kwargs = kwargs

        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random.default_rng()
            
        if base_scheduler_class is None:
            base_scheduler_class = PatternAwareScheduler
        self.base_scheduler_class = base_scheduler_class
            
        self.schedulers = {}
        self.spatial_profiles = {}
        for rid in self.receiver_ids:
            r_seed = self.seed + hash(rid) % 10000 if self.seed is not None else None
            if base_scheduler_class == PatternAwareScheduler:
                self.schedulers[rid] = base_scheduler_class(num_bands=num_bands, seed=r_seed, epsilon=0.0, **kwargs)
            else:
                try:
                    self.schedulers[rid] = base_scheduler_class(num_bands=num_bands, seed=r_seed, **kwargs)
                except TypeError:
                    self.schedulers[rid] = base_scheduler_class(num_bands=num_bands)
                    
            self.spatial_profiles[rid] = [ReceiverSpatialProfile() for _ in range(num_bands)]

        # Knowledge Map integration
        self.knowledge_map = RFKnowledgeMap(
            receiver_ids=self.receiver_ids,
            num_bands=num_bands,
            max_revisit_threshold=max_revisit_interval or 50
        )

        # Intelligence Modules
        self.interval_analyzers = {b: TemporalIntervalAnalyzer() for b in range(num_bands)}
        self.hop_predictor = FrequencyHopPredictor(num_bands=num_bands)
        self.change_detector = RFEnvironmentChangeDetector(num_bands=num_bands)
            
        self.spatial_influenced_selections = 0
        self.exploration_selections = 0
        self.exploitation_selections = 0
        self.anti_starvation_selections = 0
        self.last_selected_action = None
        self.last_explanation: Optional[ActionExplanation] = None
        self.receiver_selection_count = {rid: 0 for rid in self.receiver_ids}
        self.receiver_band_selection_count = {rid: [0]*num_bands for rid in self.receiver_ids}
        self.switching_count = 0

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        for rid, sched in self.schedulers.items():
            r_seed = self.seed + hash(rid) % 10000 if self.seed is not None else None
            sched.reset(r_seed)
            self.spatial_profiles[rid] = [ReceiverSpatialProfile() for _ in range(self.num_bands)]
        
        self.knowledge_map.reset()
        for b in range(self.num_bands):
            self.interval_analyzers[b].reset()
        self.hop_predictor.reset()
        self.change_detector.reset()

        self.spatial_influenced_selections = 0
        self.exploration_selections = 0
        self.exploitation_selections = 0
        self.anti_starvation_selections = 0
        self.last_selected_action = None
        self.last_explanation = None
        self.receiver_selection_count = {rid: 0 for rid in self.receiver_ids}
        self.receiver_band_selection_count = {rid: [0]*self.num_bands for rid in self.receiver_ids}
        self.switching_count = 0

    def _get_obs_time(self, observation: Optional[Dict[str, Any]]) -> int:
        if not observation:
            return 0
        if 'dwell_end_time' in observation:
            return observation['dwell_end_time']
        if 'observation_time' in observation:
            return observation['observation_time']
        if 'time' in observation:
            return observation['time']
        return 0

    def update(self, observation: Dict[str, Any], action: Tuple[Any, int], result: bool):
        rid_raw, band = action
        rid = str(rid_raw)
        obs_time = self._get_obs_time(observation)
        
        # Update underlying receiver scheduler
        self.schedulers[rid].update(observation, band, result)
        
        # Update spatial profile
        strength = observation.get('signal_strength', 0.0) if observation else 0.0
        self.spatial_profiles[rid][band].update(obs_time, result, strength)

        # Update Intelligence Modules
        self.interval_analyzers[band].update(obs_time, result)
        if result:
            self.hop_predictor.record_detection(obs_time, band)

        # Environmental Change Detection
        sub = self.schedulers[rid]
        belief = float(sub.beliefs[band]) if hasattr(sub, 'beliefs') else 0.5
        prof = sub.profiles[band] if hasattr(sub, 'profiles') else None
        pred_score = float(prof.predictive_score(obs_time)) if prof and hasattr(prof, 'predictive_score') else 0.5
        pred_detected = (pred_score > 0.75)
        new_events = self.change_detector.process_observation(
            time=obs_time,
            band=band,
            detected=result,
            current_belief=belief,
            predicted_detection=pred_detected
        )

        # Update RF Knowledge Map
        pri_stats = self.interval_analyzers[band].get_pri_stats()
        temporal_score = float(prof.temporal_prediction_score(obs_time)) if prof else 0.5
        prediction_score = pred_score
        prediction_confidence = float(prof.prediction_confidence) if prof and hasattr(prof, 'prediction_confidence') else 0.0
        
        # Blend PRI confidence into prediction confidence if available
        pri_conf = pri_stats.get("periodicity_confidence", 0.0)
        if pri_conf > 0:
            prediction_confidence = max(prediction_confidence, pri_conf)

        # If a major environment change occurred on this band, dampen stale prediction confidence
        if any(ev.band == band and ev.severity in ["HIGH", "MEDIUM"] for ev in new_events):
            prediction_confidence *= 0.5

        spatial_ev = float(self.spatial_profiles[rid][band].spatial_evidence(obs_time))
        
        pattern_score = 0.0
        if hasattr(sub, 'relationships') and not getattr(sub, 'disable_pattern', False):
            max_rel_score = 0.0
            for other_b in range(self.num_bands):
                if other_b != band:
                    pair = (min(band, other_b), max(band, other_b))
                    if pair in sub.relationships:
                        rel = sub.relationships[pair]
                        conf = rel.confidence() * rel.freshness(obs_time, getattr(sub, 'freshness_decay_rate', 0.01))
                        other_prof = sub.profiles[other_b]
                        other_p_score = other_prof.predictive_score(obs_time)
                        score = conf * other_p_score
                        if score > max_rel_score:
                            max_rel_score = score
            pattern_score = max_rel_score

        self.knowledge_map.record_scan_result(
            receiver_id=rid,
            band=band,
            obs_time=obs_time,
            detected=result,
            belief=belief,
            temporal_score=temporal_score,
            prediction_score=prediction_score,
            prediction_confidence=prediction_confidence,
            pattern_score=pattern_score,
            spatial_evidence=spatial_ev
        )

    def _compute_candidate_breakdown(self, obs_time: int, hop_scores: np.ndarray) -> Dict[Tuple[str, int], Dict[str, Any]]:
        """Compute the truthful candidate priority breakdown for all (receiver, band) pairs."""
        breakdown = {}
        for rid in self.receiver_ids:
            sched = self.schedulers[rid]
            combined_scores, without_pattern_scores = sched.get_priorities(obs_time)
            
            for band in range(self.num_bands):
                priority_without_spatial = combined_scores[band]
                spatial_evidence = self.spatial_profiles[rid][band].spatial_evidence(obs_time)
                entry = self.knowledge_map.get_entry(rid, band)

                # Uncertainty and staleness scoring
                unc_term = self.uncertainty_weight * entry.uncertainty
                stale_norm = min(1.0, entry.staleness / max(1.0, float(self.max_revisit_interval or 50)))
                stale_term = self.staleness_weight * stale_norm
                hop_term = self.hop_weight * hop_scores[band]

                # Environment change dampening penalty
                env_penalty = 0.2 if entry.environment_change else 0.0

                priority_with_spatial = (
                    priority_without_spatial 
                    + self.spatial_weight * spatial_evidence 
                    + unc_term 
                    + stale_term
                    + hop_term
                    - env_penalty
                )

                breakdown[(rid, band)] = {
                    "priority_without_spatial": priority_without_spatial,
                    "spatial_evidence": spatial_evidence,
                    "uncertainty": entry.uncertainty,
                    "staleness": entry.staleness,
                    "hop_score": hop_scores[band],
                    "env_penalty": env_penalty,
                    "total": priority_with_spatial,
                    "belief": entry.belief,
                    "temporal": entry.temporal_score,
                    "prediction": entry.prediction_score,
                    "pattern": entry.pattern_score
                }
        return breakdown

    def select_action(self, observation: Optional[Any] = None) -> Tuple[Any, int]:
        obs_time = self._get_obs_time(observation)
        self.knowledge_map.update_staleness(obs_time)

        # Compute Hop Prediction Scores across bands
        hop_scores = np.zeros(self.num_bands, dtype=float)
        hop_pred = self.hop_predictor.predict_next_band()
        hop_conf = hop_pred["confidence"]
        if self.hop_predictor.last_detection_band is not None:
            last_band = self.hop_predictor.last_detection_band
            trans_probs = self.hop_predictor.get_transition_probabilities(last_band)
            hop_scores = trans_probs * hop_conf
        
        # 1. Anti-Starvation Check: enforce maximum revisit deadline if configured
        if self.max_revisit_interval is not None and self.max_revisit_interval > 0:
            starving_pairs = self.knowledge_map.get_starving_bands(self.max_revisit_interval)
            if starving_pairs:
                most_starving = max(starving_pairs, key=lambda p: self.knowledge_map.get_entry(p[0], p[1]).staleness)
                rid, band = most_starving
                self.anti_starvation_selections += 1
                entry = self.knowledge_map.get_entry(rid, band)
                action = (rid, band)

                # Compute candidate breakdown and priority list with starving action at 999.0
                cand_breakdown = self._compute_candidate_breakdown(obs_time, hop_scores)
                sorted_cands = sorted(
                    cand_breakdown.items(),
                    key=lambda item: 999.0 if (item[0][0] == rid and item[0][1] == band) else item[1]["total"],
                    reverse=True
                )
                p_list = [
                    {
                        "rank": idx + 1,
                        "receiver": c_rid,
                        "band": c_band,
                        "score": 999.0 if (c_rid == rid and c_band == band) else round(float(c_meta["total"]), 4),
                        "selected": (c_rid == rid and c_band == band),
                        "status": "SELECTED" if (c_rid == rid and c_band == band) else "",
                        "selection_mode": "anti_starvation" if (c_rid == rid and c_band == band) else "exploit"
                    }
                    for idx, ((c_rid, c_band), c_meta) in enumerate(sorted_cands)
                ]

                self.last_explanation = ActionExplanation(
                    receiver_id=rid,
                    band=band,
                    dwell=1,
                    belief_score=entry.belief,
                    temporal_score=entry.temporal_score,
                    prediction_score=entry.prediction_score,
                    pattern_score=entry.pattern_score,
                    spatial_evidence=entry.spatial_evidence,
                    uncertainty_score=entry.uncertainty,
                    staleness_score=float(entry.staleness),
                    hop_score=float(hop_scores[band]),
                    total_priority=999.0,
                    is_anti_starvation=True,
                    reason=f"Band {band} on {rid} exceeded maximum revisit interval ({entry.staleness} steps unvisited)",
                    priority_list=p_list
                )
                self._record_selection(action)
                return action

        # 2. Adaptive Exploration (base epsilon + low hop confidence boost if unpredictable)
        effective_epsilon = self.epsilon
        if self.hop_predictor.last_detection_band is not None and hop_conf < 0.2:
            # When hopping is unpredictable, increase exploration to discover new bands
            effective_epsilon = min(0.35, self.epsilon + 0.15)

        if self.rng.random() < effective_epsilon:
            self.exploration_selections += 1
            rid = self.rng.choice(self.receiver_ids)
            band = int(self.rng.integers(0, self.num_bands))
            action = (rid, band)
            entry = self.knowledge_map.get_entry(rid, band)

            # Compute underlying candidate breakdown; keep ranking by cognitive priority unchanged
            cand_breakdown = self._compute_candidate_breakdown(obs_time, hop_scores)
            sorted_cands = sorted(
                cand_breakdown.items(),
                key=lambda item: item[1]["total"],
                reverse=True
            )
            p_list = [
                {
                    "rank": idx + 1,
                    "receiver": c_rid,
                    "band": c_band,
                    "score": round(float(c_meta["total"]), 4),
                    "selected": (c_rid == rid and c_band == band),
                    "status": "SELECTED (EXPLORE)" if (c_rid == rid and c_band == band) else "",
                    "selection_mode": "explore" if (c_rid == rid and c_band == band) else "exploit"
                }
                for idx, ((c_rid, c_band), c_meta) in enumerate(sorted_cands)
            ]

            self.last_explanation = ActionExplanation(
                receiver_id=rid,
                band=band,
                dwell=1,
                belief_score=entry.belief,
                temporal_score=entry.temporal_score,
                prediction_score=entry.prediction_score,
                pattern_score=entry.pattern_score,
                spatial_evidence=entry.spatial_evidence,
                uncertainty_score=entry.uncertainty,
                staleness_score=float(entry.staleness),
                hop_score=float(hop_scores[band]),
                total_priority=0.0,
                is_exploration=True,
                reason="Exploration selection (adaptive epsilon-greedy)",
                priority_list=p_list
            )
            self._record_selection(action)
            return action

        # 3. Exploitation / Evidence Fusion
        self.exploitation_selections += 1
        
        best_action_with = None
        best_score_with = -1e9
        
        best_action_without = None
        best_score_without = -1e9

        candidate_breakdown = {}

        for rid in self.receiver_ids:
            sched = self.schedulers[rid]
            combined_scores, without_pattern_scores = sched.get_priorities(obs_time)
            
            for band in range(self.num_bands):
                priority_without_spatial = combined_scores[band]
                spatial_evidence = self.spatial_profiles[rid][band].spatial_evidence(obs_time)
                entry = self.knowledge_map.get_entry(rid, band)

                # Uncertainty and staleness scoring
                unc_term = self.uncertainty_weight * entry.uncertainty
                stale_norm = min(1.0, entry.staleness / max(1.0, float(self.max_revisit_interval or 50)))
                stale_term = self.staleness_weight * stale_norm
                hop_term = self.hop_weight * hop_scores[band]

                # Environment change dampening penalty
                env_penalty = 0.2 if entry.environment_change else 0.0

                priority_with_spatial = (
                    priority_without_spatial 
                    + self.spatial_weight * spatial_evidence 
                    + unc_term 
                    + stale_term
                    + hop_term
                    - env_penalty
                )

                candidate_breakdown[(rid, band)] = {
                    "priority_without_spatial": priority_without_spatial,
                    "spatial_evidence": spatial_evidence,
                    "uncertainty": entry.uncertainty,
                    "staleness": entry.staleness,
                    "hop_score": hop_scores[band],
                    "env_penalty": env_penalty,
                    "total": priority_with_spatial,
                    "belief": entry.belief,
                    "temporal": entry.temporal_score,
                    "prediction": entry.prediction_score,
                    "pattern": entry.pattern_score
                }
                
                if priority_without_spatial > best_score_without:
                    best_score_without = priority_without_spatial
                    best_action_without = (rid, band)
                    
                if priority_with_spatial > best_score_with:
                    best_score_with = priority_with_spatial
                    best_action_with = (rid, band)
        
        if best_action_with != best_action_without and self.spatial_weight > 0:
            self.spatial_influenced_selections += 1
            
        action = best_action_with
        best_rid, best_band = action
        best_meta = candidate_breakdown[action]

        # Generate reasons for chosen action
        reasons = []
        if best_meta["belief"] > 0.6:
            reasons.append(f"high belief ({best_meta['belief']:.2f})")
        if best_meta["temporal"] > 0.7:
            reasons.append(f"strong temporal rhythm ({best_meta['temporal']:.2f})")
        if best_meta["prediction"] > 0.7:
            reasons.append(f"high predicted activity ({best_meta['prediction']:.2f})")
        if best_meta["pattern"] > 0.4:
            reasons.append(f"correlated pattern support ({best_meta['pattern']:.2f})")
        if best_meta["spatial_evidence"] > 0.4 and self.spatial_weight > 0:
            reasons.append(f"spatial evidence ({best_meta['spatial_evidence']:.2f})")
        if best_meta["hop_score"] > 0.15:
            reasons.append(f"predicted frequency hop ({best_meta['hop_score']:.2f})")
        if best_meta["uncertainty"] > 0.7 and self.uncertainty_weight > 0:
            reasons.append(f"high uncertainty exploration ({best_meta['uncertainty']:.2f})")
        if best_meta["staleness"] > 20 and self.staleness_weight > 0:
            reasons.append(f"staleness refresh ({best_meta['staleness']} steps)")
        reason_str = " + ".join(reasons) if reasons else "Highest fused cognitive priority"

        # Build truthful priority list sorted descending by cognitive priority score
        sorted_candidates = sorted(
            candidate_breakdown.items(),
            key=lambda item: item[1]["total"],
            reverse=True
        )
        priority_list = [
            {
                "rank": idx + 1,
                "receiver": c_rid,
                "band": c_band,
                "score": round(float(c_meta["total"]), 4),
                "selected": (c_rid == best_rid and c_band == best_band),
                "status": "SELECTED" if (c_rid == best_rid and c_band == best_band) else "",
                "selection_mode": "exploit"
            }
            for idx, ((c_rid, c_band), c_meta) in enumerate(sorted_candidates)
        ]

        self.last_explanation = ActionExplanation(
            receiver_id=best_rid,
            band=best_band,
            dwell=1,
            belief_score=best_meta["belief"],
            temporal_score=best_meta["temporal"],
            prediction_score=best_meta["prediction"],
            pattern_score=best_meta["pattern"],
            spatial_evidence=best_meta["spatial_evidence"],
            uncertainty_score=best_meta["uncertainty"],
            staleness_score=float(best_meta["staleness"]),
            hop_score=float(best_meta["hop_score"]),
            env_change_penalty=float(best_meta["env_penalty"]),
            total_priority=best_meta["total"],
            is_exploration=False,
            is_anti_starvation=False,
            reason=reason_str,
            priority_list=priority_list
        )

        self._record_selection(action)
        return action

    def _record_selection(self, action: Tuple[Any, int]):
        if self.last_selected_action is not None and self.last_selected_action != action:
            self.switching_count += 1
            
        self.last_selected_action = action
        rid, band = action
        self.receiver_selection_count[rid] += 1
        self.receiver_band_selection_count[rid][band] += 1

    def explain_last_decision(self) -> Dict[str, Any]:
        """Return the explanation dict for the most recent decision."""
        if self.last_explanation is not None:
            return self.last_explanation.to_dict()
        return {}

    def get_multi_receiver_metrics(self) -> Dict[str, Any]:
        """
        Compute comprehensive multi-receiver performance and utilization metrics:
        - receiver_utilization: Fraction of total scans allocated to each receiver.
        - receiver_coverage: Number of distinct bands visited by each receiver.
        - receiver_starvation: Current staleness / unvisited count per receiver.
        - per_receiver_detection_rate: Empirical detection success rate per receiver.
        """
        total_scans = sum(self.receiver_selection_count.values())
        utilization = {}
        coverage = {}
        starvation = {}
        detection_rates = {}

        for rid in self.receiver_ids:
            scans = self.receiver_selection_count[rid]
            utilization[rid] = (scans / total_scans) if total_scans > 0 else 0.0
            
            # Bands visited at least once by this receiver
            bands_visited = sum(1 for cnt in self.receiver_band_selection_count[rid] if cnt > 0)
            coverage[rid] = bands_visited / float(self.num_bands)

            # Receiver starvation: average staleness across all bands on this receiver
            entries = [self.knowledge_map.get_entry(rid, b) for b in range(self.num_bands)]
            avg_stale = float(np.mean([e.staleness for e in entries])) if entries else 0.0
            starvation[rid] = avg_stale

            # Empirical detection rate from spatial profiles
            obs_cnt = sum(self.spatial_profiles[rid][b].observation_count for b in range(self.num_bands))
            det_cnt = sum(self.spatial_profiles[rid][b].detection_count for b in range(self.num_bands))
            detection_rates[rid] = (det_cnt / obs_cnt) if obs_cnt > 0 else 0.0

        return {
            "total_scans": total_scans,
            "receiver_utilization": utilization,
            "receiver_coverage": coverage,
            "receiver_starvation": starvation,
            "per_receiver_detection_rate": detection_rates,
            "switching_count": self.switching_count
        }

    def get_state(self) -> Dict[str, Any]:
        state = {
            "exploration_selections": self.exploration_selections,
            "exploitation_selections": self.exploitation_selections,
            "anti_starvation_selections": self.anti_starvation_selections,
            "spatial_influenced_selections": self.spatial_influenced_selections,
            "receiver_selection_count": self.receiver_selection_count.copy(),
            "switching_count": self.switching_count,
            "max_staleness": self.knowledge_map.get_max_staleness(),
            "multi_receiver_metrics": self.get_multi_receiver_metrics()
        }
        if self.last_explanation is not None:
            state["last_explanation"] = self.last_explanation.to_dict()
        return state
