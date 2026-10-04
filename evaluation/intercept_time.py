"""
CASS-EW — Event-Based Intercept Time Evaluator
================================================

EVALUATION-ONLY module. This module accesses hidden ground truth
to measure how quickly the scheduler detects emitter activations.
It is NOT part of the scheduler decision path.

DEFINITIONS
-----------

Intercept Time:
    The number of time steps from the START of an emitter episode
    to the first valid interception (true-positive detection on
    the emitter's active band by the receiver scanning that band
    while the emitter is still active within the same episode).

    intercept_time = first_interception_time - episode_start_time

    A missed episode has NO intercept time. Missed episodes are
    excluded from latency statistics (mean, median, P90) and
    reported separately.

Emitter Episode:
    A contiguous run of one or more time steps during which an
    emitter is active (emitter.active == True) on a given band.

    - Periodic emitter (period=P, duty_cycle=D):
        Each duty-cycle ON window (D consecutive active steps per
        period) is one episode.

    - Intermittent emitter:
        Each contiguous run of active steps is one episode. A single
        inactive step ends the current episode. The next active step
        starts a new episode.

    - Burst emitter:
        Each burst (burst_duration consecutive active steps) is one
        episode.

    - Hopping emitter:
        Episodes are tracked per-band. When the hopper hops to a
        new band and is active, that starts a new episode on the
        new band.

    - Persistent emitter:
        The entire simulation is one long episode.

    Episode boundaries are determined by observing the hidden
    emitter state transitions (active → inactive or band change).

Signal Model (documentation only — used by environment, not here):
    formula:    SNR = base_snr / (1 + distance)
    assumptions: free-space monotonic attenuation, synthetic only
    units:      arbitrary SNR units
    noise:      none (deterministic attenuation)
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np


class EmitterEpisode:
    """Represents a single activation episode of an emitter."""

    __slots__ = [
        'emitter_name', 'band', 'start_time', 'end_time',
        'first_interception_time', 'intercepted', 'episode_id'
    ]

    def __init__(self, emitter_name: str, band: int, start_time: int, episode_id: int):
        self.emitter_name = emitter_name
        self.band = band
        self.start_time = start_time
        self.end_time = None          # set when episode ends
        self.first_interception_time = None
        self.intercepted = False
        self.episode_id = episode_id

    @property
    def intercept_latency(self) -> Optional[int]:
        """Latency in time steps. None if not intercepted."""
        if not self.intercepted or self.first_interception_time is None:
            return None
        return self.first_interception_time - self.start_time

    @property
    def is_finished(self) -> bool:
        return self.end_time is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'emitter_name': self.emitter_name,
            'band': self.band,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'first_interception_time': self.first_interception_time,
            'intercepted': self.intercepted,
            'intercept_latency': self.intercept_latency,
            'episode_id': self.episode_id,
        }


class InterceptTimeTracker:
    """
    Tracks emitter episodes and intercept times using hidden ground truth.

    IMPORTANT: This is an EVALUATION component only.
    The scheduler MUST NOT access this tracker or any of its data.

    Usage:
        tracker = InterceptTimeTracker()
        # Before each time step:
        tracker.update_emitter_states(env.time, emitters)
        # After each scan:
        tracker.record_scan(env.time, scanned_band, detected, was_active)
    """

    def __init__(self):
        self.episodes: List[EmitterEpisode] = []
        self._active_episodes: Dict[str, EmitterEpisode] = {}  # emitter_name -> current episode
        self._episode_counter = 0
        self._last_emitter_state: Dict[str, Tuple[bool, int]] = {}  # name -> (was_active, band)

        self._scanned_bands: Set[int] = set()
        self._last_visit_per_band: Dict[int, int] = {}
        self._max_staleness_per_band: Dict[int, int] = {}
        self._time_to_first_intercept: Optional[int] = None
        self._simulation_start_time: Optional[int] = None
        self._num_bands: int = 10

    def reset(self):
        """Reset all tracking state."""
        self.episodes = []
        self._active_episodes = {}
        self._episode_counter = 0
        self._last_emitter_state = {}
        self._scanned_bands = set()
        self._last_visit_per_band = {}
        self._max_staleness_per_band = {}
        self._time_to_first_intercept = None
        self._simulation_start_time = None

    def update_emitter_states(self, current_time: int, emitters: list):
        """
        Called BEFORE each scan to update episode tracking from hidden ground truth.

        For each emitter, detects activation/deactivation transitions and
        creates/closes episodes accordingly.

        Args:
            current_time: The current environment time step.
            emitters: List of emitter objects (ground truth access).
        """
        if self._simulation_start_time is None:
            self._simulation_start_time = current_time

        for emitter in emitters:
            name = emitter.name
            is_active = emitter.active
            current_band = emitter.band

            prev = self._last_emitter_state.get(name, (False, current_band))
            was_active, prev_band = prev

            # Detect episode boundaries
            if is_active and (not was_active or current_band != prev_band):
                # New episode starting (activation or band change while active)
                # Close any existing episode for this emitter first
                if name in self._active_episodes:
                    self._active_episodes[name].end_time = current_time - 1
                    del self._active_episodes[name]

                self._episode_counter += 1
                ep = EmitterEpisode(
                    emitter_name=name,
                    band=current_band,
                    start_time=current_time,
                    episode_id=self._episode_counter
                )
                self.episodes.append(ep)
                self._active_episodes[name] = ep

            elif not is_active and was_active:
                # Episode ending
                if name in self._active_episodes:
                    self._active_episodes[name].end_time = current_time - 1
                    del self._active_episodes[name]

            self._last_emitter_state[name] = (is_active, current_band)

    def finalize(self, final_time: int):
        """Close any still-open episodes and compute terminal staleness at simulation end."""
        for name, ep in list(self._active_episodes.items()):
            ep.end_time = final_time
        self._active_episodes.clear()

        # Update final staleness for all tracked bands
        for b in range(self._num_bands):
            last_t = self._last_visit_per_band.get(b, self._simulation_start_time or 0)
            staleness = final_time - last_t
            if b not in self._max_staleness_per_band or staleness > self._max_staleness_per_band[b]:
                self._max_staleness_per_band[b] = staleness

    def record_scan(self, current_time: int, scanned_band: int,
                    detected: bool, was_truly_active: bool, num_bands: int = 10):
        """
        Called AFTER each scan to check if this constitutes a valid interception.

        A valid interception requires:
        1. The scanner is tuned to the emitter's active band.
        2. The receiver detected a signal (detected == True).
        3. The band was truly active (was_truly_active == True).

        Args:
            current_time: Environment time when scan occurred.
            scanned_band: The band the receiver scanned.
            detected: Whether the receiver reported a detection.
            was_truly_active: Ground truth — was an emitter active on this band.
            num_bands: Number of RF bands in environment.
        """
        self._num_bands = num_bands
        self._scanned_bands.add(scanned_band)

        # Track staleness for this band
        prev_visit = self._last_visit_per_band.get(scanned_band, self._simulation_start_time or current_time)
        staleness = current_time - prev_visit
        if scanned_band not in self._max_staleness_per_band or staleness > self._max_staleness_per_band[scanned_band]:
            self._max_staleness_per_band[scanned_band] = staleness
        self._last_visit_per_band[scanned_band] = current_time

        if not (detected and was_truly_active):
            return

        # Record first intercept across entire simulation
        if self._time_to_first_intercept is None and self._simulation_start_time is not None:
            self._time_to_first_intercept = current_time - self._simulation_start_time

        # Find all active episodes on this band that haven't been intercepted yet
        for ep in self._active_episodes.values():
            if ep.band == scanned_band and not ep.intercepted:
                ep.intercepted = True
                ep.first_interception_time = current_time

    def get_results(self) -> Dict[str, Any]:
        """
        Compute aggregate intercept-time and coverage metrics.

        Returns dict with:
            eligible_episodes: Total episodes observed
            intercepted_episodes: Episodes with at least one interception
            missed_episodes: Episodes with no interception
            interception_rate: intercepted / eligible
            miss_rate: missed / eligible
            mean_intercept_time: Mean latency over intercepted episodes only
            median_intercept_time: Median latency over intercepted episodes only
            p90_intercept_time: 90th percentile latency
            worst_case_intercept_time: Maximum latency among intercepted episodes
            time_to_first_intercept: Latency from simulation start to very first interception
            coverage: Proportion of bands scanned at least once
            worst_case_staleness: Maximum time any band went unvisited
            all_latencies: list of individual latencies (intercepted only)
            episode_details: list of per-episode dicts
        """
        eligible = len(self.episodes)
        intercepted_eps = [ep for ep in self.episodes if ep.intercepted]
        missed_eps = [ep for ep in self.episodes if not ep.intercepted]

        latencies = [ep.intercept_latency for ep in intercepted_eps
                     if ep.intercept_latency is not None]

        worst_case_lat = max(latencies) if latencies else None
        worst_case_stale = max(self._max_staleness_per_band.values()) if self._max_staleness_per_band else 0
        coverage = len(self._scanned_bands) / max(1, self._num_bands)
        ir = len(intercepted_eps) / max(1, eligible)

        result = {
            'eligible_episodes': eligible,
            'intercepted_episodes': len(intercepted_eps),
            'missed_episodes': len(missed_eps),
            'interception_rate': ir,
            'miss_rate': 1.0 - ir,
            'mean_intercept_time': float(np.mean(latencies)) if latencies else None,
            'median_intercept_time': float(np.median(latencies)) if latencies else None,
            'p90_intercept_time': float(np.percentile(latencies, 90)) if latencies else None,
            'worst_case_intercept_time': float(worst_case_lat) if worst_case_lat is not None else None,
            'time_to_first_intercept': self._time_to_first_intercept,
            'coverage': coverage,
            'worst_case_staleness': worst_case_stale,
            'all_latencies': latencies,
            'episode_details': [ep.to_dict() for ep in self.episodes],
        }
        return result


def run_intercept_time_eval_single(scheduler, config, seed, scenario_emitters):
    """
    Run a single-receiver evaluation that additionally tracks intercept time.

    This wraps the standard evaluation loop and adds an InterceptTimeTracker
    that observes hidden ground truth for episode/latency measurement.

    The scheduler receives ONLY observable information — exactly the same
    data path as run_single_receiver_eval.
    """
    from simulator.environment import RFEnvironment, Emitter
    from simulator.receiver import VirtualReceiver

    env = RFEnvironment(num_bands=config.environment.num_bands, seed=seed)
    emitters = []
    for e_spec in scenario_emitters:
        kwargs = {k: v for k, v in e_spec.items() if k not in ['name', 'band', 'behavior', 'position']}
        e = Emitter(name=e_spec['name'], band=e_spec['band'], behavior=e_spec['behavior'], seed=seed, **kwargs)
        emitters.append(e)
        env.add_emitter(e)

    receiver = VirtualReceiver(
        num_bands=config.receiver.num_bands,
        detection_probability=config.receiver.detection_probability,
        false_alarm_probability=config.receiver.false_alarm_probability,
        switching_time=1, seed=seed
    )
    scheduler.reset(seed=seed)
    tracker = InterceptTimeTracker()

    total_detections = 0
    total_observations = 0
    total_active = 0
    false_alarms = 0
    switches = 0
    last_action = None

    while env.time < config.environment.total_time:
        # 1. Scheduler receives only observable information
        last_obs = receiver.scan_history[-1] if receiver.scan_history else None
        action = scheduler.select_action(last_obs)
        if last_action is not None and last_action != action:
            switches += 1
        last_action = action
        total_observations += 1

        # 2. Get ground truth BEFORE scan (for evaluation only)
        gt = env.get_ground_truth()
        was_active = action in gt['active_bands']
        if was_active:
            total_active += 1

        # 3. Perform scan (this advances env.time via env.step())
        detected = receiver.scan(env, action, dwell_time=1)
        obs = receiver.scan_history[-1]

        # 4. EVALUATION LAYER: Update episode tracking with hidden truth
        #    This happens AFTER the scan, using ground truth that the
        #    scheduler never sees.
        tracker.update_emitter_states(env.time, emitters)
        tracker.record_scan(env.time, action, detected, was_active)

        # 5. Update scheduler with observable only
        if detected:
            if was_active:
                total_detections += 1
            else:
                false_alarms += 1
        scheduler.update(obs, action, detected)

    tracker.finalize(env.time)

    ir = total_detections / max(1, total_active)
    dwell = sum(o['effective_duration'] for o in receiver.scan_history)

    base_results = {
        "interception_rate": ir,
        "false_alarm_rate": false_alarms / max(1, total_observations),
        "miss_rate": 1.0 - ir,
        "efficiency": total_detections / max(1, total_observations),
        "total_observations": total_observations,
        "detections": total_detections,
        "switching_count": switches,
        "total_dwell_time": dwell,
        "total_elapsed_rf_time": env.time,
        "overshoot": max(0, env.time - config.environment.total_time),
    }

    intercept_results = tracker.get_results()
    base_results['intercept_time'] = intercept_results
    return base_results


def run_intercept_time_eval_multi(scheduler, config, seed, scenario_emitters, receivers_config):
    """
    Run a multi-receiver evaluation that additionally tracks intercept time.

    Same observable contract as run_multi_receiver_eval.
    """
    from simulator.environment import Emitter
    from simulator.multi_receiver import SpatialRFEnvironment, MultiReceiverSystem

    env = SpatialRFEnvironment(num_bands=config.environment.num_bands, seed=seed)
    emitters = []
    for e_spec in scenario_emitters:
        kwargs = {k: v for k, v in e_spec.items() if k not in ['name', 'band', 'behavior', 'position']}
        e = Emitter(name=e_spec['name'], band=e_spec['band'], behavior=e_spec['behavior'], seed=seed, **kwargs)
        env.add_emitter(e, position=e_spec.get('position', (0, 0)))
        emitters.append(e)

    mrs = MultiReceiverSystem(env, receivers_config)
    scheduler.reset(seed=seed)
    tracker = InterceptTimeTracker()

    total_detections = 0
    total_observations = 0
    total_active = 0
    false_alarms = 0
    switches = 0
    last_action = None
    last_obs = None
    rx_obs = {r['id']: 0 for r in receivers_config}
    rx_det = {r['id']: 0 for r in receivers_config}

    while env.time < config.environment.total_time:
        # 1. Scheduler receives only observable information
        action = scheduler.select_action(last_obs)
        rid, band = action
        if last_action is not None and last_action != action:
            switches += 1
        last_action = action
        total_observations += 1
        rx_obs[rid] += 1

        # 2. Ground truth (evaluation only)
        gt = env.get_ground_truth()
        was_active = band in gt['active_bands']
        if was_active:
            total_active += 1

        # 3. Scan (advances time)
        detected = mrs.scan(rid, band, dwell_time=1)
        last_obs = mrs.get_receiver(rid).scan_history[-1]

        # 4. EVALUATION LAYER: episode tracking
        tracker.update_emitter_states(env.time, emitters)
        tracker.record_scan(env.time, band, detected, was_active)

        # 5. Observable update
        if detected:
            if was_active:
                total_detections += 1
                rx_det[rid] += 1
            else:
                false_alarms += 1
        scheduler.update(last_obs, action, detected)

    tracker.finalize(env.time)

    ir = total_detections / max(1, total_active)
    dwell = sum(
        sum(o['effective_duration'] for o in mrs.get_receiver(r['id']).scan_history)
        for r in receivers_config
    )

    state = scheduler.get_state() if hasattr(scheduler, 'get_state') else {}
    base_results = {
        "interception_rate": ir,
        "false_alarm_rate": false_alarms / max(1, total_observations),
        "miss_rate": 1.0 - ir,
        "efficiency": total_detections / max(1, total_observations),
        "total_observations": total_observations,
        "detections": total_detections,
        "switching_count": switches,
        "total_dwell_time": dwell,
        "total_elapsed_rf_time": env.time,
        "overshoot": max(0, env.time - config.environment.total_time),
        "spatial_influenced": state.get("spatial_influenced_selections", 0),
        "receiver_obs": rx_obs,
        "receiver_det": rx_det,
    }

    intercept_results = tracker.get_results()
    base_results['intercept_time'] = intercept_results
    return base_results