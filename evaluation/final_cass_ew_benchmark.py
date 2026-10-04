import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from config import get_default_config
from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver
from simulator.multi_receiver import SpatialRFEnvironment, MultiReceiverSystem
from algorithms.sequential import SequentialScheduler
from algorithms.random_scheduler import RandomScheduler
from algorithms.adaptive_belief import AdaptiveBeliefScheduler
from algorithms.temporal_belief import TemporalBeliefScheduler
from algorithms.phase5_temporal_profile import MultiBandTemporalProfileScheduler
from algorithms.phase6_predictive_scheduler import PredictiveScheduler
from algorithms.phase7_signal_pattern import PatternAwareScheduler
from algorithms.phase8_spatial_scheduler import SpatialScheduler
from evaluation.intercept_time import (
    run_intercept_time_eval_single,
    run_intercept_time_eval_multi,
)


# ---- Single-receiver evaluation (Phases 3-7) ----
def run_single_receiver_eval(scheduler, config, seed, scenario_emitters):
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

    total_detections = 0
    total_observations = 0
    total_active = 0
    false_alarms = 0
    switches = 0
    last_action = None

    while env.time < config.environment.total_time:
        last_obs = receiver.scan_history[-1] if receiver.scan_history else None
        action = scheduler.select_action(last_obs)
        if last_action is not None and last_action != action:
            switches += 1
        last_action = action
        total_observations += 1

        gt = env.get_ground_truth()
        was_active = action in gt['active_bands']
        if was_active:
            total_active += 1

        detected = receiver.scan(env, action, dwell_time=1)
        obs = receiver.scan_history[-1]
        if detected:
            if was_active:
                total_detections += 1
            else:
                false_alarms += 1
        scheduler.update(obs, action, detected)

    ir = total_detections / max(1, total_active)
    dwell = sum(o['effective_duration'] for o in receiver.scan_history)
    return {
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


# ---- Multi-receiver evaluation (Phase 8+) ----
def run_multi_receiver_eval(scheduler, config, seed, scenario_emitters, receivers_config):
    env = SpatialRFEnvironment(num_bands=config.environment.num_bands, seed=seed)
    emitters = []
    for e_spec in scenario_emitters:
        kwargs = {k: v for k, v in e_spec.items() if k not in ['name', 'band', 'behavior', 'position']}
        e = Emitter(name=e_spec['name'], band=e_spec['band'], behavior=e_spec['behavior'], seed=seed, **kwargs)
        env.add_emitter(e, position=e_spec.get('position', (0,0)))
        emitters.append(e)

    mrs = MultiReceiverSystem(env, receivers_config)
    scheduler.reset(seed=seed)

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
        action = scheduler.select_action(last_obs)
        rid, band = action
        if last_action is not None and last_action != action:
            switches += 1
        last_action = action
        total_observations += 1
        rx_obs[rid] += 1

        gt = env.get_ground_truth()
        was_active = band in gt['active_bands']
        if was_active:
            total_active += 1

        detected = mrs.scan(rid, band, dwell_time=1)
        last_obs = mrs.get_receiver(rid).scan_history[-1]
        if detected:
            if was_active:
                total_detections += 1
                rx_det[rid] += 1
            else:
                false_alarms += 1
        scheduler.update(last_obs, action, detected)

    ir = total_detections / max(1, total_active)
    dwell = sum(
        sum(o['effective_duration'] for o in mrs.get_receiver(r['id']).scan_history)
        for r in receivers_config
    )
    
    state = scheduler.get_state() if hasattr(scheduler, 'get_state') else {}
    return {
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


def print_result(name, results_list):
    avg = {}
    for k in results_list[0]:
        v = results_list[0][k]
        if isinstance(v, (int, float)):
            avg[k] = np.mean([r[k] for r in results_list])
    
    print(f"\n  Algorithm: {name}")
    for k in ["interception_rate", "false_alarm_rate", "miss_rate", "efficiency",
              "total_observations", "detections", "switching_count",
              "total_dwell_time", "total_elapsed_rf_time", "overshoot"]:
        if k in avg:
            print(f"    {k:30s} {avg[k]:.4f}")
    if "spatial_influenced" in avg:
        print(f"    {'spatial_influenced':30s} {avg['spatial_influenced']:.1f}")
    
    # Intercept-time & Coverage metrics
    if 'intercept_time' in results_list[0] and results_list[0]['intercept_time']:
        it_keys = ['eligible_episodes', 'intercepted_episodes', 'missed_episodes',
                   'mean_intercept_time', 'median_intercept_time', 'p90_intercept_time',
                   'worst_case_intercept_time', 'time_to_first_intercept',
                   'coverage', 'worst_case_staleness']
        print(f"    {'--- Intercept & Coverage ---':30s}")
        for k in it_keys:
            vals = [r['intercept_time'].get(k) for r in results_list if r.get('intercept_time')]
            numeric_vals = [v for v in vals if v is not None]
            if numeric_vals:
                avg_val = np.mean(numeric_vals)
                if k in ['coverage']:
                    print(f"    {k:30s} {avg_val*100.0:.1f}%")
                else:
                    print(f"    {k:30s} {avg_val:.2f}")
            else:
                print(f"    {k:30s} N/A")
        # Episode interception rate (from intercept-time tracker)
        ep_ir_vals = [r['intercept_time'].get('interception_rate', 0) for r in results_list if r.get('intercept_time')]
        if ep_ir_vals:
            print(f"    {'episode_interception_rate':30s} {np.mean(ep_ir_vals):.4f}")
    
    # Receiver allocation if available
    if "receiver_obs" in results_list[0]:
        all_rx = results_list[0]["receiver_obs"].keys()
        for rx in sorted(all_rx):
            obs_avg = np.mean([r["receiver_obs"][rx] for r in results_list])
            det_avg = np.mean([r["receiver_det"][rx] for r in results_list])
            print(f"    Receiver {rx}: Obs={obs_avg:.1f}, Det={det_avg:.1f}")
    print(f"  {'-'*45}")


def main():
    config = get_default_config()
    seeds = config.benchmark.benchmark_seeds
    nb = config.environment.num_bands
    eps = config.phase4.epsilon

    scenario = [
        {"name": "E1", "band": 1, "behavior": "periodic", "period": 7, "duty_cycle": 2, "position": (0, 0)},
        {"name": "E2", "band": 3, "behavior": "periodic", "period": 11, "duty_cycle": 2, "position": (10, 0)},
        {"name": "E3", "band": 5, "behavior": "intermittent", "activity_probability": 0.2, "position": (0, 10)},
        {"name": "E4", "band": 2, "behavior": "periodic", "period": 5, "duty_cycle": 1, "position": (5, 5)},
    ]
    rcfg = [
        {'id': 'R1', 'position': (0.0, 0.0), 'snr_threshold': 20.0},
        {'id': 'R2', 'position': (10.0, 0.0), 'snr_threshold': 20.0},
        {'id': 'R3', 'position': (0.0, 10.0), 'snr_threshold': 20.0},
    ]
    r_ids = [r['id'] for r in rcfg]

    print("=" * 60)
    print("CASS-EW SYSTEM BENCHMARK SUITE")
    print("Seeds: 42, 43, 44, 45, 46")
    print("=" * 60)

    # -------- SINGLE-RECEIVER ALGORITHMIC PROGRESSION --------
    print("\n--- Single-Receiver Algorithmic Progression ---")
    single_algos = {
        "Random Baseline": lambda s: RandomScheduler(nb, seed=s),
        "Sequential Scanner": lambda s: SequentialScheduler(nb),
        "Phase 3 (AdaptiveBelief)": lambda s: AdaptiveBeliefScheduler(nb, epsilon=eps, seed=s),
        "Phase 4 (TemporalBelief)": lambda s: TemporalBeliefScheduler(nb, epsilon=eps, belief_weight=0.7, temporal_weight=0.3, seed=s),
        "Phase 5 (MultiBandTemporal)": lambda s: MultiBandTemporalProfileScheduler(nb, epsilon=eps, belief_weight=0.7, temporal_weight=0.3, seed=s),
        "Phase 6 (Predictive)": lambda s: PredictiveScheduler(nb, epsilon=eps, belief_weight=0.3, temporal_weight=0.2, prediction_weight=0.3, seed=s),
        "Phase 7 (PatternAware)": lambda s: PatternAwareScheduler(nb, epsilon=eps, belief_weight=0.25, temporal_weight=0.2, prediction_weight=0.3, pattern_weight=0.25, seed=s),
    }
    for name, make_sched in single_algos.items():
        results = [run_intercept_time_eval_single(make_sched(s), config, s, scenario) for s in seeds]
        print_result(name, results)

    # -------- MULTI-RECEIVER ALGORITHMS & INTEGRATION --------
    print("\n--- Multi-Receiver Spatial & Full Cognitive Schedulers ---")
    multi_algos = {
        "Phase 8 (Spatial Scheduler Only, w=0.3)": lambda s: SpatialScheduler(r_ids, nb, spatial_weight=0.3, seed=s, epsilon=eps,
            belief_weight=0.2, temporal_weight=0.15, prediction_weight=0.2, pattern_weight=0.15),
        "Full Integrated Cognitive Scheduler (Knowledge Map + Anti-Starvation)": lambda s: SpatialScheduler(
            r_ids, nb, spatial_weight=0.3, seed=s, epsilon=eps,
            belief_weight=0.2, temporal_weight=0.15, prediction_weight=0.2, pattern_weight=0.15,
            uncertainty_weight=0.05, staleness_weight=0.1, max_revisit_interval=50),
    }
    for name, make_sched in multi_algos.items():
        results = [run_intercept_time_eval_multi(make_sched(s), config, s, scenario, rcfg) for s in seeds]
        print_result(name, results)

    # -------- ABLATIONS --------
    print("\n--- Component Ablations (Multi-Receiver) ---")
    ablations = [
        ("Belief Only",                   1.0, 0.0, 0.0, 0.0, 0.0),
        ("Belief + Temporal",             0.5, 0.5, 0.0, 0.0, 0.0),
        ("Belief + Temporal + Prediction", 0.3, 0.3, 0.4, 0.0, 0.0),
        ("+ Pattern",                     0.25,0.2, 0.3, 0.25,0.0),
        ("+ Spatial (Full Evidence Fusion)", 0.2, 0.15,0.2, 0.15,0.3),
    ]
    for name, bw, tw, pw, patw, sw in ablations:
        results = [run_intercept_time_eval_multi(
            SpatialScheduler(r_ids, nb, spatial_weight=sw, seed=s, epsilon=eps,
                belief_weight=bw, temporal_weight=tw, prediction_weight=pw, pattern_weight=patw),
            config, s, scenario, rcfg) for s in seeds]
        print_result(f"Ablation: {name} (bw={bw},tw={tw},pw={pw},patw={patw},sw={sw})", results)



if __name__ == '__main__':
    main()
