"""
CASS-EW — Cognitive Adaptive Smart Scan for Electronic Warfare
Unified Pipeline Entry Point (Phase 10)

SIH Problem Statement: 26055

Modes:
    synthetic   — Run synthetic RF environment evaluation
    replay      — Replay mock PDW fixture through the scheduler

Usage:
    python main.py --mode synthetic --seed 42
    python main.py --mode replay --pdw-file data/mock_pdws.json
"""

import argparse
import json
import numpy as np

from config import get_default_config
from simulator.environment import Emitter
from simulator.multi_receiver import SpatialRFEnvironment, MultiReceiverSystem
from simulator.receiver import VirtualReceiver
from algorithms.phase8_spatial_scheduler import SpatialScheduler
from algorithms.phase7_signal_pattern import PatternAwareScheduler
from data.real_pdws import (
    validate_dataset, normalize_to_observation, PDWReplayEngine,
    load_pdw_json, search_for_authorized_datasets
)


def run_synthetic(config, seed=42, verbose=True):
    """Run the full CASS-EW synthetic multi-receiver pipeline."""
    nb = config.environment.num_bands
    eps = config.phase4.epsilon

    # --- Environment Setup ---
    env = SpatialRFEnvironment(num_bands=nb, seed=seed)
    
    scenario = [
        {"name": "E1_NearR1", "band": 1, "behavior": "periodic", "period": 7, "duty_cycle": 2, "position": (0, 0)},
        {"name": "E2_NearR2", "band": 3, "behavior": "periodic", "period": 11, "duty_cycle": 2, "position": (10, 0)},
        {"name": "E3_NearR3", "band": 5, "behavior": "intermittent", "activity_probability": 0.2, "position": (0, 10)},
        {"name": "E4_Center", "band": 2, "behavior": "periodic", "period": 5, "duty_cycle": 1, "position": (5, 5)},
    ]
    
    emitters = []
    for e_spec in scenario:
        kwargs = {k: v for k, v in e_spec.items() if k not in ['name', 'band', 'behavior', 'position']}
        e = Emitter(name=e_spec['name'], band=e_spec['band'], behavior=e_spec['behavior'], seed=seed, **kwargs)
        env.add_emitter(e, position=e_spec.get('position', (0,0)))
        emitters.append(e)

    receivers_config = [
        {'id': 'R1', 'position': (0.0, 0.0), 'snr_threshold': 20.0},
        {'id': 'R2', 'position': (10.0, 0.0), 'snr_threshold': 20.0},
        {'id': 'R3', 'position': (0.0, 10.0), 'snr_threshold': 20.0},
    ]
    mrs = MultiReceiverSystem(env, receivers_config)
    r_ids = [r['id'] for r in receivers_config]

    # --- Scheduler: Full Integrated Phase 8 SpatialScheduler ---
    # Weights documented and fixed:
    #   belief=0.2, temporal=0.15, prediction=0.2, pattern=0.15, spatial=0.3
    scheduler = SpatialScheduler(
        receiver_ids=r_ids, num_bands=nb,
        spatial_weight=0.3, seed=seed, epsilon=eps,
        belief_weight=0.2, temporal_weight=0.15,
        prediction_weight=0.2, pattern_weight=0.15
    )
    scheduler.reset(seed=seed)

    # --- Main Loop ---
    total_detections = 0
    total_observations = 0
    total_active_target_times = 0
    false_alarms = 0
    last_obs = None
    last_action = None
    switches = 0
    receiver_det = {r['id']: 0 for r in receivers_config}
    receiver_obs = {r['id']: 0 for r in receivers_config}
    
    scan_log = []

    while env.time < config.environment.total_time:
        # --- Decision: select (receiver, band) ---
        action = scheduler.select_action(last_obs)
        rid, band = action
        
        if last_action is not None and last_action != action:
            switches += 1
        last_action = action

        total_observations += 1
        receiver_obs[rid] += 1
        
        gt = env.get_ground_truth()
        was_active = band in gt['active_bands']
        if was_active:
            total_active_target_times += 1

        # --- Scan ---
        detected = mrs.scan(rid, band, dwell_time=1)
        last_obs = mrs.get_receiver(rid).scan_history[-1]
        
        if detected:
            if was_active:
                total_detections += 1
                receiver_det[rid] += 1
            else:
                false_alarms += 1

        # --- Update scheduler ---
        explanation = scheduler.explain_last_decision() if hasattr(scheduler, 'explain_last_decision') else {}
        scheduler.update(last_obs, action, detected)
        
        scan_log.append({
            "time": env.time,
            "receiver": rid,
            "band": band,
            "detected": detected,
            "was_active": was_active,
            "explanation": explanation
        })

    # --- Results ---
    interception_rate = total_detections / max(1, total_active_target_times)
    results = {
        "mode": "synthetic",
        "seed": seed,
        "total_observations": total_observations,
        "detections": total_detections,
        "false_alarms": false_alarms,
        "interception_rate": interception_rate,
        "miss_rate": 1.0 - interception_rate,
        "false_alarm_rate": false_alarms / max(1, total_observations),
        "efficiency": total_detections / max(1, total_observations),
        "switching_count": switches,
        "elapsed_rf_time": env.time,
        "receiver_observations": receiver_obs,
        "receiver_detections": receiver_det,
    }
    
    state = scheduler.get_state()
    results["spatial_influenced_selections"] = state.get("spatial_influenced_selections", 0)
    results["exploration_selections"] = state.get("exploration_selections", 0)
    results["exploitation_selections"] = state.get("exploitation_selections", 0)
    results["anti_starvation_selections"] = state.get("anti_starvation_selections", 0)
    results["max_staleness"] = state.get("max_staleness", 0)

    if verbose:
        print(f"\n{'='*50}")
        print(f"CASS-EW SYNTHETIC MODE — Seed {seed}")
        print(f"{'='*50}")
        for k, v in results.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            elif isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    return results, scan_log


def run_replay(pdw_file, config, seed=42, verbose=True):
    """Run PDW replay mode."""
    nb = config.environment.num_bands
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"CASS-EW REPLAY MODE — File: {pdw_file}")
        print(f"{'='*50}")
    
    # Load and validate
    raw_records = load_pdw_json(pdw_file)
    valid, report = validate_dataset(raw_records)
    
    if verbose:
        summary = report.summary()
        print(f"  Records loaded:    {len(raw_records)}")
        print(f"  Records accepted:  {summary['accepted']}")
        print(f"  Records rejected:  {summary['rejected']}")
        print(f"  Records corrected: {summary['corrected']}")
    
    # Create scheduler (single-receiver for replay)
    scheduler = PatternAwareScheduler(num_bands=nb, seed=seed, epsilon=0.1)
    engine = PDWReplayEngine(valid, scheduler, num_bands=nb)
    log = engine.run_full_replay()
    metrics = engine.get_replay_metrics()
    
    if verbose:
        print(f"  Replayed:          {metrics['replayed_records']}")
        print(f"  Valid rate:         {metrics['valid_record_rate']:.4f}")
    
    return metrics, log


def main():
    parser = argparse.ArgumentParser(description="CASS-EW — Cognitive Adaptive Smart Scan")
    parser.add_argument('--mode', choices=['synthetic', 'replay'], default='synthetic',
                        help='Operating mode: synthetic or replay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--pdw-file', type=str, default=None, help='PDW JSON file for replay mode')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    args = parser.parse_args()

    config = get_default_config()
    
    # Check for authorized real data
    ds_status = search_for_authorized_datasets()
    if not args.quiet:
        print(f"Dataset status: {ds_status['status']}")

    if args.mode == 'synthetic':
        run_synthetic(config, seed=args.seed, verbose=not args.quiet)
    elif args.mode == 'replay':
        if args.pdw_file is None:
            print("ERROR: --pdw-file required for replay mode")
            return
        run_replay(args.pdw_file, config, seed=args.seed, verbose=not args.quiet)


if __name__ == '__main__':
    main()
