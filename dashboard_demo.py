"""
CASS-EW SIH Demonstration Dashboard
Phase 10 — Judge-facing visual demonstration

This dashboard shows the CASS-EW pipeline in action:
- Current RF scan state
- Selected receiver & band
- Belief, temporal, prediction, pattern, spatial evidence
- Detection results
- Scan history & metrics

Usage:
    python dashboard_demo.py [--seed 42] [--steps 100]

NOTE: All displayed values are from SYNTHETIC simulation.
Ground truth is labeled separately and NOT shown to the scheduler.
"""

import argparse
import numpy as np
from config import get_default_config
from simulator.environment import Emitter
from simulator.multi_receiver import SpatialRFEnvironment, MultiReceiverSystem
from algorithms.phase8_spatial_scheduler import SpatialScheduler
from evaluation.intercept_time import InterceptTimeTracker


def run_demo(seed=42, max_steps=100):
    config = get_default_config()
    nb = config.environment.num_bands
    eps = config.phase4.epsilon

    env = SpatialRFEnvironment(num_bands=nb, seed=seed)
    scenario = [
        {"name": "E1_NearR1", "band": 1, "behavior": "periodic", "period": 7, "duty_cycle": 2, "position": (0, 0)},
        {"name": "E2_NearR2", "band": 3, "behavior": "periodic", "period": 11, "duty_cycle": 2, "position": (10, 0)},
        {"name": "E3_NearR3", "band": 5, "behavior": "intermittent", "activity_probability": 0.2, "position": (0, 10)},
    ]
    emitters = []
    for e_spec in scenario:
        kwargs = {k: v for k, v in e_spec.items() if k not in ['name', 'band', 'behavior', 'position']}
        e = Emitter(name=e_spec['name'], band=e_spec['band'], behavior=e_spec['behavior'], seed=seed, **kwargs)
        env.add_emitter(e, position=e_spec.get('position', (0,0)))
        emitters.append(e)

    rcfg = [
        {'id': 'R1', 'position': (0.0, 0.0), 'snr_threshold': 20.0},
        {'id': 'R2', 'position': (10.0, 0.0), 'snr_threshold': 20.0},
        {'id': 'R3', 'position': (0.0, 10.0), 'snr_threshold': 20.0},
    ]
    mrs = MultiReceiverSystem(env, rcfg)
    r_ids = [r['id'] for r in rcfg]

    scheduler = SpatialScheduler(
        receiver_ids=r_ids, num_bands=nb,
        spatial_weight=0.3, seed=seed, epsilon=eps,
        belief_weight=0.2, temporal_weight=0.15,
        prediction_weight=0.2, pattern_weight=0.15
    )
    scheduler.reset(seed=seed)

    print("=" * 70)
    print("   CASS-EW — Cognitive Adaptive Smart Scan — SIH Demonstration")
    print("   Mode: SYNTHETIC SIMULATION")
    print(f"   Seed: {seed}   |   Bands: {nb}   |   Receivers: {len(r_ids)}")
    print("=" * 70)
    print()

    total_det = 0
    total_obs = 0
    total_active = 0
    last_obs = None
    tracker = InterceptTimeTracker()

    steps = min(max_steps, config.environment.total_time)
    
    for step in range(steps):
        if env.time >= config.environment.total_time:
            break

        action = scheduler.select_action(last_obs)
        rid, band = action
        total_obs += 1

        detected = mrs.scan(rid, band, dwell_time=1)
        last_obs = mrs.get_receiver(rid).scan_history[-1]

        gt = env.get_ground_truth()
        was_active = band in gt['active_bands']
        if was_active:
            total_active += 1
        if detected and was_active:
            total_det += 1

        # EVALUATION LAYER: intercept-time tracking (ground truth, NOT scheduler)
        tracker.update_emitter_states(env.time, [e for e in emitters])
        tracker.record_scan(env.time, band, detected, was_active)

        scheduler.update(last_obs, action, detected)

        # --- Display every 10 steps ---
        if step % 10 == 0 or step < 5:
            ir = total_det / max(1, total_obs)
            
            # Gather scheduler internal state (observable only)
            sched_state = scheduler.get_state()
            beliefs = {rid_i: [f"{b:.2f}" for b in scheduler.schedulers[rid_i].beliefs[:nb]] 
                      for rid_i in r_ids}
            
            print(f"┌── Step {step:4d}  |  Time: {env.time}")
            print(f"│  ACTION:   Receiver={rid}  Band={band}")
            print(f"│  DETECTED: {detected}  |  Running Interception Rate: {ir:.3f}")
            print(f"│  Beliefs (per receiver × band):")
            for rx_id in r_ids:
                print(f"│    {rx_id}: {beliefs[rx_id]}")
            print(f"│  Spatial influenced: {sched_state.get('spatial_influenced_selections',0)}")
            print(f"│  Exploration: {sched_state.get('exploration_selections',0)} | Exploitation: {sched_state.get('exploitation_selections',0)}")
            
            # Ground truth label
            active_bands = gt.get('active_bands', [])
            print(f"│  ╌╌╌ SIMULATION GROUND TRUTH — NOT AVAILABLE TO SCHEDULER ╌╌╌")
            print(f"│  Active bands: {sorted(active_bands)}")
            print(f"└{'─'*60}")
            print()

    # Final summary
    tracker.finalize(env.time)
    ir = total_det / max(1, total_active)
    state = scheduler.get_state()
    it_results = tracker.get_results()
    print()
    print("=" * 70)
    print("   FINAL DEMONSTRATION SUMMARY")
    print("=" * 70)
    print(f"  Total observations:          {total_obs}")
    print(f"  Detections:                  {total_det}")
    print(f"  Interception rate:           {ir:.4f}")
    print(f"  Spatial influenced actions:  {state.get('spatial_influenced_selections', 0)}")
    print(f"  Exploration selections:      {state.get('exploration_selections', 0)}")
    print(f"  Exploitation selections:     {state.get('exploitation_selections', 0)}")
    print()
    print("  --- INTERCEPT TIME (Synthetic Evaluation) ---")
    print(f"  Eligible episodes:           {it_results['eligible_episodes']}")
    print(f"  Intercepted episodes:        {it_results['intercepted_episodes']}")
    print(f"  Missed episodes:             {it_results['missed_episodes']}")
    mit = it_results['mean_intercept_time']
    mdit = it_results['median_intercept_time']
    print(f"  Mean intercept time:         {f'{mit:.2f}' if mit is not None else 'N/A (all missed)'}")
    print(f"  Median intercept time:       {f'{mdit:.2f}' if mdit is not None else 'N/A (all missed)'}")
    print()
    print("  NOTE: All results are from SYNTHETIC simulation.")
    print("  No real RF/PDW data was used in this demonstration.")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    run_demo(seed=args.seed, max_steps=args.steps)
