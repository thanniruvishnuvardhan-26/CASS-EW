"""
CASS-EW Phase 8 Benchmark
Multi-Receiver Spatial Scheduling Evaluation
Seeds: 42, 43, 44, 45, 46
Scenarios A-J as specified in Phase 8 requirements.
"""
import numpy as np
from config import get_default_config
from simulator.environment import Emitter, EmitterConfig
from simulator.multi_receiver import SpatialRFEnvironment, MultiReceiverSystem
from algorithms.phase8_spatial_scheduler import SpatialScheduler

def run_spatial_evaluation(scheduler, config, seed, scenario_emitters, receivers_config):
    base_env = SpatialRFEnvironment(num_bands=config.environment.num_bands, seed=seed)
    
    emitters = []
    for e_spec in scenario_emitters:
        kwargs = {k: v for k, v in e_spec.items() if k not in ['name', 'band', 'behavior', 'position']}
        e = Emitter(
            name=e_spec['name'],
            band=e_spec['band'],
            behavior=e_spec['behavior'],
            seed=seed,
            **kwargs
        )
        pos = e_spec.get('position', (0,0))
        base_env.add_emitter(e, position=pos)
        emitters.append(e)

    mrs = MultiReceiverSystem(base_env, receivers_config)
    scheduler.reset(seed=seed)
    
    total_detections = 0
    total_observations = 0
    total_active_target_times = 0
    false_alarms = 0
    
    emitter_detections = {e.name: 0 for e in emitters}
    emitter_opportunities = {e.name: 0 for e in emitters}
    
    receiver_observations = {r['id']: 0 for r in receivers_config}
    receiver_detections = {r['id']: 0 for r in receivers_config}
    
    number_of_band_switches = 0
    last_action = None
    last_obs = None
    
    while base_env.time < config.environment.total_time:
        action = scheduler.select_action(last_obs)
        if last_action is not None and last_action != action:
            number_of_band_switches += 1
        last_action = action
        rid, band = action
        total_observations += 1
        receiver_observations[rid] += 1
        
        for e in emitters:
            if e.active:
                emitter_opportunities[e.name] += 1
        
        gt_state = base_env.get_ground_truth()
        was_active = band in gt_state['active_bands']
        if was_active:
            total_active_target_times += 1
            
        detected = mrs.scan(rid, band, dwell_time=1)
        last_obs = mrs.get_receiver(rid).scan_history[-1]
        
        if detected:
            if was_active:
                total_detections += 1
                receiver_detections[rid] += 1
                for e in emitters:
                    if e.active and e.band == band:
                        emitter_detections[e.name] += 1
            else:
                false_alarms += 1
                
        scheduler.update(last_obs, action, detected)

    interception_rate = total_detections / max(1, total_active_target_times)
    miss_rate = 1.0 - interception_rate
    false_alarm_rate = false_alarms / max(1, total_observations)
    efficiency = total_detections / max(1, total_observations)
    
    total_dwell_time = sum([
        sum([obs['effective_duration'] for obs in mrs.get_receiver(r['id']).scan_history])
        for r in receivers_config
    ])
    
    total_switching_time = number_of_band_switches
    total_elapsed_rf_time = base_env.time
    overshoot = max(0, total_elapsed_rf_time - config.environment.total_time)
    
    # Receiver coverage: fraction of receivers that made at least one detection
    active_receivers = sum(1 for r in receivers_config if receiver_detections[r['id']] > 0)
    receiver_coverage = active_receivers / len(receivers_config)
    
    res = {
        "configured_total_time": config.environment.total_time,
        "final_environment_time": base_env.time,
        "total_observations": total_observations,
        "detections": total_detections,
        "false_alarms": false_alarms,
        "misses": total_active_target_times - total_detections,
        "interception_rate": interception_rate,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
        "efficiency": efficiency,
        "total_dwell_time": total_dwell_time,
        "total_switching_time": total_switching_time,
        "total_elapsed_rf_time": total_elapsed_rf_time,
        "overshoot": overshoot,
        "switching_count": number_of_band_switches,
        "receiver_coverage": receiver_coverage,
    }
    
    for e in emitters:
        res[f"emitter_opp_{e.name}"] = emitter_opportunities[e.name]
        res[f"emitter_det_{e.name}"] = emitter_detections[e.name]
        
    for r in receivers_config:
        res[f"receiver_obs_{r['id']}"] = receiver_observations[r['id']]
        res[f"receiver_det_{r['id']}"] = receiver_detections[r['id']]
    
    state = scheduler.get_state() if hasattr(scheduler, 'get_state') else {}
    res["spatial_influenced_selections"] = state.get("spatial_influenced_selections", 0)
    res["exploration_selections"] = state.get("exploration_selections", 0)
    res["exploitation_selections"] = state.get("exploitation_selections", 0)
    
    return res

def print_result(algo, run_results):
    avg = {}
    for k in run_results[0].keys():
        if isinstance(run_results[0][k], (int, float)):
            avg[k] = np.mean([r[k] for r in run_results])

    print(f"\nAlgorithm: {algo}")
    print(f"  configured_total_time:     {avg['configured_total_time']:.1f}")
    print(f"  total_elapsed_rf_time:     {avg['total_elapsed_rf_time']:.1f}")
    print(f"  overshoot:                 {avg['overshoot']:.1f}")
    print(f"  total_observations:        {avg['total_observations']:.1f}")
    print(f"  detections:                {avg['detections']:.1f}")
    print(f"  false_alarms:              {avg['false_alarms']:.1f}")
    print(f"  switching_count:           {avg['switching_count']:.1f}")
    print(f"  total_dwell_time:          {avg['total_dwell_time']:.1f}")
    print(f"  interception_rate:         {avg['interception_rate']:.4f}")
    print(f"  false_alarm_rate:          {avg['false_alarm_rate']:.4f}")
    print(f"  miss_rate:                 {avg['miss_rate']:.4f}")
    print(f"  efficiency:                {avg['efficiency']:.4f}")
    print(f"  receiver_coverage:         {avg['receiver_coverage']:.4f}")
    
    receiver_names = sorted(set(k[len("receiver_obs_"):] for k in avg if k.startswith("receiver_obs_")))
    for rn in receiver_names:
        print(f"    Receiver {rn}: Obs={avg[f'receiver_obs_{rn}']:.1f}, Det={avg[f'receiver_det_{rn}']:.1f}")
    
    emitter_names = sorted(set(k[len("emitter_opp_"):] for k in avg if k.startswith("emitter_opp_")))
    for en in emitter_names:
        print(f"    Emitter {en}: Opp={avg[f'emitter_opp_{en}']:.1f}, Det={avg[f'emitter_det_{en}']:.1f}")
    
    print(f"  spatial_influenced:        {avg.get('spatial_influenced_selections', 0):.1f}")
    print(f"  exploration_selections:    {avg.get('exploration_selections', 0):.1f}")
    print(f"  exploitation_selections:   {avg.get('exploitation_selections', 0):.1f}")
    print("-" * 50)

def main():
    config = get_default_config()
    seeds = config.benchmark.benchmark_seeds
    nb = config.environment.num_bands
    eps = config.phase4.epsilon
    
    # ==================== PRIMARY SCENARIO ====================
    print("=" * 60)
    print("CASS-EW PHASE 8 BENCHMARK (SEEDS 42-46)")
    print("PRIMARY SCENARIO: Spatially Distributed Emitters + 3 Receivers")
    print("=" * 60)
    
    scenario_d = [
        {"name": "E1_NearR1", "band": 1, "behavior": "periodic", "period": 7, "duty_cycle": 2, "position": (0, 0)},
        {"name": "E2_NearR2", "band": 3, "behavior": "periodic", "period": 11, "duty_cycle": 2, "position": (10, 0)},
        {"name": "E3_NearR3", "band": 5, "behavior": "intermittent", "activity_probability": 0.2, "position": (0, 10)},
        {"name": "E4_Center", "band": 2, "behavior": "periodic", "period": 5, "duty_cycle": 1, "position": (5, 5)},
    ]
    rcfg3 = [
        {'id': 'R1', 'position': (0.0, 0.0), 'snr_threshold': 20.0},
        {'id': 'R2', 'position': (10.0, 0.0), 'snr_threshold': 20.0},
        {'id': 'R3', 'position': (0.0, 10.0), 'snr_threshold': 20.0},
    ]
    r3_ids = [r['id'] for r in rcfg3]
    
    # ---------- Comparisons ----------
    algos = {
        "Random": lambda s: SpatialScheduler(r3_ids, nb, spatial_weight=0.0, seed=s, epsilon=1.0),
        "Sequential-like (eps=0, spatial=0)": lambda s: SpatialScheduler(r3_ids, nb, spatial_weight=0.0, seed=s, epsilon=0.0,
            belief_weight=0.0, temporal_weight=0.0, prediction_weight=0.0, pattern_weight=0.0),
        "Phase 7 Pattern-Aware (spatial=0)": lambda s: SpatialScheduler(r3_ids, nb, spatial_weight=0.0, seed=s, epsilon=eps,
            belief_weight=0.25, temporal_weight=0.2, prediction_weight=0.3, pattern_weight=0.25),
        "Phase 8 Spatial (w=0.3)": lambda s: SpatialScheduler(r3_ids, nb, spatial_weight=0.3, seed=s, epsilon=eps,
            belief_weight=0.2, temporal_weight=0.15, prediction_weight=0.2, pattern_weight=0.15),
    }
    
    for algo_name, make_sched in algos.items():
        results = []
        for seed in seeds:
            results.append(run_spatial_evaluation(make_sched(seed), config, seed, scenario_d, rcfg3))
        print_result(algo_name, results)
    
    # ==================== SCENARIOS A-J ====================
    print("\n" + "=" * 60)
    print("SCENARIOS A-J")
    print("=" * 60)
    
    scenarios = {
        "A: 1 emitter, 2 receivers": (
            [{"name": "E1", "band": 1, "behavior": "periodic", "period": 5, "duty_cycle": 2, "position": (0, 0)}],
            [{'id': 'R1', 'position': (0.0, 0.0), 'snr_threshold': 20.0},
             {'id': 'R2', 'position': (5.0, 0.0), 'snr_threshold': 20.0}]
        ),
        "B: 1 emitter, 3 receivers": (
            [{"name": "E1", "band": 2, "behavior": "periodic", "period": 7, "duty_cycle": 2, "position": (0, 0)}],
            [{'id': 'R1', 'position': (0.0, 0.0), 'snr_threshold': 20.0},
             {'id': 'R2', 'position': (5.0, 0.0), 'snr_threshold': 20.0},
             {'id': 'R3', 'position': (0.0, 5.0), 'snr_threshold': 20.0}]
        ),
        "C: 2 emitters, 2 receivers": (
            [{"name": "E1", "band": 1, "behavior": "periodic", "period": 5, "duty_cycle": 2, "position": (0, 0)},
             {"name": "E2", "band": 3, "behavior": "intermittent", "activity_probability": 0.3, "position": (8, 0)}],
            [{'id': 'R1', 'position': (0.0, 0.0), 'snr_threshold': 20.0},
             {'id': 'R2', 'position': (8.0, 0.0), 'snr_threshold': 20.0}]
        ),
        "D: Multiple emitters & receivers": (scenario_d, rcfg3),
        "E: Same band, multiple receivers": (
            [{"name": "E1", "band": 1, "behavior": "periodic", "period": 5, "duty_cycle": 2, "position": (3, 3)}],
            [{'id': 'R1', 'position': (0.0, 0.0), 'snr_threshold': 10.0},
             {'id': 'R2', 'position': (6.0, 6.0), 'snr_threshold': 10.0}]
        ),
        "F: Different bands, different receivers": (
            [{"name": "E1", "band": 0, "behavior": "periodic", "period": 5, "duty_cycle": 2, "position": (0, 0)},
             {"name": "E2", "band": 4, "behavior": "periodic", "period": 7, "duty_cycle": 2, "position": (10, 10)}],
            [{'id': 'R1', 'position': (0.0, 0.0), 'snr_threshold': 20.0},
             {'id': 'R2', 'position': (10.0, 10.0), 'snr_threshold': 20.0}]
        ),
        "G: Stale receiver evidence": (
            [{"name": "E1", "band": 1, "behavior": "intermittent", "activity_probability": 0.1, "position": (0, 0)}],
            [{'id': 'R1', 'position': (0.0, 0.0), 'snr_threshold': 20.0},
             {'id': 'R2', 'position': (20.0, 0.0), 'snr_threshold': 20.0}]
        ),
        "H: Competing receiver-band opportunities": (
            [{"name": "E1", "band": 1, "behavior": "periodic", "period": 3, "duty_cycle": 1, "position": (0, 0)},
             {"name": "E2", "band": 1, "behavior": "periodic", "period": 5, "duty_cycle": 2, "position": (10, 0)},
             {"name": "E3", "band": 2, "behavior": "periodic", "period": 7, "duty_cycle": 3, "position": (5, 5)}],
            [{'id': 'R1', 'position': (0.0, 0.0), 'snr_threshold': 20.0},
             {'id': 'R2', 'position': (10.0, 0.0), 'snr_threshold': 20.0},
             {'id': 'R3', 'position': (5.0, 5.0), 'snr_threshold': 20.0}]
        ),
        "I: Spatially separated emitters": (
            [{"name": "E1", "band": 0, "behavior": "periodic", "period": 5, "duty_cycle": 2, "position": (0, 0)},
             {"name": "E2", "band": 3, "behavior": "periodic", "period": 5, "duty_cycle": 2, "position": (50, 50)}],
            [{'id': 'R1', 'position': (0.0, 0.0), 'snr_threshold': 20.0},
             {'id': 'R2', 'position': (50.0, 50.0), 'snr_threshold': 20.0}]
        ),
        "J: Overlapping emitter activity": (
            [{"name": "E1", "band": 1, "behavior": "periodic", "period": 4, "duty_cycle": 2, "position": (0, 0)},
             {"name": "E2", "band": 1, "behavior": "periodic", "period": 6, "duty_cycle": 3, "position": (0, 0)}],
            [{'id': 'R1', 'position': (0.0, 0.0), 'snr_threshold': 10.0},
             {'id': 'R2', 'position': (5.0, 0.0), 'snr_threshold': 10.0}]
        ),
    }
    
    for scn_name, (emitters, rcfgs) in scenarios.items():
        rids = [r['id'] for r in rcfgs]
        results = []
        for seed in seeds:
            sched = SpatialScheduler(rids, nb, spatial_weight=0.3, seed=seed, epsilon=eps,
                belief_weight=0.2, temporal_weight=0.15, prediction_weight=0.2, pattern_weight=0.15)
            results.append(run_spatial_evaluation(sched, config, seed, emitters, rcfgs))
        print_result(f"Scenario {scn_name}", results)
    
    # ==================== ABLATIONS ====================
    print("\n" + "=" * 60)
    print("ABLATIONS")
    print("=" * 60)
    
    ablations = [
        ("No Spatial",            0.25, 0.2, 0.3, 0.25, 0.0),
        ("Spatial-Only",          0.0,  0.0, 0.0, 0.0,  1.0),
        ("Temporal + Spatial",    0.0,  0.5, 0.0, 0.0,  0.5),
        ("Predictive + Spatial",  0.0,  0.0, 0.5, 0.0,  0.5),
        ("Pattern + Spatial",     0.0,  0.0, 0.0, 0.5,  0.5),
        ("Full Combination",      0.2,  0.15,0.2, 0.15, 0.3),
    ]
    
    for name, bw, tw, pw, patw, sw in ablations:
        results = []
        for seed in seeds:
            sched = SpatialScheduler(r3_ids, nb, spatial_weight=sw, seed=seed, epsilon=eps,
                belief_weight=bw, temporal_weight=tw, prediction_weight=pw, pattern_weight=patw)
            results.append(run_spatial_evaluation(sched, config, seed, scenario_d, rcfg3))
        print_result(f"Ablation: {name} (bw={bw}, tw={tw}, pw={pw}, patw={patw}, sw={sw})", results)

if __name__ == '__main__':
    main()
