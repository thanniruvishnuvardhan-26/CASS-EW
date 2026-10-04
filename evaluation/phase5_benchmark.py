import numpy as np
import copy
from config import get_default_config
from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver
from algorithms.sequential import SequentialScheduler
from algorithms.random_scheduler import RandomScheduler
from algorithms.adaptive_belief import AdaptiveBeliefScheduler
from algorithms.temporal_belief import TemporalBeliefScheduler
from algorithms.phase5_temporal_profile import MultiBandTemporalProfileScheduler

def run_evaluation(scheduler, config, seed):
    env = RFEnvironment(num_bands=config.environment.num_bands, seed=seed)
    for e_spec in config.environment.emitters:
        env.add_emitter(Emitter(
            name=e_spec.name,
            band=e_spec.band,
            behavior=e_spec.behavior,
            activity_probability=e_spec.activity_probability,
            hop_bands=e_spec.hop_bands,
            seed=seed
        ))

    receiver = VirtualReceiver(
        num_bands=config.receiver.num_bands,
        detection_probability=config.receiver.detection_probability,
        false_alarm_probability=config.receiver.false_alarm_probability,
        switching_time=1,
        seed=seed
    )
    
    scheduler.reset(seed=seed)
    
    total_detections = 0
    total_observations = 0
    total_active_target_times = 0
    false_alarms = 0
    
    action_counts = np.zeros(config.environment.num_bands, dtype=int)
    detection_counts = np.zeros(config.environment.num_bands, dtype=int)
    
    number_of_band_switches = 0
    last_action = None
    
    temporal_guided_selections = 0
    temporal_unguided_selections = 0
    
    while env.time < config.environment.total_time:
        last_obs = receiver.scan_history[-1] if receiver.scan_history else None
        action = scheduler.select_action(last_obs)
        if last_action is not None and last_action != action:
            number_of_band_switches += 1
        last_action = action
        
        # Check if temporal evidence existed for the chosen band (Phase 4 only)
        if hasattr(scheduler, 'temporal_model'):
            p, _, _ = scheduler.temporal_model.estimate_period(action)
            if p is not None:
                temporal_guided_selections += 1
            else:
                temporal_unguided_selections += 1
        
        action_counts[action] += 1
        total_observations += 1
        
        gt_state = env.get_ground_truth()
        was_active = action in gt_state['active_bands']
        if was_active:
            total_active_target_times += 1
            
        detected = receiver.scan(env, action, dwell_time=1)
        observation = receiver.scan_history[-1]
        
        if detected:
            detection_counts[action] += 1
            if was_active:
                total_detections += 1
            else:
                false_alarms += 1
                
        scheduler.update(observation, action, detected)

    interception_rate = total_detections / max(1, total_active_target_times)
    misses = total_active_target_times - total_detections
    miss_rate = 1.0 - interception_rate
    false_alarm_rate = false_alarms / max(1, total_observations)
    efficiency = total_detections / max(1, total_observations)
    
    total_dwell_time = sum([obs['effective_duration'] if isinstance(obs, dict) else obs.effective_duration for obs in receiver.scan_history])
    total_switching_time = number_of_band_switches * receiver.switching_time
    total_elapsed_rf_time = env.time
    overshoot = total_elapsed_rf_time - config.environment.total_time
    if overshoot < 0:
        overshoot = 0
    
    res = {
        "configured_total_time": config.environment.total_time,
        "initial_environment_time": 0,
        "final_environment_time": env.time,
        "total_observations": total_observations,
        "detections": total_detections,
        "false_alarms": false_alarms,
        "misses": misses,
        "interception_rate": interception_rate,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
        "efficiency": efficiency,
        "total_dwell_time": total_dwell_time,
        "total_switching_time": total_switching_time,
        "total_elapsed_rf_time": total_elapsed_rf_time,
        "overshoot": overshoot,
        "number_of_band_switches": number_of_band_switches,
        "per_band_observations": action_counts.tolist(),
        "per_band_detections": detection_counts.tolist(),
    }
    
    state = scheduler.get_state() if hasattr(scheduler, 'get_state') else {}
    
    if hasattr(scheduler, 'temporal_model'):
        res.update({
            "temporal_valid_bands": state.get("temporal_valid_bands", 0),
            "temporal_avg_intervals": state.get("temporal_avg_intervals", 0),
            "temporal_avg_period": state.get("temporal_avg_period", 0.0),
            "temporal_avg_mad": state.get("temporal_avg_mad", 0.0),
            "temporal_guided_selections": temporal_guided_selections,
            "temporal_unguided_selections": temporal_unguided_selections
        })
    elif hasattr(scheduler, 'profiles'):
        res.update({
            "temporal_valid_bands": state.get("temporal_valid_bands", 0),
            "temporal_avg_period": state.get("temporal_avg_period", 0.0),
            "temporal_avg_mad": state.get("temporal_avg_mad", 0.0),
            "temporal_avg_intervals": state.get("temporal_avg_intervals", 0),
            "temporal_avg_reliability": state.get("temporal_avg_reliability", 0.0),
            "temporal_avg_freshness": state.get("temporal_avg_freshness", 0.0),
            "exploration_selections": state.get("exploration_selections", 0),
            "exploitation_selections": state.get("exploitation_selections", 0),
            "temporal_influenced_selections": state.get("temporal_influenced_selections", 0),
            "number_of_bands_observed": sum(1 for x in action_counts if x > 0),
        })
        
    return res

def print_result(algo, run_results):
    for i, r in enumerate(run_results):
        print(f"\n[Seed {i} (Run {i+1})] Algorithm: {algo}")
        for k, v in r.items():
            if not isinstance(v, list):
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
    
    avg = {}
    for k in run_results[0].keys():
        if isinstance(run_results[0][k], list):
            avg[k] = np.mean([r[k] for r in run_results], axis=0).tolist()
        else:
            avg[k] = np.mean([r[k] for r in run_results])

    print(f"Algorithm: {algo}")
    print(f"configured_total_time: {avg['configured_total_time']:.1f}")
    print(f"initial_environment_time: {avg['initial_environment_time']:.1f}")
    print(f"final_environment_time: {avg['final_environment_time']:.1f}")
    print(f"total_dwell_time: {avg['total_dwell_time']:.1f}")
    print(f"total_switching_time: {avg['total_switching_time']:.1f}")
    print(f"total_elapsed_rf_time: {avg['total_elapsed_rf_time']:.1f}")
    if "overshoot" in avg:
        print(f"overshoot: {avg['overshoot']:.1f}")
    print(f"total_observations (scans): {avg['total_observations']:.1f}")
    print(f"number_of_band_switches: {avg['number_of_band_switches']:.1f}")
    
    print(f"detections: {avg['detections']:.1f}")
    print(f"false_alarms: {avg['false_alarms']:.1f}")
    print(f"misses: {avg['misses']:.1f}")
    print(f"interception_rate: {avg['interception_rate']:.4f}")
    print(f"false_alarm_rate: {avg['false_alarm_rate']:.4f}")
    print(f"miss_rate: {avg['miss_rate']:.4f}")
    print(f"efficiency: {avg['efficiency']:.4f}")
    
    if "temporal_valid_bands" in avg:
        print(f"temporal_valid_bands: {avg['temporal_valid_bands']:.1f}")
        print(f"temporal_avg_period: {avg['temporal_avg_period']:.2f}")
        print(f"temporal_avg_mad: {avg['temporal_avg_mad']:.2f}")
        print(f"temporal_avg_intervals: {avg['temporal_avg_intervals']:.1f}")
        if "temporal_guided_selections" in avg:
            print(f"temporal_guided_selections: {avg['temporal_guided_selections']:.1f}")
            print(f"temporal_unguided_selections: {avg['temporal_unguided_selections']:.1f}")
        if "temporal_avg_reliability" in avg:
            print(f"temporal_avg_reliability: {avg['temporal_avg_reliability']:.2f}")
            print(f"temporal_avg_freshness: {avg['temporal_avg_freshness']:.2f}")
            print(f"exploration_selections: {avg['exploration_selections']:.1f}")
            print(f"exploitation_selections: {avg['exploitation_selections']:.1f}")
            print(f"temporal_influenced_selections: {avg['temporal_influenced_selections']:.1f}")
            print(f"number_of_bands_observed: {avg['number_of_bands_observed']:.1f}")
        
    print("-" * 40)

def main():
    config = get_default_config()
    seeds = config.benchmark.benchmark_seeds
    
    print("==================================================")
    print("CASS-EW PHASE 5 BENCHMARK (SEEDS 42-46)")
    print("==================================================")
    
    seq_results = []
    rand_results = []
    adapt_results = []
    temporal_results = []
    phase5_results = []
    
    for seed in seeds:
        seq_sched = SequentialScheduler(num_bands=config.environment.num_bands)
        seq_results.append(run_evaluation(seq_sched, config, seed))
        
        rand_sched = RandomScheduler(num_bands=config.environment.num_bands, seed=seed)
        rand_results.append(run_evaluation(rand_sched, config, seed))
        
        adapt_sched = AdaptiveBeliefScheduler(
            num_bands=config.environment.num_bands,
            epsilon=config.phase4.epsilon,
            seed=seed
        )
        adapt_results.append(run_evaluation(adapt_sched, config, seed))
        
        temporal_sched = TemporalBeliefScheduler(
            num_bands=config.environment.num_bands,
            epsilon=config.phase4.epsilon,
            belief_weight=config.phase4.belief_weight,
            temporal_weight=config.phase4.temporal_weight,
            seed=seed
        )
        temporal_results.append(run_evaluation(temporal_sched, config, seed))
        phase5_sched = MultiBandTemporalProfileScheduler(
            num_bands=config.environment.num_bands,
            epsilon=config.phase4.epsilon,
            belief_weight=0.7,
            temporal_weight=0.3,
            seed=seed
        )
        phase5_results.append(run_evaluation(phase5_sched, config, seed))

        
    print_result("Sequential", seq_results)
    print_result("Random", rand_results)
    print_result("AdaptiveBelief (Phase 3)", adapt_results)
    print_result("TemporalBelief (Phase 4)", temporal_results)
    print_result("MultiBandTemporalProfile (Phase 5)", phase5_results)

    
    print("==================================================")
    print("TEMPORAL ABLATION STUDY")
    print("==================================================")
    
    ablations = [
        (1.0, 0.0, False, False, "Pure Belief"),
        (0.0, 1.0, False, False, "Pure Temporal"),
        (0.7, 0.3, False, False, "Hybrid Default (70/30)"),
        (0.7, 0.3, True, False, "Reliability Disabled"),
        (0.7, 0.3, False, True, "Freshness Disabled")
    ]
    
    for bw, tw, dr, df, name in ablations:
        res = []
        for seed in seeds:
            sched = MultiBandTemporalProfileScheduler(
                num_bands=config.environment.num_bands,
                epsilon=config.phase4.epsilon,
                belief_weight=bw,
                temporal_weight=tw,
                disable_reliability=dr,
                disable_freshness=df,
                seed=seed
            )
            res.append(run_evaluation(sched, config, seed))
        print_result(f"Ablation: {name}", res)

if __name__ == "__main__":
    main()
