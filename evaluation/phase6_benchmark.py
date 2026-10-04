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
from algorithms.phase6_predictive_scheduler import PredictiveScheduler

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
    
    while env.time < config.environment.total_time:
        last_obs = receiver.scan_history[-1] if receiver.scan_history else None
        action = scheduler.select_action(last_obs)
        if last_action is not None and last_action != action:
            number_of_band_switches += 1
        last_action = action
        
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
    }
    
    state = scheduler.get_state() if hasattr(scheduler, 'get_state') else {}
    
    if hasattr(scheduler, 'prediction_weight'):
        res.update({
            "prediction_opportunities": state.get("prediction_opportunities", 0),
            "prediction_hits": state.get("prediction_hits", 0),
            "prediction_misses": state.get("prediction_misses", 0),
            "prediction_hit_rate": state.get("prediction_hit_rate", 0.0),
            "mean_absolute_prediction_error": state.get("mean_absolute_prediction_error", 0.0),
            "prediction_coverage": state.get("prediction_coverage", 0.0),
            "prediction_confidence_mean": state.get("prediction_confidence_mean", 0.0),
            "predictive_influenced_selections": state.get("predictive_influenced_selections", 0)
        })

    return res

def print_result(algo, run_results):
    avg = {}
    for k in run_results[0].keys():
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
    
    if "prediction_opportunities" in avg:
        print(f"prediction_opportunities: {avg['prediction_opportunities']:.1f}")
        print(f"prediction_hits: {avg['prediction_hits']:.1f}")
        print(f"prediction_misses: {avg['prediction_misses']:.1f}")
        print(f"prediction_hit_rate: {avg['prediction_hit_rate']:.4f}")
        print(f"mean_absolute_prediction_error: {avg['mean_absolute_prediction_error']:.4f}")
        print(f"prediction_coverage: {avg['prediction_coverage']:.4f}")
        print(f"prediction_confidence_mean: {avg['prediction_confidence_mean']:.4f}")
        print(f"predictive_influenced_selections: {avg['predictive_influenced_selections']:.1f}")
        
    print("-" * 40)

def main():
    config = get_default_config()
    seeds = config.benchmark.benchmark_seeds
    
    print("==================================================")
    print("CASS-EW PHASE 6 BENCHMARK (SEEDS 42-46)")
    print("==================================================")
    
    seq_results, rand_results, adapt_results, temporal_results, phase5_results, phase6_results = [], [], [], [], [], []
    
    for seed in seeds:
        seq_results.append(run_evaluation(SequentialScheduler(num_bands=config.environment.num_bands), config, seed))
        rand_results.append(run_evaluation(RandomScheduler(num_bands=config.environment.num_bands, seed=seed), config, seed))
        adapt_results.append(run_evaluation(AdaptiveBeliefScheduler(num_bands=config.environment.num_bands, epsilon=config.phase4.epsilon, seed=seed), config, seed))
        temporal_results.append(run_evaluation(TemporalBeliefScheduler(num_bands=config.environment.num_bands, epsilon=config.phase4.epsilon, belief_weight=config.phase4.belief_weight, temporal_weight=config.phase4.temporal_weight, seed=seed), config, seed))
        phase5_results.append(run_evaluation(MultiBandTemporalProfileScheduler(num_bands=config.environment.num_bands, epsilon=config.phase4.epsilon, belief_weight=0.7, temporal_weight=0.3, seed=seed), config, seed))
        phase6_results.append(run_evaluation(PredictiveScheduler(num_bands=config.environment.num_bands, epsilon=config.phase4.epsilon, belief_weight=0.4, temporal_weight=0.3, prediction_weight=0.3, seed=seed), config, seed))
        
    print_result("Sequential", seq_results)
    print_result("Random", rand_results)
    print_result("AdaptiveBelief (Phase 3)", adapt_results)
    print_result("TemporalBelief (Phase 4)", temporal_results)
    print_result("MultiBandTemporalProfile (Phase 5)", phase5_results)
    print_result("PredictiveScheduler (Phase 6)", phase6_results)

    print("==================================================")
    print("ABLATIONS")
    print("==================================================")

    ablations = [
        ("Pure Belief", 1.0, 0.0, 0.0),
        ("Pure Temporal", 0.0, 1.0, 0.0),
        ("Belief + Temporal", 0.5, 0.5, 0.0),
        ("Prediction only", 0.0, 0.0, 1.0),
        ("Belief + Prediction", 0.5, 0.0, 0.5),
        ("Temporal + Prediction", 0.0, 0.5, 0.5),
        ("Belief + Temporal + Prediction", 0.4, 0.3, 0.3),
    ]

    for name, bw, tw, pw in ablations:
        ab_res = []
        for seed in seeds:
            ab_res.append(run_evaluation(PredictiveScheduler(
                num_bands=config.environment.num_bands,
                epsilon=config.phase4.epsilon,
                belief_weight=bw,
                temporal_weight=tw,
                prediction_weight=pw,
                seed=seed
            ), config, seed))
        print_result(f"Ablation: {name}", ab_res)

if __name__ == '__main__':
    main()
