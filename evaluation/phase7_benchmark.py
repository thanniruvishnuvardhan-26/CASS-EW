import numpy as np
import copy
from config import get_default_config
from simulator.environment import RFEnvironment, Emitter, EmitterConfig
from simulator.receiver import VirtualReceiver
from algorithms.sequential import SequentialScheduler
from algorithms.random_scheduler import RandomScheduler
from algorithms.adaptive_belief import AdaptiveBeliefScheduler
from algorithms.temporal_belief import TemporalBeliefScheduler
from algorithms.phase5_temporal_profile import MultiBandTemporalProfileScheduler
from algorithms.phase6_predictive_scheduler import PredictiveScheduler
from algorithms.phase7_signal_pattern import PatternAwareScheduler

def run_evaluation(scheduler, config, seed, scenario_emitters=None):
    env = RFEnvironment(num_bands=config.environment.num_bands, seed=seed)
    
    emitters = []
    if scenario_emitters is None:
        for e_spec in config.environment.emitters:
            emitters.append(Emitter(
                name=e_spec.name,
                band=e_spec.band,
                behavior=e_spec.behavior,
                activity_probability=e_spec.activity_probability,
                hop_bands=e_spec.hop_bands,
                seed=seed
            ))
    else:
        for e_spec in scenario_emitters:
            kwargs = {k: v for k, v in e_spec.items() if k not in ['name', 'band', 'behavior']}
            emitters.append(Emitter(
                name=e_spec['name'],
                band=e_spec['band'],
                behavior=e_spec['behavior'],
                seed=seed,
                **kwargs
            ))
            
    for e in emitters:
        env.add_emitter(e)

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
    
    emitter_detections = {e.name: 0 for e in emitters}
    emitter_opportunities = {e.name: 0 for e in emitters}
    emitter_observations = {e.name: 0 for e in emitters}
    
    number_of_band_switches = 0
    last_action = None
    
    while env.time < config.environment.total_time:
        last_obs = receiver.scan_history[-1] if receiver.scan_history else None
        action = scheduler.select_action(last_obs)
        if last_action is not None and last_action != action:
            number_of_band_switches += 1
        last_action = action
        
        total_observations += 1
        
        # Track per-emitter ops
        for e in emitters:
            if e.active:
                emitter_opportunities[e.name] += 1
                if e.band == action:
                    emitter_observations[e.name] += 1
        
        gt_state = env.get_ground_truth()
        was_active = action in gt_state['active_bands']
        if was_active:
            total_active_target_times += 1
            
        detected = receiver.scan(env, action, dwell_time=1)
        observation = receiver.scan_history[-1]
        
        if detected:
            if was_active:
                total_detections += 1
                for e in emitters:
                    if e.active and e.band == action:
                        emitter_detections[e.name] += 1
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
    
    for e in emitters:
        res[f"emitter_opp_{e.name}"] = emitter_opportunities[e.name]
        res[f"emitter_det_{e.name}"] = emitter_detections[e.name]
        res[f"emitter_obs_{e.name}"] = emitter_observations[e.name]
        res[f"emitter_int_rate_{e.name}"] = emitter_detections[e.name] / max(1, emitter_opportunities[e.name])
    
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
        
    if hasattr(scheduler, 'pattern_weight'):
        res.update({
            "pattern_influenced_selections": state.get("pattern_influenced_selections", 0),
            "valid_pattern_relationships": state.get("valid_pattern_relationships", 0),
            "pattern_candidates_total": state.get("pattern_candidates_total", 0),
            "competing_candidates_count": state.get("competing_candidates_count", 0),
            "pattern_confidence_mean": state.get("pattern_confidence_mean", 0.0)
        })

    return res

def print_result(algo, run_results):
    avg = {}
    for k in run_results[0].keys():
        if isinstance(run_results[0][k], (int, float)):
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
    
    # Per Emitter Metrics
    emitter_names = set([k[len("emitter_opp_"):] for k in avg.keys() if k.startswith("emitter_opp_")])
    for en in sorted(emitter_names):
        print(f"  Emitter {en}: Opportunities={avg[f'emitter_opp_{en}']:.1f}, Observations={avg[f'emitter_obs_{en}']:.1f}, Detections={avg[f'emitter_det_{en}']:.1f}, IntRate={avg[f'emitter_int_rate_{en}']:.4f}")
    
    if "prediction_opportunities" in avg:
        print(f"prediction_opportunities: {avg['prediction_opportunities']:.1f}")
        print(f"prediction_hits: {avg['prediction_hits']:.1f}")
        print(f"prediction_misses: {avg['prediction_misses']:.1f}")
        print(f"prediction_hit_rate: {avg['prediction_hit_rate']:.4f}")
        print(f"mean_absolute_prediction_error: {avg['mean_absolute_prediction_error']:.4f}")
        print(f"prediction_coverage: {avg['prediction_coverage']:.4f}")
        print(f"prediction_confidence_mean: {avg['prediction_confidence_mean']:.4f}")
        print(f"predictive_influenced_selections: {avg['predictive_influenced_selections']:.1f}")
        
    if "pattern_influenced_selections" in avg:
        print(f"pattern_influenced_selections: {avg['pattern_influenced_selections']:.1f}")
        print(f"valid_pattern_relationships: {avg['valid_pattern_relationships']:.1f}")
        print(f"pattern_candidates_total: {avg['pattern_candidates_total']:.1f}")
        print(f"competing_candidates_count: {avg['competing_candidates_count']:.1f}")
        print(f"pattern_confidence_mean: {avg['pattern_confidence_mean']:.4f}")
        
    print("-" * 40)

def main():
    config = get_default_config()
    seeds = config.benchmark.benchmark_seeds
    
    print("==================================================")
    print("CASS-EW PHASE 7 BENCHMARK (SEEDS 42-46)")
    print("Scenario: Multiple Periodic Emitters (Different Periods) + Intermittent")
    print("==================================================")
    
    scenario = [
        {"name": "Emitter1_Periodic", "band": 1, "behavior": "periodic", "period": 7, "duty_cycle": 2},
        {"name": "Emitter2_Periodic", "band": 3, "behavior": "periodic", "period": 11, "duty_cycle": 2},
        {"name": "Emitter3_Intermittent", "band": 5, "behavior": "intermittent", "activity_probability": 0.2},
    ]
    
    seq_results, rand_results, adapt_results, temporal_results, phase5_results, phase6_results, phase7_results = [], [], [], [], [], [], []
    
    for seed in seeds:
        seq_results.append(run_evaluation(SequentialScheduler(num_bands=config.environment.num_bands), config, seed, scenario))
        rand_results.append(run_evaluation(RandomScheduler(num_bands=config.environment.num_bands, seed=seed), config, seed, scenario))
        adapt_results.append(run_evaluation(AdaptiveBeliefScheduler(num_bands=config.environment.num_bands, epsilon=config.phase4.epsilon, seed=seed), config, seed, scenario))
        temporal_results.append(run_evaluation(TemporalBeliefScheduler(num_bands=config.environment.num_bands, epsilon=config.phase4.epsilon, belief_weight=config.phase4.belief_weight, temporal_weight=config.phase4.temporal_weight, seed=seed), config, seed, scenario))
        phase5_results.append(run_evaluation(MultiBandTemporalProfileScheduler(num_bands=config.environment.num_bands, epsilon=config.phase4.epsilon, belief_weight=0.7, temporal_weight=0.3, seed=seed), config, seed, scenario))
        phase6_results.append(run_evaluation(PredictiveScheduler(num_bands=config.environment.num_bands, epsilon=config.phase4.epsilon, belief_weight=0.3, temporal_weight=0.2, prediction_weight=0.3, seed=seed), config, seed, scenario))
        phase7_results.append(run_evaluation(PatternAwareScheduler(num_bands=config.environment.num_bands, epsilon=config.phase4.epsilon, belief_weight=0.25, temporal_weight=0.2, prediction_weight=0.3, pattern_weight=0.25, seed=seed), config, seed, scenario))
        
    print_result("Sequential", seq_results)
    print_result("Random", rand_results)
    print_result("AdaptiveBelief (Phase 3)", adapt_results)
    print_result("TemporalBelief (Phase 4)", temporal_results)
    print_result("MultiBandTemporalProfile (Phase 5)", phase5_results)
    print_result("PredictiveScheduler (Phase 6)", phase6_results)
    print_result("PatternAwareScheduler (Phase 7)", phase7_results)

    print("==================================================")
    print("ABLATIONS")
    print("==================================================")

    ablations = [
        ("Pure Belief", 1.0, 0.0, 0.0, 0.0),
        ("Phase 6 Predictive", 0.4, 0.3, 0.3, 0.0),
        ("Belief + Temporal", 0.5, 0.5, 0.0, 0.0),
        ("Belief + Temporal + Prediction", 0.3, 0.3, 0.4, 0.0),
        ("Pattern-only", 0.0, 0.0, 0.0, 1.0),
        ("Predictive + Pattern", 0.0, 0.0, 0.5, 0.5),
        ("Belief + Temporal + Prediction + Pattern", 0.25, 0.2, 0.3, 0.25),
    ]

    for name, bw, tw, pw, patw in ablations:
        ab_res = []
        for seed in seeds:
            ab_res.append(run_evaluation(PatternAwareScheduler(
                num_bands=config.environment.num_bands,
                epsilon=config.phase4.epsilon,
                belief_weight=bw,
                temporal_weight=tw,
                prediction_weight=pw,
                pattern_weight=patw,
                seed=seed
            ), config, seed, scenario))
        print_result(f"Ablation: {name}", ab_res)

if __name__ == '__main__':
    main()
