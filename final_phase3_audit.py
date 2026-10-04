import numpy as np
from config import get_default_config
from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver
from algorithms.sequential import SequentialScheduler
from algorithms.random_scheduler import RandomScheduler
from algorithms.adaptive_belief import AdaptiveBeliefScheduler

def run_evaluation(scheduler, config, seed, epsilon=None):
    env = RFEnvironment(num_bands=config.environment.num_bands, seed=seed)
    for e_spec in config.environment.emitters:
        env.add_emitter(Emitter(
            name=e_spec.name,
            band=e_spec.band,
            behavior=e_spec.behavior,
            activity_probability=e_spec.activity_probability,
            hop_bands=e_spec.hop_bands
        ))

    receiver = VirtualReceiver(
        num_bands=config.receiver.num_bands,
        detection_probability=config.receiver.detection_probability,
        false_alarm_probability=config.receiver.false_alarm_probability,
        switching_time=1  # adding some switching time to make it non-zero for test
    )
    
    scheduler.reset(seed=seed)
    if epsilon is not None and hasattr(scheduler, 'epsilon'):
        scheduler.epsilon = epsilon
    
    total_detections = 0
    total_observations = 0
    total_active_target_times = 0
    false_alarms = 0
    
    action_counts = np.zeros(config.environment.num_bands, dtype=int)
    detection_counts = np.zeros(config.environment.num_bands, dtype=int)
    
    number_of_band_switches = 0
    last_action = None
    
    while env.time < config.environment.total_time:
        action = scheduler.select_action()
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
        
        if detected:
            detection_counts[action] += 1
            if was_active:
                total_detections += 1
            else:
                false_alarms += 1
                
        scheduler.update(None, action, detected)

    interception_rate = total_detections / max(1, total_active_target_times)
    misses = total_active_target_times - total_detections
    miss_rate = 1.0 - interception_rate
    false_alarm_rate = false_alarms / total_observations
    efficiency = total_detections / total_observations
    
    total_dwell_time = sum([obs['effective_duration'] if isinstance(obs, dict) else obs.effective_duration for obs in receiver.scan_history])
    total_switching_time = number_of_band_switches * receiver.switching_time
    
    return {
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
        "number_of_band_switches": number_of_band_switches,
        "per_band_observations": action_counts.tolist(),
        "per_band_detections": detection_counts.tolist()
    }

def print_result(algo, run_results):
    avg = {}
    for k in run_results[0].keys():
        if isinstance(run_results[0][k], list):
            avg[k] = np.mean([r[k] for r in run_results], axis=0).tolist()
        else:
            avg[k] = np.mean([r[k] for r in run_results])

    print(f"Algorithm: {algo}")
    print(f"total_observations: {avg['total_observations']:.1f}")
    print(f"detections: {avg['detections']:.1f}")
    print(f"false_alarms: {avg['false_alarms']:.1f}")
    print(f"misses: {avg['misses']:.1f}")
    print(f"interception_rate: {avg['interception_rate']:.4f}")
    print(f"false_alarm_rate: {avg['false_alarm_rate']:.4f}")
    print(f"miss_rate: {avg['miss_rate']:.4f}")
    print(f"efficiency: {avg['efficiency']:.4f}")
    print(f"total_dwell_time: {avg['total_dwell_time']:.1f}")
    print(f"total_switching_time: {avg['total_switching_time']:.1f}")
    print(f"number_of_band_switches: {avg['number_of_band_switches']:.1f}")
    print(f"per_band_observations: {avg['per_band_observations']}")
    print(f"per_band_detections: {avg['per_band_detections']}")
    print("-" * 40)

def main():
    config = get_default_config()
    seeds = config.benchmark.benchmark_seeds
    
    print("==================================================")
    print("1. EVALUATION (SEEDS 42-46)")
    print("==================================================")
    seq_results = []
    rand_results = []
    adapt_results = []
    
    for seed in seeds:
        seq_sched = SequentialScheduler(num_bands=config.environment.num_bands)
        seq_results.append(run_evaluation(seq_sched, config, seed))
        
        rand_sched = RandomScheduler(num_bands=config.environment.num_bands, seed=seed)
        rand_results.append(run_evaluation(rand_sched, config, seed))
        
        adapt_sched = AdaptiveBeliefScheduler(
            num_bands=config.environment.num_bands,
            initial_belief=config.phase3.initial_belief,
            belief_hit_update=config.phase3.belief_hit_update,
            belief_miss_update=config.phase3.belief_miss_update,
            epsilon=config.phase3.epsilon,
            seed=seed
        )
        adapt_results.append(run_evaluation(adapt_sched, config, seed))
        
    print_result("Sequential", seq_results)
    print_result("Random", rand_results)
    print_result("Adaptive (epsilon=0.1)", adapt_results)
    
    print("==================================================")
    print("2. ADAPTIVE SCHEDULER EPSILON VARIATION")
    print("==================================================")
    
    for eps in [0.0, 0.1, 1.0]:
        res = []
        for seed in seeds:
            adapt_sched = AdaptiveBeliefScheduler(
                num_bands=config.environment.num_bands,
                initial_belief=config.phase3.initial_belief,
                belief_hit_update=config.phase3.belief_hit_update,
                belief_miss_update=config.phase3.belief_miss_update,
                epsilon=eps,
                seed=seed
            )
            res.append(run_evaluation(adapt_sched, config, seed))
        print_result(f"Adaptive (epsilon={eps})", res)
        
    print("==================================================")
    print("3. AdaptiveBeliefScheduler.last_observed")
    print("==================================================")
    print("AdaptiveBeliefScheduler.last_observed represents exactly:")
    print("scheduler decision count (step_counter).")
    print("It is NOT RF time. The scheduler does not receive the current RFEnvironment time.")
    
    print("\n==================================================")
    print("4. Receiver -> Scheduler Contract")
    print("==================================================")
    print("Exact object/fields passed from VirtualReceiver to AdaptiveBeliefScheduler.update():")
    print("- observation: None")
    print("- action: int (the tuned band, e.g., 5)")
    print("- result: bool (the detection_result from VirtualReceiver.scan())")

if __name__ == "__main__":
    main()
