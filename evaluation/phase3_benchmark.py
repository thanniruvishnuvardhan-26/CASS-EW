import numpy as np
from config import get_default_config
from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver
from algorithms.sequential import SequentialScheduler
from algorithms.random_scheduler import RandomScheduler
from algorithms.adaptive_belief import AdaptiveBeliefScheduler

def run_evaluation(scheduler, config, seed):
    env = RFEnvironment(num_bands=config.environment.num_bands, seed=seed)
    for e_spec in config.environment.emitters:
        # Avoid the legacy behavior string to bypass __new__ if we want, 
        # but Emitter(name=..., band=..., behavior=...) is standard for legacy testing.
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
        false_alarm_probability=config.receiver.false_alarm_probability
    )
    
    scheduler.reset(seed=seed)
    
    total_detections = 0
    total_observations = 0
    total_active_target_times = 0
    false_alarms = 0
    
    action_counts = np.zeros(config.environment.num_bands, dtype=int)
    
    # We will step up to total_time
    while env.time < config.environment.total_time:
        action = scheduler.select_action()
        action_counts[action] += 1
        total_observations += 1
        
        # We need the ground truth for evaluation metrics
        # The scheduler does NOT see this.
        gt_state = env.get_ground_truth()
        was_active = action in gt_state['active_bands']
        if was_active:
            total_active_target_times += 1
            
        detected = receiver.scan(env, action, dwell_time=1)
        
        if detected:
            if was_active:
                total_detections += 1
            else:
                false_alarms += 1
                
        scheduler.update(None, action, detected)

    interception_rate = total_detections / max(1, total_active_target_times)
    miss_rate = 1.0 - interception_rate
    false_alarm_rate = false_alarms / total_observations
    efficiency = total_detections / total_observations
    
    return {
        "total_observations": total_observations,
        "detections": total_detections,
        "interception_rate": interception_rate,
        "miss_rate": miss_rate,
        "false_alarm_rate": false_alarm_rate,
        "efficiency": efficiency,
        "action_counts": action_counts.tolist()
    }

def main():
    config = get_default_config()
    seeds = config.benchmark.benchmark_seeds
    
    print("CASS-EW PHASE 3 COMMON-TRACE BENCHMARK")
    print("========================================")
    
    results = {
        "Sequential": [],
        "Random": [],
        "Adaptive": []
    }
    
    for seed in seeds:
        # 1. Sequential
        seq_sched = SequentialScheduler(num_bands=config.environment.num_bands)
        results["Sequential"].append(run_evaluation(seq_sched, config, seed))
        
        # 2. Random
        rand_sched = RandomScheduler(num_bands=config.environment.num_bands, seed=seed)
        results["Random"].append(run_evaluation(rand_sched, config, seed))
        
        # 3. Adaptive Belief
        adapt_sched = AdaptiveBeliefScheduler(
            num_bands=config.environment.num_bands,
            initial_belief=config.phase3.initial_belief,
            belief_hit_update=config.phase3.belief_hit_update,
            belief_miss_update=config.phase3.belief_miss_update,
            epsilon=config.phase3.epsilon,
            seed=seed
        )
        results["Adaptive"].append(run_evaluation(adapt_sched, config, seed))

    # Aggregate and print
    print(f"{'Algorithm':<15} | {'Interception':<12} | {'False Alarm':<12} | {'Miss Rate':<10} | {'Efficiency':<10}")
    print("-" * 75)
    
    for name, runs in results.items():
        avg_interception = np.mean([r["interception_rate"] for r in runs]) * 100
        avg_fa = np.mean([r["false_alarm_rate"] for r in runs]) * 100
        avg_miss = np.mean([r["miss_rate"] for r in runs]) * 100
        avg_eff = np.mean([r["efficiency"] for r in runs])
        print(f"{name:<15} | {avg_interception:>11.2f}% | {avg_fa:>11.2f}% | {avg_miss:>9.2f}% | {avg_eff:>10.4f}")
        
    # Print per-band observation counts for the last seed as an example
    print("\nAction Distribution (Seed 46):")
    for name, runs in results.items():
        print(f"{name:<15}: {runs[-1]['action_counts']}")

if __name__ == "__main__":
    main()
