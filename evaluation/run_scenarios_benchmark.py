"""
CASS-EW Final Benchmark & Scenario Evaluation Suite.
Runs all schedulers across Scenarios S1 through S10 across multiple seeds (42, 43, 44, 45, 46).
Outputs results to results/final_benchmark.csv and results/final_benchmark.json.
"""

import sys
import os
import json
import csv
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_default_config
from simulator.scenarios import get_scenario_suite
from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver
from simulator.multi_receiver import SpatialRFEnvironment, MultiReceiverSystem
from evaluation.intercept_time import InterceptTimeTracker

from algorithms.sequential import SequentialScheduler
from algorithms.random_scheduler import RandomScheduler
from algorithms.adaptive_belief import AdaptiveBeliefScheduler
from algorithms.ucb_scheduler import UCB1Scheduler
from algorithms.rl_benchmark_adapter import RLBenchmarkAdapter
from algorithms.phase8_spatial_scheduler import SpatialScheduler


def evaluate_single_receiver(scheduler, emitters_spec, seed: int, total_time: int = 500, num_bands: int = 10) -> Dict[str, Any]:
    env = RFEnvironment(num_bands=num_bands, seed=seed)
    emitters = []
    for e_spec in emitters_spec:
        kwargs = {k: v for k, v in e_spec.items() if k not in ['name', 'band', 'behavior', 'position']}
        e = Emitter(name=e_spec['name'], band=e_spec['band'], behavior=e_spec['behavior'], seed=seed, **kwargs)
        emitters.append(e)
        env.add_emitter(e)

    receiver = VirtualReceiver(
        num_bands=num_bands,
        detection_probability=0.90,
        false_alarm_probability=0.05,
        switching_time=1,
        seed=seed
    )
    scheduler.reset(seed=seed)
    tracker = InterceptTimeTracker()

    total_detections = 0
    total_observations = 0
    total_active = 0
    false_alarms = 0
    switches = 0
    last_action = None

    while env.time < total_time:
        last_obs = receiver.scan_history[-1] if receiver.scan_history else None
        action = scheduler.select_action(last_obs)
        band = action[1] if isinstance(action, (tuple, list)) else action
        band = int(band)

        if last_action is not None and last_action != action:
            switches += 1
        last_action = action
        total_observations += 1

        gt = env.get_ground_truth()
        was_active = band in gt['active_bands']
        if was_active:
            total_active += 1

        detected = receiver.scan(env, band, dwell_time=1)
        obs = receiver.scan_history[-1]

        tracker.update_emitter_states(env.time, emitters)
        tracker.record_scan(env.time, band, detected, was_active, num_bands=num_bands)

        if detected:
            if was_active:
                total_detections += 1
            else:
                false_alarms += 1
        scheduler.update(obs, action, detected)

    tracker.finalize(env.time)
    it_res = tracker.get_results()

    ir = total_detections / max(1, total_active)
    dwell = sum(o['effective_duration'] for o in receiver.scan_history)

    return {
        "interception_rate": float(ir),
        "miss_rate": float(1.0 - ir),
        "false_alarm_rate": float(false_alarms / max(1, total_observations)),
        "mean_intercept_time": float(it_res.get('mean_intercept_time') or 0.0),
        "median_intercept_time": float(it_res.get('median_intercept_time') or 0.0),
        "p90_intercept_time": float(it_res.get('p90_intercept_time') or 0.0),
        "worst_case_intercept_time": float(it_res.get('worst_case_intercept_time') or 0.0),
        "efficiency": float(total_detections / max(1, total_observations)),
        "coverage": float(it_res.get('coverage') or 0.0),
        "worst_case_staleness": float(it_res.get('worst_case_staleness') or 0.0),
        "total_observations": total_observations,
        "detections": total_detections,
        "switching_count": switches,
        "receiver_utilization": {"R1": 1.0}
    }


def evaluate_multi_receiver(scheduler, emitters_spec, receivers_spec, seed: int, total_time: int = 500, num_bands: int = 10) -> Dict[str, Any]:
    env = SpatialRFEnvironment(num_bands=num_bands, seed=seed)
    emitters = []
    for e_spec in emitters_spec:
        kwargs = {k: v for k, v in e_spec.items() if k not in ['name', 'band', 'behavior', 'position']}
        e = Emitter(name=e_spec['name'], band=e_spec['band'], behavior=e_spec['behavior'], seed=seed, **kwargs)
        env.add_emitter(e, position=e_spec.get('position', (0.0, 0.0)))
        emitters.append(e)

    mrs = MultiReceiverSystem(env, receivers_spec)
    scheduler.reset(seed=seed)
    tracker = InterceptTimeTracker()

    total_detections = 0
    total_observations = 0
    total_active = 0
    false_alarms = 0
    switches = 0
    last_action = None
    last_obs = None
    rx_obs = {r['id']: 0 for r in receivers_spec}

    while env.time < total_time:
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

        tracker.update_emitter_states(env.time, emitters)
        tracker.record_scan(env.time, band, detected, was_active, num_bands=num_bands)

        if detected:
            if was_active:
                total_detections += 1
            else:
                false_alarms += 1
        scheduler.update(last_obs, action, detected)

    tracker.finalize(env.time)
    it_res = tracker.get_results()

    ir = total_detections / max(1, total_active)
    utilization = {rid: rx_obs[rid] / max(1, total_observations) for rid in rx_obs}

    return {
        "interception_rate": float(ir),
        "miss_rate": float(1.0 - ir),
        "false_alarm_rate": float(false_alarms / max(1, total_observations)),
        "mean_intercept_time": float(it_res.get('mean_intercept_time') or 0.0),
        "median_intercept_time": float(it_res.get('median_intercept_time') or 0.0),
        "p90_intercept_time": float(it_res.get('p90_intercept_time') or 0.0),
        "worst_case_intercept_time": float(it_res.get('worst_case_intercept_time') or 0.0),
        "efficiency": float(total_detections / max(1, total_observations)),
        "coverage": float(it_res.get('coverage') or 0.0),
        "worst_case_staleness": float(it_res.get('worst_case_staleness') or 0.0),
        "total_observations": total_observations,
        "detections": total_detections,
        "switching_count": switches,
        "receiver_utilization": utilization
    }


class MultiReceiverSchedulerAdapter:
    """Adapts single-receiver baselines to multi-receiver environments round-robin."""
    def __init__(self, make_fn, receiver_ids: List[str], seed: int = None):
        self.receiver_ids = receiver_ids
        self.schedulers = {rid: make_fn(seed) for rid in receiver_ids}
        self.current_idx = 0

    def reset(self, seed: int = None):
        for rid, s in self.schedulers.items():
            s.reset(seed)
        self.current_idx = 0

    def select_action(self, observation=None):
        rid = self.receiver_ids[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.receiver_ids)
        band = self.schedulers[rid].select_action(observation)
        band = band[1] if isinstance(band, (tuple, list)) else band
        return (rid, int(band))

    def update(self, observation, action, result):
        rid, band = action
        self.schedulers[rid].update(observation, band, result)


def run_comprehensive_benchmark(seeds: List[int] = [42, 43, 44, 45, 46], total_time: int = 500):
    scenarios = get_scenario_suite()
    num_bands = 10

    # Define algorithm instantiators
    def make_schedulers(r_ids, seed):
        if len(r_ids) == 1:
            return {
                "Sequential": SequentialScheduler(num_bands),
                "Random": RandomScheduler(num_bands, seed=seed),
                "AdaptiveBelief": AdaptiveBeliefScheduler(num_bands, epsilon=0.1, seed=seed),
                "UCB1": UCB1Scheduler(num_bands, c=1.414, seed=seed),
                "RL_Experimental": RLBenchmarkAdapter(num_bands, seed=seed),
                "CognitiveAdaptive_Integrated": SpatialScheduler(
                    r_ids, num_bands=num_bands, seed=seed, epsilon=0.1,
                    spatial_weight=0.0,
                    belief_weight=0.2, temporal_weight=0.15,
                    prediction_weight=0.2, pattern_weight=0.15,
                    uncertainty_weight=0.05, staleness_weight=0.1, max_revisit_interval=50
                )
            }
        else:
            return {
                "Sequential": MultiReceiverSchedulerAdapter(lambda s: SequentialScheduler(num_bands), r_ids, seed),
                "Random": MultiReceiverSchedulerAdapter(lambda s: RandomScheduler(num_bands, seed=s), r_ids, seed),
                "AdaptiveBelief": MultiReceiverSchedulerAdapter(lambda s: AdaptiveBeliefScheduler(num_bands, epsilon=0.1, seed=s), r_ids, seed),
                "UCB1": MultiReceiverSchedulerAdapter(lambda s: UCB1Scheduler(num_bands, c=1.414, seed=s), r_ids, seed),
                "RL_Experimental": MultiReceiverSchedulerAdapter(lambda s: RLBenchmarkAdapter(num_bands, seed=s), r_ids, seed),
                "CognitiveAdaptive_Integrated": SpatialScheduler(
                    r_ids, num_bands=num_bands, seed=seed, epsilon=0.1,
                    spatial_weight=0.3,
                    belief_weight=0.2, temporal_weight=0.15,
                    prediction_weight=0.2, pattern_weight=0.15,
                    uncertainty_weight=0.05, staleness_weight=0.1, max_revisit_interval=50
                )
            }

    raw_results = []
    summary_results = []

    print("============================================================")
    print("STARTING CASS-EW MULTI-SEED SCENARIO BENCHMARK (S1 - S10)")
    print(f"Seeds: {seeds}, Time per run: {total_time} steps")
    print("============================================================")

    for sc_name, sc_spec in scenarios.items():
        is_multi = len(sc_spec["receivers"]) > 1
        r_ids = [r["id"] for r in sc_spec["receivers"]]
        category = sc_spec.get("category", "tuning")

        print(f"\n---> Evaluating Scenario: {sc_name} ({sc_spec['name']}) [{category.upper()}]")

        for algo_name in ["Sequential", "Random", "AdaptiveBelief", "UCB1", "RL_Experimental", "CognitiveAdaptive_Integrated"]:
            algo_seed_runs = []

            for seed in seeds:
                schedulers = make_schedulers(r_ids, seed)
                sched = schedulers[algo_name]

                if not is_multi:
                    # Single receiver execution
                    metrics = evaluate_single_receiver(sched, sc_spec["emitters"], seed, total_time=total_time, num_bands=num_bands)
                else:
                    # Multi-receiver execution
                    metrics = evaluate_multi_receiver(sched, sc_spec["emitters"], sc_spec["receivers"], seed, total_time=total_time, num_bands=num_bands)

                record = {
                    "scenario": sc_name,
                    "scenario_id": sc_spec["id"],
                    "category": category,
                    "algorithm": algo_name,
                    "seed": seed,
                    **metrics
                }
                raw_results.append(record)
                algo_seed_runs.append(record)

            # Compute mean and std dev across seeds
            metric_keys = [
                "interception_rate", "miss_rate", "false_alarm_rate",
                "mean_intercept_time", "median_intercept_time", "p90_intercept_time",
                "worst_case_intercept_time", "efficiency", "coverage", "worst_case_staleness"
            ]
            agg = {
                "scenario": sc_name,
                "scenario_id": sc_spec["id"],
                "category": category,
                "algorithm": algo_name,
                "num_seeds": len(seeds)
            }
            for mk in metric_keys:
                vals = [r[mk] for r in algo_seed_runs]
                agg[f"{mk}_mean"] = float(np.mean(vals))
                agg[f"{mk}_std"] = float(np.std(vals))
                # 95% confidence interval margin (approx 1.96 * std / sqrt(n))
                agg[f"{mk}_ci95"] = float(1.96 * np.std(vals) / np.sqrt(len(vals)))

            summary_results.append(agg)
            print(f"  {algo_name:30s} | IR: {agg['interception_rate_mean']*100.0:5.1f}% (±{agg['interception_rate_std']*100.0:4.1f}%) | "
                  f"Miss: {agg['miss_rate_mean']*100.0:5.1f}% | Eff: {agg['efficiency_mean']*100.0:4.2f}% | "
                  f"Cov: {agg['coverage_mean']*100.0:5.1f}% | WorstStale: {agg['worst_case_staleness_mean']:5.1f}")

    # Save to results/final_benchmark.csv and results/final_benchmark.json
    results_dir = Path(PROJECT_ROOT) / "results"
    results_dir.mkdir(exist_ok=True)

    csv_path = results_dir / "final_benchmark.csv"
    json_path = results_dir / "final_benchmark.json"

    # Write CSV
    if raw_results:
        # Flatten receiver_utilization dict for CSV
        csv_records = []
        for r in raw_results:
            row = dict(r)
            row["receiver_utilization"] = json.dumps(row.get("receiver_utilization", {}))
            csv_records.append(row)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_records[0].keys()))
            writer.writeheader()
            writer.writerows(csv_records)

    # Write JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "seeds": seeds,
                "total_time": total_time,
                "num_scenarios": len(scenarios),
                "algorithms": ["Sequential", "Random", "AdaptiveBelief", "UCB1", "RL_Experimental", "CognitiveAdaptive_Integrated"]
            },
            "summary": summary_results,
            "raw": raw_results
        }, f, indent=2)

    print(f"\n[OK] Benchmark results successfully written to:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    return summary_results


if __name__ == '__main__':
    run_comprehensive_benchmark()
