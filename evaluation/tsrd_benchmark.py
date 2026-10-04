"""
TSRD Common-Trace Benchmark & Evaluator
CASS-EW SIH Problem Statement 26055

Executes rigorous common-trace evaluation of CASS-EW schedulers on TSRD datasets.

CRITICAL EXPERIMENT DESIGN:
1. COMMON-TRACE:
   All schedulers face the EXACT SAME underlying RF environment, same dataset file,
   same seed, same receiver parameters, and same evaluation duration.
2. CORRECT METRIC DEFINITIONS:
   - True Opportunity: Signal was physically present in the scanned band during dwell.
   - Hit: Opportunity existed AND receiver reported detection.
   - Miss: Opportunity existed AND receiver failed to detect.
   - False Alarm: No signal opportunity existed AND receiver reported detection.
   - Interception Rate (P_d empirical): Hits / Opportunities
   - Miss Rate: Misses / Opportunities
   - False Alarm Rate (P_fa empirical): False Alarms / (Total Scans - Opportunities)
   - Scan Efficiency: Hits / Total Dwells
   - Intercept Times: Measured time to first intercept each active emitter.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

from data.tsrd_adapter import TSRDAdapter
from data.band_map import BandMap
from simulator.dataset_rf_environment import DatasetRFEnvironment, DatasetVirtualReceiver
from data.tsrd_scheduler_bridge import TSRDSchedulerBridge

# Schedulers
from algorithms.phase8_spatial_scheduler import SpatialScheduler
from algorithms.sequential import SequentialScheduler
from algorithms.random_scheduler import RandomScheduler
from algorithms.ucb_scheduler import UCB1Scheduler
from algorithms.rl_benchmark_adapter import RLBenchmarkAdapter
from algorithms.rl_scheduler import RLScheduler


@dataclass
class SchedulerBenchmarkResult:
    scheduler_name: str
    total_scans: int
    total_dwell_s: float
    total_simulated_time_s: float
    # Observation-level metrics
    observation_opportunities: int  # Scans where signal was physically present in chosen band
    observation_detections: int     # Scans where receiver reported detection
    hits: int                       # True positives (signal present & detected)
    misses: int                     # False negatives (signal present & missed)
    false_alarms: int               # False positives (signal absent & detected)
    quiet_scans: int                # Scans where signal was absent in chosen band
    receiver_pd: float              # Receiver Pd = hits / observation_opportunities
    receiver_pfa: float             # Receiver Pfa = false_alarms / quiet_scans
    scan_efficiency: float          # hits / total_scans
    # Ground-truth global event metrics
    total_ground_truth_emitters: int
    intercepted_emitters: int
    global_emitter_interception_rate: float # intercepted_emitters / total_ground_truth_emitters
    # Latency metrics (first detection time - ground truth onset time)
    first_intercept_delays: Dict[str, float]
    mean_intercept_time_s: float
    median_intercept_time_s: float
    p90_intercept_time_s: float
    # Diagnostics
    per_band_scans: List[int]
    per_band_hits: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TSRDBenchmarkSuite:
    """
    Common-trace benchmarking suite for TSRD datasets.
    """

    def __init__(
        self,
        dataset_path: str,
        num_bands: int = 10,
        receiver_id: str = "RX_0",
        base_dwell_s: float = 0.005,  # 5ms dwell
        max_steps: int = 200,
        seed: int = 42
    ):
        self.dataset_path = dataset_path
        self.num_bands = num_bands
        self.receiver_id = receiver_id
        self.base_dwell_s = base_dwell_s
        self.max_steps = max_steps
        self.seed = seed

        self.adapter = TSRDAdapter(dataset_path)
        self.report = self.adapter.get_report()
        self.band_map = BandMap.from_tsrd_receiver(self.report.receiver_metadata, default_num_bands=num_bands)

        # Pre-compute ground-truth emitter onset times
        self.emitter_onsets: Dict[str, float] = {}
        obs_list = list(self.adapter.iter_observations())
        gt_list = list(self.adapter.iter_ground_truth())
        for o, g in zip(obs_list, gt_list):
            e_str = f"Emitter_{g.emitter_label}"
            if e_str not in self.emitter_onsets:
                self.emitter_onsets[e_str] = o.timestamp_s

    def run_scheduler(self, scheduler_name: str, scheduler_instance: Any, trace_seed: int = 42) -> SchedulerBenchmarkResult:
        """
        Runs a single scheduler through the common environment trace.
        """
        env = DatasetRFEnvironment(self.adapter, band_map=self.band_map, seed=trace_seed)
        rx = DatasetVirtualReceiver(env, receiver_id=self.receiver_id, seed=trace_seed)
        bridge = TSRDSchedulerBridge(scheduler_instance, self.band_map, base_dwell_s=self.base_dwell_s, receiver_id=self.receiver_id)

        # Reset scheduler if possible
        if hasattr(scheduler_instance, 'reset'):
            try:
                scheduler_instance.reset(seed=trace_seed)
            except TypeError:
                scheduler_instance.reset()

        total_scans = 0
        total_opportunities = 0
        total_detections = 0
        total_hits = 0
        total_misses = 0
        total_false_alarms = 0
        total_dwell_s = 0.0

        per_band_scans = [0] * self.num_bands
        per_band_hits = [0] * self.num_bands

        first_detection_times: Dict[str, float] = {}

        sim_start_time = env.time_s

        for step in range(self.max_steps):
            if env.time_s >= env.max_time_s:
                break

            decision = bridge.get_next_decision()
            band = decision["selected_band"]
            dwell_s = decision["dwell_seconds"]

            obs = rx.execute_dwell(band=band, dwell_s=dwell_s)
            eval_record = rx.get_last_evaluator_record()
            bridge.update_observation(obs, decision)

            total_scans += 1
            total_dwell_s += obs.effective_duration_s
            per_band_scans[band] += 1

            if eval_record.opportunity:
                total_opportunities += 1

            if obs.detected:
                total_detections += 1

            if eval_record.is_hit:
                total_hits += 1
                per_band_hits[band] += 1
                for lbl in eval_record.underlying_emitter_labels:
                    lbl_str = f"Emitter_{lbl}"
                    if lbl_str not in first_detection_times:
                        first_detection_times[lbl_str] = obs.observation_time_s

            if eval_record.is_miss:
                total_misses += 1

            if eval_record.is_false_alarm:
                total_false_alarms += 1

        total_sim_time = env.time_s - sim_start_time
        quiet_scans = total_scans - total_opportunities

        # Metric definitions:
        # Receiver Pd = Hits / Opportunities (empirical detector Pd on tuned band)
        rx_pd = (total_hits / total_opportunities) if total_opportunities > 0 else 0.0
        # Receiver Pfa = False Alarms / Quiet Scans
        rx_pfa = (total_false_alarms / quiet_scans) if quiet_scans > 0 else 0.0
        # Scan Efficiency = Hits / Total Scans
        scan_eff = (total_hits / total_scans) if total_scans > 0 else 0.0

        # Global emitter interception rate = Intercepted Emitters / Total Ground Truth Emitters
        total_gt_emitters = len(self.emitter_onsets)
        intercepted_emitters = len(first_detection_times)
        global_ir = (intercepted_emitters / total_gt_emitters) if total_gt_emitters > 0 else 0.0

        # Latencies = first_detection_time - ground_truth_onset_time
        delays: Dict[str, float] = {}
        for e_str, det_time in first_detection_times.items():
            onset = self.emitter_onsets.get(e_str, sim_start_time)
            delays[e_str] = max(0.0, det_time - onset)

        delay_vals = list(delays.values())
        mean_intercept = float(np.mean(delay_vals)) if delay_vals else float(total_sim_time)
        median_intercept = float(np.median(delay_vals)) if delay_vals else float(total_sim_time)
        p90_intercept = float(np.percentile(delay_vals, 90)) if delay_vals else float(total_sim_time)

        return SchedulerBenchmarkResult(
            scheduler_name=scheduler_name,
            total_scans=total_scans,
            total_dwell_s=total_dwell_s,
            total_simulated_time_s=total_sim_time,
            observation_opportunities=total_opportunities,
            observation_detections=total_detections,
            hits=total_hits,
            misses=total_misses,
            false_alarms=total_false_alarms,
            quiet_scans=quiet_scans,
            receiver_pd=rx_pd,
            receiver_pfa=rx_pfa,
            scan_efficiency=scan_eff,
            total_ground_truth_emitters=total_gt_emitters,
            intercepted_emitters=intercepted_emitters,
            global_emitter_interception_rate=global_ir,
            first_intercept_delays=delays,
            mean_intercept_time_s=mean_intercept,
            median_intercept_time_s=median_intercept,
            p90_intercept_time_s=p90_intercept,
            per_band_scans=per_band_scans,
            per_band_hits=per_band_hits
        )

    def run_all_baselines(self, trace_seed: int = 42, include_rl: bool = True) -> Dict[str, SchedulerBenchmarkResult]:
        """
        Evaluates Sequential, Random, UCB1, SpatialScheduler, and optionally Experimental RL.
        """
        schedulers = {}

        # 1. Sequential Scanner
        seq = SequentialScheduler(num_bands=self.num_bands)
        schedulers["Sequential Scan"] = seq

        # 2. Random Scanner
        rnd = RandomScheduler(num_bands=self.num_bands, seed=trace_seed)
        schedulers["Random Scan"] = rnd

        # 3. UCB1 Scheduler
        ucb = UCB1Scheduler(num_bands=self.num_bands, seed=trace_seed)
        schedulers["UCB1 Baseline"] = ucb

        # 4. Cognitive Adaptive Integrated SpatialScheduler (CURRENT FINAL)
        spatial = SpatialScheduler(
            receiver_ids=[self.receiver_id],
            num_bands=self.num_bands,
            seed=trace_seed,
            epsilon=0.1,
            spatial_weight=0.25,
            hop_weight=0.25,
            uncertainty_weight=0.1,
            staleness_weight=0.1,
            max_revisit_interval=15  # Enforce anti-starvation band visits
        )
        schedulers["CASS-EW Cognitive Adaptive"] = spatial

        # 5. Experimental Q-Learning Baseline
        if include_rl:
            rl_adapter = RLBenchmarkAdapter(num_bands=self.num_bands, seed=trace_seed)
            schedulers["Experimental Q-Learning"] = rl_adapter

        results = {}
        for name, sched in schedulers.items():
            res = self.run_scheduler(name, sched, trace_seed=trace_seed)
            results[name] = res

        return results


if __name__ == "__main__":
    suite = TSRDBenchmarkSuite("data/tsrd_fixtures/sample_stare.h5", max_steps=100, seed=42)
    res_dict = suite.run_all_baselines(trace_seed=42)
    print("=== TSRD COMMON-TRACE BENCHMARK AUDITED RESULTS ===")
    for name, r in res_dict.items():
        print(f"[{name}] Hits: {r.hits}/{r.observation_opportunities} | Rx Pd: {r.receiver_pd:.3f} | Rx Pfa: {r.false_alarms}/{r.quiet_scans} ({r.receiver_pfa:.3f}) | Scan Eff: {r.scan_efficiency:.3f} | Global Emitter Intercept: {r.intercepted_emitters}/{r.total_ground_truth_emitters} ({r.global_emitter_interception_rate:.1%}) | Onset Intercept Latency (Mean): {r.mean_intercept_time_s:.4f}s | Median: {r.median_intercept_time_s:.4f}s | P90: {r.p90_intercept_time_s:.4f}s")
