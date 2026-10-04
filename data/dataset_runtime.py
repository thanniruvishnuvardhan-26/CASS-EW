"""
Dataset Runtime Bridge & Replay Evaluation Runner
CASS-EW SIH Problem Statement 26055 - Phase 5

Connects user datasets (CSV, JSON, HDF5) to the FROZEN-CORE CASS-EW runtime:
Dataset -> Format Detection -> Field Mapping -> Unit Normalization -> Validation
        -> DatasetRFEnvironment -> DatasetVirtualReceiver
        -> CASS-EW Observation Contract -> SpatialScheduler -> Next Band Decision
        -> Causal Advance -> Metrics Collection -> Structured DatasetRunResult

Guarantees:
- Zero frozen core modifications.
- Strict anti-leakage: ground truth is quarantined and never seen by scheduler.
- Zero future pulse access: scheduler only observes past and present dwell intervals.
- Non-modulo deterministic physical band mapping.
- Reproducibility across identical seeds.
- Safe metrics calculation with zero-division protection and explicit null/unavailable handling.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np

from data.band_map import BandMap
from data.time_normalization import TimeNormalizer
from data.dataset_adapter import (
    ObservablePDW,
    GroundTruthPDW,
    DatasetValidationReport,
    BaseDatasetAdapter,
    DatasetConfig,
    load_dataset
)
from simulator.dataset_rf_environment import (
    DatasetRFEnvironment,
    DatasetVirtualReceiver,
    DatasetObservation,
    EvaluatorPulseRecord
)
from data.tsrd_scheduler_bridge import TSRDSchedulerBridge
from algorithms.phase8_spatial_scheduler import SpatialScheduler


# =====================================================================
# DATASET RUN CONFIGURATION
# =====================================================================

@dataclass
class DatasetRunConfig:
    """
    Configuration parameters for a single dataset replay run against CASS-EW.
    """
    dataset_path: str
    dataset_config: Optional[DatasetConfig] = None
    seed: int = 42
    steps: int = 100
    dwell_time_s: float = 0.005  # 5 ms nominal dwell
    initial_band: int = 0
    receiver_id: str = "RX_0"
    num_bands: int = 10
    evaluation_mode: str = "dataset_replay"  # "dataset_replay"
    enable_ground_truth: bool = True
    output_path: Optional[str] = None
    coverage_freq_range_mhz: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.dataset_config is not None:
            d["dataset_config"] = self.dataset_config.to_dict()
        return d


# =====================================================================
# DATASET RUN RESULT
# =====================================================================

@dataclass
class DatasetRunResult:
    """
    Comprehensive structured output of a dataset replay run.
    Contains metadata, validation audit, runtime history, and post-scan evaluation metrics.
    """
    # Dataset metadata
    dataset_path: str
    format: str
    total_records: int
    valid_records: int
    invalid_records: int
    validation_status: str  # "PASS" or "FAILED"
    validation_warnings: List[str]
    validation_errors: List[str]
    resolved_field_mapping: Dict[str, str]
    input_units: Dict[str, str]
    canonical_units: Dict[str, str]
    conversion_summary: Dict[str, str]
    frequency_coverage_mhz: Tuple[float, float]

    # Run metadata
    seed: int
    requested_steps: int
    executed_scans: int
    dwell_time_s: float
    initial_band: int
    simulation_duration_s: float
    evaluation_mode: str

    # Runtime results
    band_decisions: List[int]
    per_band_scan_counts: List[int]
    per_band_hit_counts: List[int]
    total_detections: int

    # Evaluation results (audited metrics)
    observation_opportunities: int
    hits: int
    misses: int
    false_alarms: int
    quiet_scans: int
    receiver_pd: Optional[float]
    receiver_pfa: Optional[float]
    scan_efficiency: Optional[float]
    total_ground_truth_emitters: Optional[int]
    intercepted_emitters: Optional[int]
    global_emitter_interception_rate: Optional[float]
    mean_intercept_time_s: Optional[float]
    median_intercept_time_s: Optional[float]
    p90_intercept_time_s: Optional[float]
    first_intercept_delays: Dict[str, float]

    # Explainability & Diagnostic logs
    warnings: List[str] = field(default_factory=list)
    explanations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text_summary(self) -> str:
        pd_str = f"{self.receiver_pd:.3f}" if self.receiver_pd is not None else "N/A"
        pfa_str = f"{self.receiver_pfa:.4f}" if self.receiver_pfa is not None else "N/A"
        eff_str = f"{self.scan_efficiency:.3f}" if self.scan_efficiency is not None else "N/A"
        ir_str = f"{self.global_emitter_interception_rate * 100:.1f}%" if self.global_emitter_interception_rate is not None else "N/A"
        mean_lat_str = f"{self.mean_intercept_time_s:.4f}s" if self.mean_intercept_time_s is not None else "N/A"

        summary = (
            f"=== CASS-EW DATASET REPLAY RUN RESULT ({self.validation_status}) ===\n"
            f"Dataset: {self.dataset_path} [{self.format.upper()}]\n"
            f"Records: {self.valid_records} valid / {self.total_records} total\n"
            f"Mapping: {self.resolved_field_mapping}\n"
            f"Units: {self.conversion_summary}\n"
            f"Coverage: [{self.frequency_coverage_mhz[0]:.1f}, {self.frequency_coverage_mhz[1]:.1f}] MHz\n"
            f"Runtime: {self.executed_scans} scans across {self.simulation_duration_s:.4f}s (dwell: {self.dwell_time_s}s, seed: {self.seed})\n"
            f"--- EVALUATION METRICS ---\n"
            f"Observation Opportunities: {self.observation_opportunities}\n"
            f"Hits: {self.hits} | Misses: {self.misses} | False Alarms: {self.false_alarms}\n"
            f"Receiver Pd: {pd_str} (Hits / Opportunities)\n"
            f"Receiver Pfa: {pfa_str} (False Alarms / Quiet Scans)\n"
            f"Scan Efficiency: {eff_str} (Hits / Total Scans)\n"
            f"Global Emitter Interception: {self.intercepted_emitters}/{self.total_ground_truth_emitters} ({ir_str})\n"
            f"Mean Intercept Latency: {mean_lat_str}\n"
            f"Warnings: {len(self.warnings) + len(self.validation_warnings)}\n"
        )
        return summary


# =====================================================================
# DATASET RUNTIME RUNNER
# =====================================================================

class DatasetRuntimeRunner:
    """
    Executes an ingested user dataset through the CASS-EW runtime pipeline.
    """

    def __init__(self, config: DatasetRunConfig):
        self.config = config

    def run(self) -> DatasetRunResult:
        """
        Executes the dataset replay pipeline and returns DatasetRunResult.
        """
        # 1. Ingest & Validate Dataset
        adapter: BaseDatasetAdapter = load_dataset(
            file_path=self.config.dataset_path,
            config=self.config.dataset_config,
            coverage_freq_range_mhz=self.config.coverage_freq_range_mhz
        )
        report: DatasetValidationReport = adapter.get_report()

        run_warnings: List[str] = []

        # Handle Invalid Dataset
        if not report.is_valid:
            cov = self.config.coverage_freq_range_mhz or (0.0, 18000.0)
            return DatasetRunResult(
                dataset_path=self.config.dataset_path,
                format=report.format,
                total_records=report.total_rows,
                valid_records=report.valid_rows,
                invalid_records=report.invalid_rows,
                validation_status="FAILED",
                validation_warnings=report.warnings,
                validation_errors=report.validation_errors,
                resolved_field_mapping=report.resolved_field_mapping,
                input_units=report.input_units,
                canonical_units=report.canonical_units,
                conversion_summary=report.conversion_summary,
                frequency_coverage_mhz=cov,
                seed=self.config.seed,
                requested_steps=self.config.steps,
                executed_scans=0,
                dwell_time_s=self.config.dwell_time_s,
                initial_band=self.config.initial_band,
                simulation_duration_s=0.0,
                evaluation_mode=self.config.evaluation_mode,
                band_decisions=[],
                per_band_scan_counts=[0] * self.config.num_bands,
                per_band_hit_counts=[0] * self.config.num_bands,
                total_detections=0,
                observation_opportunities=0,
                hits=0,
                misses=0,
                false_alarms=0,
                quiet_scans=0,
                receiver_pd=None,
                receiver_pfa=None,
                scan_efficiency=None,
                total_ground_truth_emitters=None,
                intercepted_emitters=None,
                global_emitter_interception_rate=None,
                mean_intercept_time_s=None,
                median_intercept_time_s=None,
                p90_intercept_time_s=None,
                first_intercept_delays={},
                warnings=["Dataset validation failed; execution aborted before scheduler invocation."]
            )

        # 2. Check for empty dataset records
        if report.record_count == 0:
            cov = self.config.coverage_freq_range_mhz or (0.0, 18000.0)
            return DatasetRunResult(
                dataset_path=self.config.dataset_path,
                format=report.format,
                total_records=report.total_rows,
                valid_records=0,
                invalid_records=report.total_rows,
                validation_status="FAILED",
                validation_warnings=report.warnings,
                validation_errors=["Dataset contains 0 valid records."],
                resolved_field_mapping=report.resolved_field_mapping,
                input_units=report.input_units,
                canonical_units=report.canonical_units,
                conversion_summary=report.conversion_summary,
                frequency_coverage_mhz=cov,
                seed=self.config.seed,
                requested_steps=self.config.steps,
                executed_scans=0,
                dwell_time_s=self.config.dwell_time_s,
                initial_band=self.config.initial_band,
                simulation_duration_s=0.0,
                evaluation_mode=self.config.evaluation_mode,
                band_decisions=[],
                per_band_scan_counts=[0] * self.config.num_bands,
                per_band_hit_counts=[0] * self.config.num_bands,
                total_detections=0,
                observation_opportunities=0,
                hits=0,
                misses=0,
                false_alarms=0,
                quiet_scans=0,
                receiver_pd=None,
                receiver_pfa=None,
                scan_efficiency=None,
                total_ground_truth_emitters=None,
                intercepted_emitters=None,
                global_emitter_interception_rate=None,
                mean_intercept_time_s=None,
                median_intercept_time_s=None,
                p90_intercept_time_s=None,
                first_intercept_delays={},
                warnings=["Dataset contains 0 valid records; execution skipped."]
            )

        # 3. Build BandMap
        # If user configured explicit coverage, use it; otherwise use dataset receiver metadata or span
        if self.config.coverage_freq_range_mhz is not None:
            min_f, max_f = self.config.coverage_freq_range_mhz
            band_map = BandMap(num_bands=self.config.num_bands, min_freq_mhz=min_f, max_freq_mhz=max_f)
        elif report.receiver_metadata.get("freq_range_mhz") is not None:
            band_map = BandMap.from_tsrd_receiver(report.receiver_metadata, default_num_bands=self.config.num_bands)
        else:
            # Use detected frequency span from validation report with comfortable margins
            min_f = max(0.0, report.frequency_range_mhz[0] - 100.0)
            max_f = report.frequency_range_mhz[1] + 100.0
            if max_f <= min_f:
                max_f = min_f + 1000.0
            band_map = BandMap(num_bands=self.config.num_bands, min_freq_mhz=min_f, max_freq_mhz=max_f)

        # 4. Construct Causal Environment & Virtual Receiver
        env = DatasetRFEnvironment(adapter=adapter, band_map=band_map, seed=self.config.seed)
        rx = DatasetVirtualReceiver(
            env=env,
            receiver_id=self.config.receiver_id,
            seed=self.config.seed
        )

        # 5. Instantiate Production SpatialScheduler (FROZEN CORE)
        scheduler = SpatialScheduler(
            receiver_ids=[self.config.receiver_id],
            num_bands=self.config.num_bands,
            seed=self.config.seed,
            epsilon=0.05  # Production balance of exploitation and anti-starvation
        )

        # 6. Instantiate Scheduler Bridge
        bridge = TSRDSchedulerBridge(
            scheduler=scheduler,
            band_map=band_map,
            base_dwell_s=self.config.dwell_time_s,
            receiver_id=self.config.receiver_id
        )

        # Pre-calculate Ground-Truth Emitter Onsets for Latency Evaluation (Quarantined)
        emitter_onsets: Dict[str, float] = {}
        has_gt_col = ("emitter_label" in report.resolved_field_mapping and report.resolved_field_mapping["emitter_label"] != "") or (report.format == "hdf5" and "labels" in report.resolved_field_mapping)
        has_ground_truth = self.config.enable_ground_truth and has_gt_col and (report.label_count > 0)
        if has_ground_truth:
            obs_stream = list(adapter.iter_observations())
            gt_stream = list(adapter.iter_ground_truth())
            for o, g in zip(obs_stream, gt_stream):
                lbl_key = f"Emitter_{g.emitter_label}"
                if lbl_key not in emitter_onsets:
                    emitter_onsets[lbl_key] = o.timestamp_s

        # 7. Execute Dwell Steps
        sim_start_time = env.time_s
        total_scans = 0
        total_opportunities = 0
        total_detections = 0
        total_hits = 0
        total_misses = 0
        total_false_alarms = 0
        total_dwell_s = 0.0

        band_decisions: List[int] = []
        per_band_scans = [0] * self.config.num_bands
        per_band_hits = [0] * self.config.num_bands
        first_detection_times: Dict[str, float] = {}
        explanations: List[Dict[str, Any]] = []

        for step in range(self.config.steps):
            if env.time_s >= env.max_time_s:
                run_warnings.append(f"Simulation ended early at step {step}: reached end of dataset time horizon ({env.max_time_s:.4f}s).")
                break

            # A. Get decision from scheduler
            decision = bridge.get_next_decision()
            band = decision["selected_band"]
            dwell_s = decision["dwell_seconds"]
            band_decisions.append(band)
            if decision.get("explanation"):
                explanations.append({"step": step, "band": band, "explanation": decision["explanation"]})

            # B. Execute dwell in physical environment
            obs = rx.execute_dwell(band=band, dwell_s=dwell_s)
            eval_record: EvaluatorPulseRecord = rx.get_last_evaluator_record()

            # C. Feed strictly observable PDW features back to scheduler (ZERO LEAKAGE)
            bridge.update_observation(obs, decision)

            # D. Record metrics
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
                if has_ground_truth:
                    for lbl in eval_record.underlying_emitter_labels:
                        lbl_key = f"Emitter_{lbl}"
                        if lbl_key not in first_detection_times:
                            first_detection_times[lbl_key] = obs.observation_time_s

            if eval_record.is_miss:
                total_misses += 1

            if eval_record.is_false_alarm:
                total_false_alarms += 1

        total_sim_time = max(0.0, env.time_s - sim_start_time)
        quiet_scans = max(0, total_scans - total_opportunities)

        # 8. Compute Audited Evaluation Metrics
        rx_pd = (total_hits / total_opportunities) if total_opportunities > 0 else (None if total_scans == 0 else 0.0)
        rx_pfa = (total_false_alarms / quiet_scans) if quiet_scans > 0 else (None if total_scans == 0 else 0.0)
        scan_eff = (total_hits / total_scans) if total_scans > 0 else None

        if has_ground_truth and len(emitter_onsets) > 0:
            total_gt_emitters = len(emitter_onsets)
            intercepted_emitters = len(first_detection_times)
            global_ir = float(intercepted_emitters / total_gt_emitters)

            delays: Dict[str, float] = {}
            for e_key, det_time in first_detection_times.items():
                onset = emitter_onsets.get(e_key, sim_start_time)
                delays[e_key] = max(0.0, det_time - onset)

            delay_vals = list(delays.values())
            mean_lat = float(np.mean(delay_vals)) if delay_vals else None
            median_lat = float(np.median(delay_vals)) if delay_vals else None
            p90_lat = float(np.percentile(delay_vals, 90)) if delay_vals else None
        else:
            total_gt_emitters = None
            intercepted_emitters = None
            global_ir = None
            delays = {}
            mean_lat = None
            median_lat = None
            p90_lat = None
            run_warnings.append("Ground truth evaluation unavailable (no emitter labels present in dataset).")

        result = DatasetRunResult(
            dataset_path=self.config.dataset_path,
            format=report.format,
            total_records=report.total_rows,
            valid_records=report.valid_rows,
            invalid_records=report.invalid_rows,
            validation_status="PASS",
            validation_warnings=report.warnings,
            validation_errors=[],
            resolved_field_mapping=report.resolved_field_mapping,
            input_units=report.input_units,
            canonical_units=report.canonical_units,
            conversion_summary=report.conversion_summary,
            frequency_coverage_mhz=(band_map.boundaries[0][0], band_map.boundaries[-1][1]),
            seed=self.config.seed,
            requested_steps=self.config.steps,
            executed_scans=total_scans,
            dwell_time_s=self.config.dwell_time_s,
            initial_band=self.config.initial_band,
            simulation_duration_s=total_sim_time,
            evaluation_mode=self.config.evaluation_mode,
            band_decisions=band_decisions,
            per_band_scan_counts=per_band_scans,
            per_band_hit_counts=per_band_hits,
            total_detections=total_detections,
            observation_opportunities=total_opportunities,
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
            mean_intercept_time_s=mean_lat,
            median_intercept_time_s=median_lat,
            p90_intercept_time_s=p90_lat,
            first_intercept_delays=delays,
            warnings=run_warnings,
            explanations=explanations[:20]  # Store first 20 decisions for lightweight reporting
        )

        # 9. Output to file if configured
        if self.config.output_path is not None:
            out_p = Path(self.config.output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, 'w', encoding='utf-8') as f:
                f.write(result.to_json(indent=2))

        return result


def run_dataset(
    dataset_path: str,
    dataset_config: Optional[DatasetConfig] = None,
    seed: int = 42,
    steps: int = 100,
    dwell_time_s: float = 0.005,
    initial_band: int = 0,
    output_path: Optional[str] = None,
    coverage_freq_range_mhz: Optional[Tuple[float, float]] = None
) -> DatasetRunResult:
    """Convenience functional wrapper around DatasetRuntimeRunner."""
    cfg = DatasetRunConfig(
        dataset_path=dataset_path,
        dataset_config=dataset_config,
        seed=seed,
        steps=steps,
        dwell_time_s=dwell_time_s,
        initial_band=initial_band,
        output_path=output_path,
        coverage_freq_range_mhz=coverage_freq_range_mhz
    )
    runner = DatasetRuntimeRunner(cfg)
    return runner.run()
