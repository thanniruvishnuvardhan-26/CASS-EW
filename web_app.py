from flask import Flask, jsonify, request, render_template
import threading
import json
import os

from config import get_default_config
from simulator.environment import Emitter
from simulator.multi_receiver import SpatialRFEnvironment, MultiReceiverSystem
from algorithms.phase8_spatial_scheduler import SpatialScheduler
from evaluation.intercept_time import InterceptTimeTracker

app = Flask(__name__, static_folder='frontend/static', template_folder='frontend/templates')

# Global simulation state
sim_state = {
    'env': None,
    'mrs': None,
    'scheduler': None,
    'tracker': None,
    'emitters': [],
    'config': None,
    'total_obs': 0,
    'total_det': 0,
    'total_active': 0,
    'last_obs': None,
    'events': []
}

lock = threading.Lock()

def add_event(msg):
    sim_state['events'].append(f"[{sim_state['env'].time:04d}] {msg}")
    if len(sim_state['events']) > 100:
        sim_state['events'] = sim_state['events'][-100:]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/reset', methods=['POST'])
def reset():
    data = request.json or {}
    seed = int(data.get('seed', 42))
    
    with lock:
        config = get_default_config()
        nb = config.environment.num_bands
        eps = config.phase4.epsilon

        env = SpatialRFEnvironment(num_bands=nb, seed=seed)
        
        scenario_name = data.get('scenario', 'Mixed Environment')
        scenario = []
        if scenario_name == 'Deterministic Hopping':
            scenario = [
                {"name": "E1_Hop", "band": 1, "behavior": "hopping", "hop_bands": [1, 3, 5], "hop_interval": 5, "position": (0, 0)},
                {"name": "E2_Hop", "band": 2, "behavior": "hopping", "hop_bands": [2, 4, 6], "hop_interval": 7, "position": (10, 10)}
            ]
        elif scenario_name == 'Periodic':
            scenario = [
                {"name": "E1_Per", "band": 2, "behavior": "periodic", "period": 5, "duty_cycle": 2, "position": (5, 5)},
                {"name": "E2_Per", "band": 4, "behavior": "periodic", "period": 10, "duty_cycle": 3, "position": (8, 2)}
            ]
        elif scenario_name == 'Stationary / Fixed':
            scenario = [
                {"name": "E1_Fix", "band": 0, "behavior": "persistent", "position": (2, 8)},
                {"name": "E2_Fix", "band": 9, "behavior": "persistent", "position": (12, 12)}
            ]
        elif scenario_name == 'Frequency Agile':
            scenario = [
                {"name": "E1_FA", "band": 3, "behavior": "hopping", "hop_bands": [0, 3, 6, 9], "hop_interval": 3, "position": (5, 5)},
                {"name": "E2_FA", "band": 7, "behavior": "hopping", "hop_bands": [1, 2, 7, 8], "hop_interval": 4, "position": (10, 10)}
            ]
        elif scenario_name == 'Random Hopping':
            scenario = [
                {"name": "E1_Rand", "behavior": "random_hopping", "hop_bands": [0, 2, 4, 6, 8], "activity_probability": 0.5, "position": (3, 3)},
                {"name": "E2_Rand", "behavior": "random_hopping", "hop_bands": [1, 3, 5, 7, 9], "activity_probability": 0.5, "position": (7, 7)}
            ]
        else: # Mixed Environment (Default)
            scenario = [
                {"name": "E1_NearR1", "band": 1, "behavior": "periodic", "period": 7, "duty_cycle": 2, "position": (0, 0)},
                {"name": "E2_NearR2", "band": 3, "behavior": "periodic", "period": 11, "duty_cycle": 2, "position": (10, 0)},
                {"name": "E3_NearR3", "band": 5, "behavior": "intermittent", "activity_probability": 0.2, "position": (0, 10)},
            ]

        emitters = []
        for e_spec in scenario:
            kwargs = {k: v for k, v in e_spec.items() if k not in ['name', 'band', 'behavior', 'position']}
            e = Emitter(name=e_spec['name'], band=e_spec.get('band', 0), behavior=e_spec['behavior'], seed=seed, **kwargs)
            env.add_emitter(e, position=e_spec.get('position', (0,0)))
            emitters.append(e)

        rcfg = [
            {'id': 'R1', 'position': (0.0, 0.0), 'snr_threshold': 20.0},
            {'id': 'R2', 'position': (10.0, 0.0), 'snr_threshold': 20.0},
            {'id': 'R3', 'position': (0.0, 10.0), 'snr_threshold': 20.0},
        ]
        mrs = MultiReceiverSystem(env, rcfg)
        r_ids = [r['id'] for r in rcfg]
        
        algorithm_name = data.get('algorithm', 'Phase 8 Spatial Integrated')
        if algorithm_name in ('Phase 8 Spatial Integrated', 'Cognitive Adaptive (Integrated)'):
            scheduler = SpatialScheduler(
                receiver_ids=r_ids, num_bands=nb,
                spatial_weight=0.3, seed=seed, epsilon=eps,
                belief_weight=0.2, temporal_weight=0.15,
                prediction_weight=0.2, pattern_weight=0.15,
                uncertainty_weight=0.05, staleness_weight=0.1,
                hop_weight=0.15, max_revisit_interval=50
            )
        else:
            from algorithms.sequential import SequentialScheduler
            from algorithms.random_scheduler import RandomScheduler
            from algorithms.adaptive_belief import AdaptiveBeliefScheduler
            from algorithms.temporal_belief import TemporalBeliefScheduler
            from algorithms.phase5_temporal_profile import MultiBandTemporalProfileScheduler
            from algorithms.phase6_predictive_scheduler import PredictiveScheduler
            from algorithms.phase7_signal_pattern import PatternAwareScheduler
            from algorithms.ucb_scheduler import UCB1Scheduler
            
            alg_map = {
                'Sequential': SequentialScheduler,
                'Random': RandomScheduler,
                'Adaptive Belief': AdaptiveBeliefScheduler,
                'UCB1': UCB1Scheduler,
                'Temporal Belief': TemporalBeliefScheduler,
                'Phase 5 Temporal': MultiBandTemporalProfileScheduler,
                'Phase 6 Predictive': PredictiveScheduler,
                'Phase 7 Pattern-Aware': PatternAwareScheduler
            }
            
            class MultiReceiverAdapter:
                def __init__(self, base_class, receiver_ids, num_bands, seed=None):
                    self.receiver_ids = receiver_ids
                    self.schedulers = {}
                    for rid in receiver_ids:
                        try:
                            self.schedulers[rid] = base_class(num_bands=num_bands, seed=seed)
                        except TypeError:
                            self.schedulers[rid] = base_class(num_bands=num_bands)
                    self.current_rid_idx = 0
                def reset(self, seed=None):
                    for rid in self.receiver_ids:
                        try:
                            self.schedulers[rid].reset(seed)
                        except TypeError:
                            try:
                                self.schedulers[rid].reset()
                            except:
                                pass
                    self.current_rid_idx = 0
                def select_action(self, obs):
                    rid = self.receiver_ids[self.current_rid_idx]
                    self.current_rid_idx = (self.current_rid_idx + 1) % len(self.receiver_ids)
                    band = self.schedulers[rid].select_action(obs)
                    return (rid, band)
                def update(self, obs, action, result):
                    rid, band = action
                    self.schedulers[rid].update(obs, band, result)
                def get_state(self):
                    return {}
            
            base_cls = alg_map.get(algorithm_name, PatternAwareScheduler)
            scheduler = MultiReceiverAdapter(base_cls, r_ids, nb, seed=seed)

        scheduler.reset(seed=seed)
        
        sim_state['env'] = env
        sim_state['mrs'] = mrs
        sim_state['scheduler'] = scheduler
        sim_state['tracker'] = InterceptTimeTracker()
        sim_state['emitters'] = emitters
        sim_state['config'] = config
        sim_state['total_obs'] = 0
        sim_state['total_det'] = 0
        sim_state['total_active'] = 0
        sim_state['last_obs'] = None
        sim_state['events'] = []
        sim_state['active_scenario'] = scenario_name
        sim_state['active_algorithm'] = algorithm_name
        
        add_event(f"Simulation reset with seed {seed}, Scenario: {scenario_name}, Alg: {algorithm_name}")

    return jsonify({
        "status": "ok", 
        "seed": seed, 
        "num_bands": nb, 
        "active_scenario": scenario_name,
        "active_algorithm": algorithm_name
    })

@app.route('/api/step', methods=['POST'])
def step():
    data = request.json or {}
    steps = int(data.get('steps', 1))
    
    with lock:
        env = sim_state['env']
        if not env:
            return jsonify({"error": "Not initialized"}), 400
        
        mrs = sim_state['mrs']
        scheduler = sim_state['scheduler']
        tracker = sim_state['tracker']
        emitters = sim_state['emitters']
        
        step_events = []
        last_action = None
        
        for _ in range(steps):
            if env.time >= sim_state['config'].environment.total_time:
                break
                
            action = scheduler.select_action(sim_state['last_obs'])
            rid, band = action
            sim_state['total_obs'] += 1

            detected = mrs.scan(rid, band, dwell_time=1)
            sim_state['last_obs'] = mrs.get_receiver(rid).scan_history[-1]

            gt = env.get_ground_truth()
            was_active = band in gt['active_bands']
            if was_active:
                sim_state['total_active'] += 1
            if detected and was_active:
                sim_state['total_det'] += 1

            tracker.update_emitter_states(env.time, [e for e in emitters])
            tracker.record_scan(env.time, band, detected, was_active)

            scheduler.update(sim_state['last_obs'], action, detected)
            
            if detected:
                add_event(f"{rid} tuned Band {band} — Signal detected (emitter activity observed)")
            else:
                add_event(f"{rid} tuned Band {band} — No signal detected")

            # Check if environment change detector produced events
            if hasattr(scheduler, 'change_detector') and scheduler.change_detector.recent_events:
                latest_ev = scheduler.change_detector.recent_events[-1]
                if latest_ev.time == env.time:
                    add_event(f"Environment change detected: type={latest_ev.change_type}, band={latest_ev.band}, severity={latest_ev.severity}, conf={latest_ev.confidence:.2f}")

            last_action = {"receiver": rid, "band": band, "detected": detected}

        tracker_results = tracker.get_results()
        gt = env.get_ground_truth()
        sched_state = scheduler.get_state()
        
        # Prepare belief components
        beliefs = {}
        for r_id in scheduler.receiver_ids:
            sub = scheduler.schedulers[r_id]
            b_vals = sub.beliefs.tolist() if hasattr(sub, 'beliefs') else None
            t_vals = sub.temporal_scores.tolist() if hasattr(sub, 'temporal_scores') else None
            p_vals = sub.prediction_scores.tolist() if hasattr(sub, 'prediction_scores') else None
            obs_vals = sub.observations.tolist() if hasattr(sub, 'observations') else None
            det_vals = sub.detections.tolist() if hasattr(sub, 'detections') else None
            last_obs_vals = sub.last_observed.tolist() if hasattr(sub, 'last_observed') else None
            
            beliefs[r_id] = {
                "belief": b_vals,
                "temporal": t_vals,
                "prediction": p_vals,
                "observations": obs_vals,
                "detections": det_vals,
                "last_observed": last_obs_vals
            }

        # Knowledge Map representation if supported by scheduler
        rf_km = {}
        if hasattr(scheduler, 'knowledge_map'):
            for r_id in scheduler.receiver_ids:
                rf_km[r_id] = [
                    scheduler.knowledge_map.get_entry(r_id, b).to_dict()
                    for b in range(scheduler.num_bands)
                ]

        decision_explanation = {}
        if hasattr(scheduler, 'explain_last_decision'):
            decision_explanation = scheduler.explain_last_decision()

        response = {
            "time": env.time,
            "action": last_action,
            "ground_truth": {
                "active_bands": gt['active_bands'],
                "emitters": [{"name": e.name, "position": env.emitter_positions[e].tolist(), "band": e.band, "active": e.active} for e in emitters]
            },
            "receivers": [{"id": rid, "position": sim_state['mrs'].proxy_envs[rid].receiver_position.tolist()} for rid in sim_state['mrs'].get_all_receiver_ids()],
            "metrics": {
                "total_obs": sim_state['total_obs'],
                "total_det": sim_state['total_det'],
                "interception_rate": sim_state['total_det'] / max(1, sim_state['total_active']),
                "spatial_influenced": sched_state.get('spatial_influenced_selections', 0),
                "exploration": sched_state.get('exploration_selections', 0),
                "exploitation": sched_state.get('exploitation_selections', 0),
                "anti_starvation": sched_state.get('anti_starvation_selections', 0),
                "max_staleness": sched_state.get('max_staleness', 0)
            },
            "intercept_time": tracker_results,
            "beliefs": beliefs,
            "knowledge_map": rf_km,
            "decision_explanation": decision_explanation,
            "events": sim_state['events'][-20:]
        }
        
    return jsonify(response)

@app.route('/api/tsrd/validate', methods=['POST'])
def tsrd_validate():
    """Validates an uploaded or local dataset file (HDF5, JSON, CSV)."""
    data = request.json or {}
    file_path = data.get('file_path', 'data/tsrd_fixtures/sample_stare.h5')
    try:
        from data.tsrd_adapter import TSRDAdapter
        adapter = TSRDAdapter(file_path)
        report = adapter.get_report()
        return jsonify({
            "status": "success",
            "report": report.to_dict()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


@app.route('/api/tsrd/preview', methods=['POST'])
def tsrd_preview():
    """Returns the first N scheduler-visible records with strict label isolation."""
    data = request.json or {}
    file_path = data.get('file_path', 'data/tsrd_fixtures/sample_stare.h5')
    limit = int(data.get('limit', 10))
    try:
        from data.tsrd_adapter import TSRDAdapter
        adapter = TSRDAdapter(file_path)
        obs_iter = adapter.iter_observations()
        records = []
        for _ in range(limit):
            try:
                obs = next(obs_iter)
                records.append({
                    "index": obs.index,
                    "timestamp_us": obs.timestamp_us,
                    "timestamp_s": obs.timestamp_s,
                    "frequency_mhz": obs.frequency_mhz,
                    "pulse_width_us": obs.pulse_width_us,
                    "angle_of_arrival_deg": obs.angle_of_arrival_deg,
                    "amplitude_dbm": obs.amplitude_dbm,
                    "receiver_id": obs.receiver_id
                })
            except StopIteration:
                break
        return jsonify({
            "status": "success",
            "preview": records,
            "quarantine_guarantee": "Zero emitter labels or ground truth exposed"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400



@app.route('/api/tsrd/benchmark', methods=['POST'])
def tsrd_run_benchmark():
    """Runs a common-trace benchmark over a TSRD file."""
    data = request.json or {}
    file_path = data.get('file_path', 'data/tsrd_fixtures/sample_stare.h5')
    steps = int(data.get('steps', 100))
    seed = int(data.get('seed', 42))
    include_rl = bool(data.get('include_rl', True))
    try:
        from evaluation.tsrd_benchmark import TSRDBenchmarkSuite
        suite = TSRDBenchmarkSuite(dataset_path=file_path, max_steps=steps, seed=seed)
        results = suite.run_all_baselines(trace_seed=seed, include_rl=include_rl)
        serialized = {k: v.to_dict() for k, v in results.items()}
        return jsonify({
            "status": "success",
            "dataset": file_path,
            "scan_mode": suite.report.scan_mode,
            "causal_suitability": suite.report.causal_suitability,
            "results": serialized
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# =====================================================================
# PHASE 6: USER DATASET IMPORT, VALIDATION & REPLAY ENDPOINTS
# =====================================================================

import tempfile
from pathlib import Path
from werkzeug.utils import secure_filename
from data.dataset_adapter import DatasetConfig, BaseDatasetAdapter, load_dataset, CANONICAL_TARGET_UNITS
from data.dataset_runtime import DatasetRunConfig, DatasetRuntimeRunner

UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'cass_ew_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_DATASET_EXTENSIONS = {'.csv', '.json', '.h5', '.hdf5'}

# Memory store for active dataset results for export
active_dataset_results = {}

def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_DATASET_EXTENSIONS

@app.route('/api/dataset/upload', methods=['POST'])
def dataset_upload():
    """Uploads a user dataset file (CSV, JSON, HDF5) safely into controlled temporary storage."""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if not allowed_file(filename):
        return jsonify({
            "status": "error",
            "message": f"Unsupported file type '{ext}'. Allowed extensions: .csv, .json, .h5, .hdf5"
        }), 400

    dest_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(dest_path)

    # Initial inspection
    try:
        inspect_res = inspect_dataset_file(dest_path)
        return jsonify({
            "status": "success",
            "file_path": dest_path,
            "filename": filename,
            "format": inspect_res["format"],
            "raw_keys": inspect_res["raw_keys"],
            "preview_rows": inspect_res["preview_rows"],
            "canonical_target_units": CANONICAL_TARGET_UNITS
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"File uploaded but failed to inspect: {str(e)}"}), 400


def inspect_dataset_file(file_path: str):
    """Safely reads column headers / keys and sample rows for field mapping UI."""
    path = Path(file_path)
    ext = path.suffix.lower()
    raw_keys = []
    preview_rows = []

    if ext == ".csv":
        import csv
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            raw_keys = [k.strip() for k in (reader.fieldnames or [])]
            for i, r in enumerate(reader):
                if i >= 5:
                    break
                preview_rows.append(r)
        return {"format": "csv", "raw_keys": raw_keys, "preview_rows": preview_rows}

    elif ext == ".json":
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            raw_keys = list(data[0].keys())
            preview_rows = data[:5]
        return {"format": "json", "raw_keys": raw_keys, "preview_rows": preview_rows}

    elif ext in [".h5", ".hdf5"]:
        import h5py
        with h5py.File(file_path, 'r') as f:
            raw_keys = list(f.keys())
            if 'data' in f:
                d = f['data'][:5]
                preview_rows = d.tolist() if hasattr(d, 'tolist') else []
        return {"format": "hdf5", "raw_keys": raw_keys, "preview_rows": preview_rows}
    else:
        raise ValueError(f"Unsupported format {ext}")


@app.route('/api/dataset/inspect', methods=['POST'])
def dataset_inspect():
    """Inspects an existing dataset path on server."""
    data = request.json or {}
    file_path = data.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"status": "error", "message": f"Dataset file not found: {file_path}"}), 404
    try:
        res = inspect_dataset_file(file_path)
        return jsonify({
            "status": "success",
            "file_path": file_path,
            "filename": os.path.basename(file_path),
            "format": res["format"],
            "raw_keys": res["raw_keys"],
            "preview_rows": res["preview_rows"],
            "canonical_target_units": CANONICAL_TARGET_UNITS
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/dataset/validate', methods=['POST'])
def dataset_validate():
    """Applies user field mappings and explicit units, producing a DatasetValidationReport."""
    data = request.json or {}
    file_path = data.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"status": "error", "message": f"Dataset file not found: {file_path}"}), 404

    field_mapping = data.get('field_mapping', {})
    units = data.get('units', {})
    # Filter empty mappings
    clean_mapping = {k: v for k, v in field_mapping.items() if v and str(v).strip()}
    clean_units = {k: v for k, v in units.items() if v and str(v).strip()}

    cov_range = None
    if 'freq_min' in data and 'freq_max' in data and data['freq_min'] is not None and data['freq_max'] is not None:
        cov_range = (float(data['freq_min']), float(data['freq_max']))

    ds_config = DatasetConfig(
        field_mapping=clean_mapping,
        units=clean_units,
        coverage_freq_range_mhz=cov_range
    )

    try:
        adapter: BaseDatasetAdapter = load_dataset(file_path=file_path, config=ds_config)
        report = adapter.get_report()
        return jsonify({
            "status": "success",
            "report": report.to_dict()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/dataset/run', methods=['POST'])
def dataset_run():
    """Executes the Phase 5 DatasetRuntimeRunner using the user dataset and configuration."""
    data = request.json or {}
    file_path = data.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return jsonify({"status": "error", "message": f"Dataset file not found: {file_path}"}), 404

    field_mapping = data.get('field_mapping', {})
    units = data.get('units', {})
    clean_mapping = {k: v for k, v in field_mapping.items() if v and str(v).strip()}
    clean_units = {k: v for k, v in units.items() if v and str(v).strip()}

    steps = int(data.get('steps', 100))
    dwell_time_s = float(data.get('dwell_time_s', 0.005))
    seed = int(data.get('seed', 42))
    initial_band = int(data.get('initial_band', 0))

    cov_range = None
    if 'freq_min' in data and 'freq_max' in data and data['freq_min'] is not None and data['freq_max'] is not None:
        cov_range = (float(data['freq_min']), float(data['freq_max']))

    ds_config = DatasetConfig(
        field_mapping=clean_mapping,
        units=clean_units,
        coverage_freq_range_mhz=cov_range
    )

    run_config = DatasetRunConfig(
        dataset_path=file_path,
        dataset_config=ds_config,
        seed=seed,
        steps=steps,
        dwell_time_s=dwell_time_s,
        initial_band=initial_band,
        coverage_freq_range_mhz=cov_range
    )

    try:
        runner = DatasetRuntimeRunner(run_config)
        result = runner.run()
        res_dict = result.to_dict()
        active_dataset_results['latest'] = res_dict

        return jsonify({
            "status": "success",
            "result": res_dict
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/dataset/export', methods=['GET'])
def dataset_export():
    """Exports the latest DatasetRunResult as JSON."""
    if 'latest' not in active_dataset_results:
        return jsonify({"status": "error", "message": "No dataset run result available to export"}), 404
    return jsonify(active_dataset_results['latest'])


if __name__ == '__main__':
    app.run(debug=True, port=5000)


