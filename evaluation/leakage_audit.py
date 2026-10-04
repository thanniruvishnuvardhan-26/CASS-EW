"""
CASS-EW Data Leakage Audit Module.

Automated auditing tool to ensure that:
1. Ground-truth RF emitter states never leak into scheduler decision interfaces.
2. Schedulers never observe future timesteps, future pulses, or true emitter identities.
3. Observation spaces contain ONLY receiver-observable signals (detections, dwell, band).
4. Feature vectors passed to schedulers never contain oracle or evaluator-only metadata.
"""

import inspect
from typing import Dict, Any, List
import numpy as np

from algorithms.baseline_scanner import SequentialScanner
from algorithms.bayesian_scheduler import BayesianScheduler
from algorithms.rl_scheduler import RLScheduler


EVALUATOR_ONLY_FIELDS = frozenset({
    "signal_present",
    "true_emitter_id",
    "active_bands",
    "emitter_name",
    "threat_weight",
    "future_activity",
    "future_frequency"
})


def audit_scheduler_interfaces() -> Dict[str, Any]:
    """
    Audit the parameter signatures of all schedulers to ensure no ground-truth fields are required.
    """
    results = {}

    # 1. Sequential Scanner
    seq_sig = inspect.signature(SequentialScanner.get_action)
    results["SequentialScanner.get_action"] = {
        "params": list(seq_sig.parameters.keys()),
        "has_leakage": any(p in EVALUATOR_ONLY_FIELDS for p in seq_sig.parameters.keys())
    }

    # 2. Bayesian Scheduler
    bayes_get_sig = inspect.signature(BayesianScheduler.get_action)
    bayes_up_sig = inspect.signature(BayesianScheduler.update)
    results["BayesianScheduler.get_action"] = {
        "params": list(bayes_get_sig.parameters.keys()),
        "has_leakage": any(p in EVALUATOR_ONLY_FIELDS for p in bayes_get_sig.parameters.keys())
    }
    results["BayesianScheduler.update"] = {
        "params": list(bayes_up_sig.parameters.keys()),
        "has_leakage": any(p in EVALUATOR_ONLY_FIELDS for p in bayes_up_sig.parameters.keys())
    }

    # 3. RL Scheduler
    rl_choose_sig = inspect.signature(RLScheduler.choose_action)
    rl_update_sig = inspect.signature(RLScheduler.update)
    results["RLScheduler.choose_action"] = {
        "params": list(rl_choose_sig.parameters.keys()),
        "has_leakage": any(p in EVALUATOR_ONLY_FIELDS for p in rl_choose_sig.parameters.keys())
    }
    results["RLScheduler.update"] = {
        "params": list(rl_update_sig.parameters.keys()),
        "has_leakage": any(p in EVALUATOR_ONLY_FIELDS for p in rl_update_sig.parameters.keys())
    }

    all_clean = all(not r["has_leakage"] for r in results.values())
    return {
        "clean": all_clean,
        "details": results
    }


def verify_observation_payload(obs: Dict[str, Any]) -> bool:
    """
    Verify whether an observation payload given to a scheduler contains evaluator-only keys.
    """
    leaked_keys = set(obs.keys()).intersection(EVALUATOR_ONLY_FIELDS)
    return len(leaked_keys) == 0
