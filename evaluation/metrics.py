"""
CASS-EW Unified Evaluation Metrics Module.

Provides mathematically rigorous, standardized metrics for Electronic Warfare
receiver scheduling evaluation:
1. Spectrum Interception Rate (Operational metric across all emitter opportunities)
2. Spectrum Miss Rate (Complement of Interception Rate)
3. False Alarm Rate (Conditional false alarm probability on inactive bands)
4. Scan Efficiency (Interceptions per unit dwell time)
5. Receiver-conditional Pd and Pfa (Detector hardware fidelity)
6. Event-based Time-To-Intercept (TTI) across distinct emitter activation windows
"""

from typing import List, Dict, Set, Tuple, Optional, Any
import numpy as np


def compute_benchmark_metrics(
    trace: List[Set[int]],
    observations: List[Dict[str, Any]],
    total_dwell: int
) -> Dict[str, float]:
    """
    Compute operational EW benchmark metrics matching CASS-EW baseline.

    Parameters:
        trace: List of sets of active band indices per time step.
        observations: List of observation dicts containing:
            - 'signal': bool (ground truth signal present on scanned band)
            - 'detected': bool (receiver detection flag)
        total_dwell: Total dwell units spent across the evaluation.

    Returns:
        Dict of standardized benchmark metrics.
    """
    total_opportunities = sum(len(active_bands) for active_bands in trace)
    intercepted = 0
    scanned_active = 0
    false_alarms = 0
    scanned_inactive = 0

    for obs in observations:
        signal = obs["signal"]
        detected = obs["detected"]

        if signal:
            scanned_active += 1
            if detected:
                intercepted += 1
        else:
            scanned_inactive += 1
            if detected:
                false_alarms += 1

    missed = total_opportunities - intercepted

    interception_rate = (
        (intercepted / total_opportunities * 100.0)
        if total_opportunities > 0 else 0.0
    )
    miss_rate = (
        (missed / total_opportunities * 100.0)
        if total_opportunities > 0 else 0.0
    )
    false_alarm_rate = (
        (false_alarms / scanned_inactive * 100.0)
        if scanned_inactive > 0 else 0.0
    )
    efficiency = (
        (intercepted / total_dwell)
        if total_dwell > 0 else 0.0
    )

    # Receiver hardware conditional Pd: P(detect | scanned and signal present)
    receiver_pd = (
        (intercepted / scanned_active * 100.0)
        if scanned_active > 0 else 0.0
    )

    return {
        "opportunities": total_opportunities,
        "intercepted": intercepted,
        "missed": missed,
        "false_alarms": false_alarms,
        "scanned_active": scanned_active,
        "scanned_inactive": scanned_inactive,
        "interception_rate": interception_rate,
        "miss_rate": miss_rate,
        "false_alarm_rate": false_alarm_rate,
        "efficiency": efficiency,
        "receiver_pd": receiver_pd,
    }


def compute_event_based_tti(
    emitter_activation_events: List[Dict[str, Any]],
    scan_log: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute rigorous event-based Time-To-Intercept (TTI) for discrete emitter activations.

    Parameters:
        emitter_activation_events: List of distinct activation cycles:
            - 'emitter_id': str or int
            - 'band': int
            - 't_start': int (activation onset time)
            - 't_end': int (activation cessation time, inclusive)
        scan_log: List of scan records:
            - 'time': int
            - 'band': int
            - 'detected': bool

    Returns:
        Dict with mean, median, p90, p95 intercept times, detected/missed counts.
    """
    intercept_delays = []
    missed_events = 0
    detected_events = 0

    # Build fast lookup map for detections: (time, band) -> detected
    detection_lookup = {
        (rec["time"], rec["band"]): rec["detected"]
        for rec in scan_log
    }

    for event in emitter_activation_events:
        t_start = event["t_start"]
        t_end = event["t_end"]
        band = event["band"]

        detected_at = None
        for t in range(t_start, t_end + 1):
            if detection_lookup.get((t, band), False):
                detected_at = t
                break

        if detected_at is not None:
            intercept_delays.append(detected_at - t_start)
            detected_events += 1
        else:
            missed_events += 1

    if intercept_delays:
        delays_arr = np.array(intercept_delays, dtype=float)
        mean_tti = float(np.mean(delays_arr))
        median_tti = float(np.median(delays_arr))
        p90_tti = float(np.percentile(delays_arr, 90))
        p95_tti = float(np.percentile(delays_arr, 95))
    else:
        mean_tti = float("nan")
        median_tti = float("nan")
        p90_tti = float("nan")
        p95_tti = float("nan")

    return {
        "mean_tti": mean_tti,
        "median_tti": median_tti,
        "p90_tti": p90_tti,
        "p95_tti": p95_tti,
        "detected_events": detected_events,
        "missed_events": missed_events,
        "total_events": len(emitter_activation_events),
        "raw_delays": intercept_delays,
    }
