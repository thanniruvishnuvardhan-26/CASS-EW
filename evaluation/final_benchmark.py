import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.baseline_scanner import SequentialScanner
from algorithms.bayesian_scheduler import BayesianScheduler
from algorithms.rl_scheduler import (
    RLScheduler,
    calculate_reward
)


NUM_BANDS = 10
TOTAL_TIME = 500


# ============================================================
# COMMON RF + DETECTOR TRACE
# ============================================================

def generate_trace(seed, total_time=TOTAL_TIME):

    rng = np.random.default_rng(seed)

    trace = []

    hopping_sequence = [6, 7, 8, 9]
    hop_index = 0
    periodic_active = False

    for _ in range(total_time):

        active_bands = set()

        # Intermittent emitter
        if rng.random() < 0.7:
            active_bands.add(2)

        # Periodic emitter
        periodic_active = not periodic_active

        if periodic_active:
            active_bands.add(5)

        # Frequency-hopping emitter
        if rng.random() < 0.8:

            active_bands.add(
                hopping_sequence[hop_index]
            )

            hop_index = (
                hop_index + 1
            ) % len(hopping_sequence)

        trace.append(active_bands)

    return trace


def generate_detector_trace(seed):

    rng = np.random.default_rng(seed)

    detection_random = rng.random(
        (TOTAL_TIME, NUM_BANDS)
    )

    false_alarm_random = rng.random(
        (TOTAL_TIME, NUM_BANDS)
    )

    return detection_random, false_alarm_random


# ============================================================
# SCAN ONE TIME POINT
# ============================================================

def observe(
    trace,
    detection_random,
    false_alarm_random,
    time_index,
    band
):

    signal_present = (
        band in trace[time_index]
    )

    if signal_present:

        detected = (
            detection_random[
                time_index,
                band
            ] < 0.90
        )

    else:

        detected = (
            false_alarm_random[
                time_index,
                band
            ] < 0.05
        )

    return signal_present, detected


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    trace,
    observations,
    total_dwell
):

    total_opportunities = 0
    intercepted = 0
    scanned_active = 0
    false_alarms = 0
    scanned_inactive = 0

    # Every active band at every time is an opportunity.
    for active_bands in trace:

        total_opportunities += len(
            active_bands
        )

    for observation in observations:

        signal = observation["signal"]
        detected = observation["detected"]

        if signal:

            scanned_active += 1

            if detected:
                intercepted += 1

        else:

            scanned_inactive += 1

            if detected:
                false_alarms += 1

    missed = (
        total_opportunities
        - intercepted
    )

    interception_rate = (
        intercepted
        / total_opportunities
        * 100
        if total_opportunities > 0
        else 0
    )

    miss_rate = (
        missed
        / total_opportunities
        * 100
        if total_opportunities > 0
        else 0
    )

    false_alarm_rate = (
        false_alarms
        / scanned_inactive
        * 100
        if scanned_inactive > 0
        else 0
    )

    efficiency = (
        intercepted
        / total_dwell
        if total_dwell > 0
        else 0
    )

    return {
        "opportunities": total_opportunities,
        "intercepted": intercepted,
        "missed": missed,
        "false_alarms": false_alarms,
        "interception_rate": interception_rate,
        "miss_rate": miss_rate,
        "false_alarm_rate": false_alarm_rate,
        "efficiency": efficiency
    }


# ============================================================
# SEQUENTIAL
# ============================================================

def evaluate_sequential(
    trace,
    detection_random,
    false_alarm_random
):

    scanner = SequentialScanner(
        NUM_BANDS
    )

    observations = []

    time_index = 0

    while time_index < len(trace):

        band = scanner.get_action()

        signal, detected = observe(
            trace,
            detection_random,
            false_alarm_random,
            time_index,
            band
        )

        observations.append({
            "time": time_index,
            "band": band,
            "signal": signal,
            "detected": detected
        })

        time_index += 1

    return calculate_metrics(
        trace,
        observations,
        len(observations)
    )


# ============================================================
# BAYESIAN
# ============================================================

def evaluate_bayesian(
    trace,
    detection_random,
    false_alarm_random
):

    scheduler = BayesianScheduler(
        NUM_BANDS
    )

    observations = []

    time_index = 0

    while time_index < len(trace):

        band, dwell = (
            scheduler.get_action()
        )

        last_detected = False

        for _ in range(dwell):

            if time_index >= len(trace):
                break

            signal, detected = observe(
                trace,
                detection_random,
                false_alarm_random,
                time_index,
                band
            )

            observations.append({
                "time": time_index,
                "band": band,
                "signal": signal,
                "detected": detected
            })

            last_detected = detected

            time_index += 1

        scheduler.update(
            band,
            last_detected
        )

    return calculate_metrics(
        trace,
        observations,
        len(observations)
    )


# ============================================================
# RL TRAINING
# ============================================================

def train_rl(training_seed=None):

    scheduler = RLScheduler(
        NUM_BANDS,
        (1, 2, 3),
        0.1,
        0.9,
        0.3,
        seed=training_seed
    )

    for episode in range(300):

        t_offset = (training_seed if training_seed is not None else 0) * 1000

        trace = generate_trace(
            t_offset + 10000 + episode,
            300
        )

        detection_random, false_alarm_random = (
            generate_detector_trace(
                t_offset + 20000 + episode
            )
        )

        belief = np.ones(
            NUM_BANDS
        ) * 0.1

        time_index = 0

        while time_index < len(trace):

            # Current state
            state = scheduler.get_state(
                belief
            )

            # Choose action
            action = scheduler.choose_action(
                state,
                training=True
            )

            # Convert action
            band, dwell = (
                scheduler.action_to_parameters(
                    action
                )
            )

            # Store starting time of this dwell
            dwell_start = time_index

            detections = 0

            # ------------------------------------------------
            # Execute dwell
            # ------------------------------------------------

            for _ in range(dwell):

                if time_index >= len(trace):
                    break

                signal, detected = observe(
                    trace,
                    detection_random,
                    false_alarm_random,
                    time_index,
                    band
                )

                if detected:
                    detections += 1

                time_index += 1

            # ------------------------------------------------
            # Calculate improved reward
            # ------------------------------------------------

            reward = 0.0

            dwell_end = time_index

            for t in range(
                dwell_start,
                dwell_end
            ):

                signal, detected = observe(
                    trace,
                    detection_random,
                    false_alarm_random,
                    t,
                    band
                )

                false_alarm = (
                    not signal
                    and detected
                )

                reward += calculate_reward(
                    detected=detected,
                    signal_present=signal,
                    dwell_time=1,
                    false_alarm=false_alarm
                )

            # ------------------------------------------------
            # Belief update
            # ------------------------------------------------

            if detections > 0:

                belief[band] = min(
                    0.99,
                    belief[band] + 0.3
                )

            else:

                belief[band] = max(
                    0.01,
                    belief[band] - 0.1
                )

            # Slight decay for unobserved bands
            for b in range(NUM_BANDS):

                if b != band:

                    belief[b] *= 0.99

            # Next state
            next_state = (
                scheduler.get_state(
                    belief
                )
            )

            # Q-learning update
            scheduler.update(
                state,
                action,
                reward,
                next_state
            )

        # Reduce exploration
        scheduler.decay_epsilon()

    # Evaluation must be greedy
    scheduler.epsilon = 0.0

    return scheduler


# ============================================================
# RL EVALUATION
# ============================================================

def evaluate_rl(
    scheduler,
    trace,
    detection_random,
    false_alarm_random
):

    belief = np.ones(
        NUM_BANDS
    ) * 0.1

    observations = []

    time_index = 0

    while time_index < len(trace):

        # Current state
        state = scheduler.get_state(
            belief
        )

        # Greedy action
        action = scheduler.choose_action(
            state,
            training=False
        )

        # Convert action
        band, dwell = (
            scheduler.action_to_parameters(
                action
            )
        )

        detections = 0
        last_detected = False

        # ------------------------------------------------
        # Execute dwell
        # ------------------------------------------------

        for _ in range(dwell):

            if time_index >= len(trace):
                break

            signal, detected = observe(
                trace,
                detection_random,
                false_alarm_random,
                time_index,
                band
            )

            observations.append({
                "time": time_index,
                "band": band,
                "signal": signal,
                "detected": detected
            })

            if detected:
                detections += 1

            last_detected = detected

            time_index += 1

        # ------------------------------------------------
        # Belief update
        # ------------------------------------------------

        if detections > 0:

            belief[band] = min(
                0.99,
                belief[band] + 0.3
            )

        else:

            belief[band] = max(
                0.01,
                belief[band] - 0.1
            )

        # Slight decay for unobserved bands
        for b in range(NUM_BANDS):

            if b != band:

                belief[b] *= 0.99

    return calculate_metrics(
        trace,
        observations,
        len(observations)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nCASS-EW FINAL CONTROLLED BENCHMARK"
    )

    print(
        "===================================="
    )

    # Multiple common test traces
    seeds = [
        42,
        43,
        44,
        45,
        46
    ]

    print(
        "\nGenerating common RF traces..."
    )

    traces = []

    detector_traces = []

    for seed in seeds:

        traces.append(
            generate_trace(
                seed
            )
        )

        detector_traces.append(
            generate_detector_trace(
                seed + 500
            )
        )

    # --------------------------------------------------------
    # Train RL
    # --------------------------------------------------------

    print(
        "Training RL scheduler..."
    )

    rl_scheduler = train_rl()

    print(
        "RL training completed."
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    all_results = {
        "Sequential": [],
        "Bayesian": [],
        "RL": []
    }

    for i, trace in enumerate(
        traces
    ):

        detection_random, false_alarm_random = (
            detector_traces[i]
        )

        # Sequential
        all_results["Sequential"].append(
            evaluate_sequential(
                trace,
                detection_random,
                false_alarm_random
            )
        )

        # Bayesian
        all_results["Bayesian"].append(
            evaluate_bayesian(
                trace,
                detection_random,
                false_alarm_random
            )
        )

        # RL
        all_results["RL"].append(
            evaluate_rl(
                rl_scheduler,
                trace,
                detection_random,
                false_alarm_random
            )
        )

    # --------------------------------------------------------
    # Print average results
    # --------------------------------------------------------

    print(
        "\nAlgorithm       Interception   False Alarm   "
        "Miss Rate   Efficiency"
    )

    print(
        "------------------------------------------------------"
    )

    for name, results in all_results.items():

        interception = np.mean([
            r["interception_rate"]
            for r in results
        ])

        false_alarm = np.mean([
            r["false_alarm_rate"]
            for r in results
        ])

        miss = np.mean([
            r["miss_rate"]
            for r in results
        ])

        efficiency = np.mean([
            r["efficiency"]
            for r in results
        ])

        print(
            f"{name:<16}"
            f"{interception:>10.2f}%"
            f"{false_alarm:>13.2f}%"
            f"{miss:>11.2f}%"
            f"{efficiency:>13.4f}"
        )

    print(
        "\nEvaluation completed."
    )


if __name__ == "__main__":

    main()