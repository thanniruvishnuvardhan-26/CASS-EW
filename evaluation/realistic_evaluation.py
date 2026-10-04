import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver

from algorithms.baseline_scanner import SequentialScanner
from algorithms.bayesian_scheduler import BayesianScheduler
from algorithms.rl_scheduler import RLScheduler


NUM_BANDS = 10
STEPS = 200


def create_environment():
    environment = RFEnvironment(NUM_BANDS)

    environment.add_emitter(
        Emitter("Emitter_A", 2, "intermittent", 0.7)
    )

    environment.add_emitter(
        Emitter("Emitter_B", 5, "periodic")
    )

    environment.add_emitter(
        Emitter("Emitter_C", 7, "hopping", 0.8, [6, 7, 8, 9])
    )

    return environment


def update_belief(belief, band, detected):

    if detected:
        belief[band] = min(
            0.99,
            belief[band] + 0.3
        )
    else:
        belief[band] = max(
            0.01,
            belief[band] - 0.1
        )

    for b in range(NUM_BANDS):
        if b != band:
            belief[b] *= 0.99

    return belief


def evaluate_sequential(seed):

    np.random.seed(seed)

    environment = create_environment()

    receiver = VirtualReceiver(
        NUM_BANDS,
        detection_probability=0.90,
        false_alarm_probability=0.05
    )

    scanner = SequentialScanner(NUM_BANDS)

    true_detections = 0
    false_alarms = 0
    misses = 0
    total_scans = 0
    total_dwell = 0

    for _ in range(STEPS):

        band = scanner.get_action()

        receiver.scan(
            environment,
            band,
            1
        )

        result = receiver.get_last_result()

        signal = result["signal_present"]
        detected = result["detected"]

        total_scans += 1
        total_dwell += 1

        if signal and detected:
            true_detections += 1

        elif signal and not detected:
            misses += 1

        elif not signal and detected:
            false_alarms += 1

    return {
        "true_detections": true_detections,
        "false_alarms": false_alarms,
        "misses": misses,
        "scans": total_scans,
        "dwell": total_dwell
    }


def evaluate_bayesian(seed):

    np.random.seed(seed)

    environment = create_environment()

    receiver = VirtualReceiver(
        NUM_BANDS,
        detection_probability=0.90,
        false_alarm_probability=0.05
    )

    scheduler = BayesianScheduler(NUM_BANDS)

    true_detections = 0
    false_alarms = 0
    misses = 0
    total_scans = 0
    total_dwell = 0

    for _ in range(STEPS):

        band, dwell_time = scheduler.get_action()

        receiver.scan(
            environment,
            band,
            dwell_time
        )

        for result in receiver.scan_history[-dwell_time:]:

            signal = result["signal_present"]
            detected = result["detected"]

            total_scans += 1

            if signal and detected:
                true_detections += 1

            elif signal and not detected:
                misses += 1

            elif not signal and detected:
                false_alarms += 1

        total_dwell += dwell_time

        last_result = receiver.get_last_result()

        scheduler.update(
            band,
            last_result["detected"]
        )

    return {
        "true_detections": true_detections,
        "false_alarms": false_alarms,
        "misses": misses,
        "scans": total_scans,
        "dwell": total_dwell
    }


def train_rl():

    np.random.seed(42)

    scheduler = RLScheduler(
        NUM_BANDS,
        (1, 2, 3),
        0.1,
        0.9,
        0.3
    )

    for _ in range(300):

        environment = create_environment()

        receiver = VirtualReceiver(
            NUM_BANDS,
            detection_probability=0.90,
            false_alarm_probability=0.05
        )

        belief = np.ones(NUM_BANDS) * 0.1

        for _ in range(100):

            state = scheduler.get_state(belief)

            action = scheduler.choose_action(
                state,
                training=True
            )

            band, dwell_time = scheduler.action_to_parameters(
                action
            )

            receiver.scan(
                environment,
                band,
                dwell_time
            )

            detected = receiver.get_last_result()["detected"]

            belief = update_belief(
                belief,
                band,
                detected
            )

            reward = 10.0 if detected else 0.0
            reward -= 0.5 * dwell_time

            next_state = scheduler.get_state(belief)

            scheduler.update(
                state,
                action,
                reward,
                next_state
            )

        scheduler.decay_epsilon()

    scheduler.epsilon = 0.0

    return scheduler


def evaluate_rl(scheduler, seed):

    np.random.seed(seed)

    environment = create_environment()

    receiver = VirtualReceiver(
        NUM_BANDS,
        detection_probability=0.90,
        false_alarm_probability=0.05
    )

    belief = np.ones(NUM_BANDS) * 0.1

    true_detections = 0
    false_alarms = 0
    misses = 0
    total_scans = 0
    total_dwell = 0

    for _ in range(STEPS):

        state = scheduler.get_state(belief)

        action = scheduler.choose_action(
            state,
            training=False
        )

        band, dwell_time = scheduler.action_to_parameters(
            action
        )

        receiver.scan(
            environment,
            band,
            dwell_time
        )

        results = receiver.scan_history[-dwell_time:]

        for result in results:

            signal = result["signal_present"]
            detected = result["detected"]

            total_scans += 1

            if signal and detected:
                true_detections += 1

            elif signal and not detected:
                misses += 1

            elif not signal and detected:
                false_alarms += 1

        total_dwell += dwell_time

        detected = receiver.get_last_result()["detected"]

        belief = update_belief(
            belief,
            band,
            detected
        )

    return {
        "true_detections": true_detections,
        "false_alarms": false_alarms,
        "misses": misses,
        "scans": total_scans,
        "dwell": total_dwell
    }


def calculate_metrics(result):

    signal_events = (
        result["true_detections"]
        + result["misses"]
    )

    detection_rate = (
        result["true_detections"]
        / signal_events * 100
        if signal_events > 0 else 0
    )

    false_alarm_rate = (
        result["false_alarms"]
        / result["scans"] * 100
        if result["scans"] > 0 else 0
    )

    scan_efficiency = (
        result["true_detections"]
        / result["dwell"]
        if result["dwell"] > 0 else 0
    )

    return detection_rate, false_alarm_rate, scan_efficiency


def main():

    print("\nCASS-EW Realistic Evaluation")
    print("============================")

    print("\nTraining RL scheduler...")

    rl_scheduler = train_rl()

    print("RL training completed.")

    seeds = [42, 43, 44, 45, 46]

    algorithms = {
        "Sequential": [],
        "Bayesian": [],
        "RL": []
    }

    for seed in seeds:

        algorithms["Sequential"].append(
            evaluate_sequential(seed)
        )

        algorithms["Bayesian"].append(
            evaluate_bayesian(seed)
        )

        algorithms["RL"].append(
            evaluate_rl(
                rl_scheduler,
                seed
            )
        )

    print(
        "\nAlgorithm       Detection    False Alarm    Efficiency"
    )

    print(
        "---------------------------------------------------------"
    )

    for name, results in algorithms.items():

        metrics = [
            calculate_metrics(result)
            for result in results
        ]

        detection = np.mean(
            [m[0] for m in metrics]
        )

        false_alarm = np.mean(
            [m[1] for m in metrics]
        )

        efficiency = np.mean(
            [m[2] for m in metrics]
        )

        print(
            f"{name:<16}"
            f"{detection:>8.2f}%"
            f"{false_alarm:>13.2f}%"
            f"{efficiency:>14.4f}"
        )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()