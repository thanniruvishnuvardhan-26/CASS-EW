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
        belief[band] = min(0.99, belief[band] + 0.3)
    else:
        belief[band] = max(0.01, belief[band] - 0.1)

    for b in range(NUM_BANDS):
        if b != band:
            belief[b] *= 0.99

    return belief


def evaluate_sequential(seed):
    np.random.seed(seed)

    environment = create_environment()
    receiver = VirtualReceiver(NUM_BANDS)
    scanner = SequentialScanner(NUM_BANDS)

    detections = 0
    dwell = 0

    for _ in range(STEPS):
        band = scanner.get_action()
        detected = receiver.scan(environment, band, 1)

        if detected:
            detections += 1

        dwell += 1

    return detections, dwell


def evaluate_bayesian(seed):
    np.random.seed(seed)

    environment = create_environment()
    receiver = VirtualReceiver(NUM_BANDS)
    scheduler = BayesianScheduler(NUM_BANDS)

    detections = 0
    dwell = 0

    for _ in range(STEPS):
        band, dwell_time = scheduler.get_action()

        detected = receiver.scan(
            environment,
            band,
            dwell_time
        )

        if detected:
            detections += 1

        dwell += dwell_time

        scheduler.update(band, detected)

    return detections, dwell


def train_rl(seed):
    np.random.seed(seed)

    scheduler = RLScheduler(
        NUM_BANDS,
        (1, 2, 3),
        0.1,
        0.9,
        0.3
    )

    for _ in range(300):

        environment = create_environment()
        receiver = VirtualReceiver(NUM_BANDS)

        belief = np.ones(NUM_BANDS) * 0.1

        for _ in range(100):

            state = scheduler.get_state(belief)

            action = scheduler.choose_action(
                state,
                training=True
            )

            band, dwell_time = scheduler.action_to_parameters(action)

            detected = receiver.scan(
                environment,
                band,
                dwell_time
            )

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
    receiver = VirtualReceiver(NUM_BANDS)

    belief = np.ones(NUM_BANDS) * 0.1

    detections = 0
    dwell = 0

    for _ in range(STEPS):

        state = scheduler.get_state(belief)

        action = scheduler.choose_action(
            state,
            training=False
        )

        band, dwell_time = scheduler.action_to_parameters(action)

        detected = receiver.scan(
            environment,
            band,
            dwell_time
        )

        if detected:
            detections += 1

        dwell += dwell_time

        belief = update_belief(
            belief,
            band,
            detected
        )

    return detections, dwell


def print_result(name, detections, dwell):

    detection_rate = (detections / STEPS) * 100
    efficiency = detections / dwell

    print(
        f"{name:<18}"
        f"{detection_rate:>10.2f}%"
        f"{efficiency:>18.4f}"
    )


if __name__ == "__main__":

    print("\nCASS-EW Controlled Algorithm Evaluation")
    print("---------------------------------------")

    seeds = [42, 43, 44, 45, 46]

    rl_scheduler = train_rl(42)

    results = {
        "Sequential": [],
        "Bayesian": [],
        "RL": []
    }

    for seed in seeds:

        results["Sequential"].append(
            evaluate_sequential(seed)
        )

        results["Bayesian"].append(
            evaluate_bayesian(seed)
        )

        results["RL"].append(
            evaluate_rl(rl_scheduler, seed)
        )

    print(
        "\nAlgorithm          Detection Rate     Scan Efficiency"
    )
    print(
        "-------------------------------------------------------"
    )

    for name, values in results.items():

        detection_rates = [
            (d / STEPS) * 100
            for d, dwell in values
        ]

        efficiencies = [
            d / dwell
            for d, dwell in values
        ]

        print(
            f"{name:<18}"
            f"{np.mean(detection_rates):>10.2f}%"
            f"{np.mean(efficiencies):>18.4f}"
        )

    print("\nEvaluation completed.")