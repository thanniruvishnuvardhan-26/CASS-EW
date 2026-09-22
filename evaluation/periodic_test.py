import numpy as np

from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver

from algorithms.baseline_scanner import SequentialScanner
from algorithms.bayesian_scheduler import BayesianScheduler
from algorithms.rl_scheduler import RLScheduler


NUM_BANDS = 10
STEPS = 200
PERIODIC_BAND = 5


def create_periodic_environment():

    environment = RFEnvironment(NUM_BANDS)

    # Main periodic emitter
    environment.add_emitter(
        Emitter(
            "Periodic_Emitter",
            PERIODIC_BAND,
            "periodic"
        )
    )

    # Additional intermittent emitter
    environment.add_emitter(
        Emitter(
            "Intermittent_Emitter",
            2,
            "intermittent",
            0.3
        )
    )

    return environment


def test_sequential(seed):

    np.random.seed(seed)

    environment = create_periodic_environment()

    receiver = VirtualReceiver(
        NUM_BANDS,
        detection_probability=0.90,
        false_alarm_probability=0.05
    )

    scanner = SequentialScanner(NUM_BANDS)

    detections = 0
    periodic_detections = 0

    for _ in range(STEPS):

        band = scanner.get_action()

        receiver.scan(
            environment,
            band,
            1
        )

        result = receiver.get_last_result()

        if result["detected"]:
            detections += 1

        if (
            result["signal_present"]
            and result["detected"]
            and band == PERIODIC_BAND
        ):
            periodic_detections += 1

    return detections, periodic_detections


def test_bayesian(seed):

    np.random.seed(seed)

    environment = create_periodic_environment()

    receiver = VirtualReceiver(
        NUM_BANDS,
        detection_probability=0.90,
        false_alarm_probability=0.05
    )

    scheduler = BayesianScheduler(NUM_BANDS)

    detections = 0
    periodic_detections = 0

    for _ in range(STEPS):

        band, dwell_time = scheduler.get_action()

        receiver.scan(
            environment,
            band,
            dwell_time
        )

        results = receiver.scan_history[-dwell_time:]

        for result in results:

            if result["detected"]:
                detections += 1

            if (
                result["signal_present"]
                and result["detected"]
                and band == PERIODIC_BAND
            ):
                periodic_detections += 1

        scheduler.update(
            band,
            receiver.get_last_result()["detected"]
        )

    return detections, periodic_detections


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

        environment = create_periodic_environment()

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

            reward = (
                10.0 if detected else 0.0
            )

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


def test_rl(scheduler, seed):

    np.random.seed(seed)

    environment = create_periodic_environment()

    receiver = VirtualReceiver(
        NUM_BANDS,
        detection_probability=0.90,
        false_alarm_probability=0.05
    )

    belief = np.ones(NUM_BANDS) * 0.1

    detections = 0
    periodic_detections = 0

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

            if result["detected"]:
                detections += 1

            if (
                result["signal_present"]
                and result["detected"]
                and band == PERIODIC_BAND
            ):
                periodic_detections += 1

        detected = receiver.get_last_result()["detected"]

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

    return detections, periodic_detections


def main():

    print("\nCASS-EW Periodic Emitter Evaluation")
    print("===================================")

    seeds = [42, 43, 44, 45, 46]

    print("\nTraining RL scheduler...")

    rl_scheduler = train_rl()

    print("RL training completed.\n")

    results = {
        "Sequential": [],
        "Bayesian": [],
        "RL": []
    }

    for seed in seeds:

        results["Sequential"].append(
            test_sequential(seed)
        )

        results["Bayesian"].append(
            test_bayesian(seed)
        )

        results["RL"].append(
            test_rl(
                rl_scheduler,
                seed
            )
        )

    print(
        "Algorithm          Total Detections    "
        "Periodic Detections"
    )

    print(
        "------------------------------------------------"
    )

    for name, values in results.items():

        total = np.mean(
            [v[0] for v in values]
        )

        periodic = np.mean(
            [v[1] for v in values]
        )

        print(
            f"{name:<20}"
            f"{total:>10.2f}"
            f"{periodic:>20.2f}"
        )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()