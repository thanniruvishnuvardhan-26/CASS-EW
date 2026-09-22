import numpy as np

from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver

from algorithms.baseline_scanner import SequentialScanner
from algorithms.bayesian_scheduler import BayesianScheduler
from algorithms.rl_scheduler import RLScheduler


NUM_BANDS = 10
MAX_TIME = 200


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


def measure_sequential(seed):

    np.random.seed(seed)

    environment = create_environment()

    receiver = VirtualReceiver(
        NUM_BANDS,
        detection_probability=0.90,
        false_alarm_probability=0.05
    )

    scanner = SequentialScanner(NUM_BANDS)

    intercept_times = {}
    active_since = {}

    for time in range(1, MAX_TIME + 1):

        band = scanner.get_action()

        receiver.scan(
            environment,
            band,
            1
        )

        result = receiver.get_last_result()

        # Track when each emitter becomes active
        for emitter in environment.emitters:

            if emitter.active:

                if emitter.name not in active_since:
                    active_since[emitter.name] = time

            else:
                active_since.pop(emitter.name, None)

        # Detection of an active emitter
        if result["signal_present"] and result["detected"]:

            for emitter in environment.emitters:

                if emitter.active and emitter.band == band:

                    if emitter.name not in intercept_times:
                        intercept_times[emitter.name] = (
                            time - active_since[emitter.name]
                        )

    return intercept_times


def measure_bayesian(seed):

    np.random.seed(seed)

    environment = create_environment()

    receiver = VirtualReceiver(
        NUM_BANDS,
        detection_probability=0.90,
        false_alarm_probability=0.05
    )

    scheduler = BayesianScheduler(NUM_BANDS)

    intercept_times = {}
    active_since = {}

    time = 0

    while time < MAX_TIME:

        band, dwell_time = scheduler.get_action()

        for _ in range(dwell_time):

            time += 1

            if time > MAX_TIME:
                break

            receiver.scan(
                environment,
                band,
                1
            )

            result = receiver.get_last_result()

            for emitter in environment.emitters:

                if emitter.active:

                    if emitter.name not in active_since:
                        active_since[emitter.name] = time

                else:
                    active_since.pop(emitter.name, None)

            if result["signal_present"] and result["detected"]:

                for emitter in environment.emitters:

                    if emitter.active and emitter.band == band:

                        if emitter.name not in intercept_times:
                            intercept_times[emitter.name] = (
                                time - active_since[emitter.name]
                            )

        scheduler.update(
            band,
            receiver.get_last_result()["detected"]
        )

    return intercept_times


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

            band, dwell_time = scheduler.action_to_parameters(action)

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


def measure_rl(scheduler, seed):

    np.random.seed(seed)

    environment = create_environment()

    receiver = VirtualReceiver(
        NUM_BANDS,
        detection_probability=0.90,
        false_alarm_probability=0.05
    )

    belief = np.ones(NUM_BANDS) * 0.1

    intercept_times = {}
    active_since = {}

    time = 0

    while time < MAX_TIME:

        state = scheduler.get_state(belief)

        action = scheduler.choose_action(
            state,
            training=False
        )

        band, dwell_time = scheduler.action_to_parameters(action)

        for _ in range(dwell_time):

            time += 1

            if time > MAX_TIME:
                break

            receiver.scan(
                environment,
                band,
                1
            )

            result = receiver.get_last_result()

            for emitter in environment.emitters:

                if emitter.active:

                    if emitter.name not in active_since:
                        active_since[emitter.name] = time

                else:
                    active_since.pop(emitter.name, None)

            if result["signal_present"] and result["detected"]:

                for emitter in environment.emitters:

                    if emitter.active and emitter.band == band:

                        if emitter.name not in intercept_times:
                            intercept_times[emitter.name] = (
                                time - active_since[emitter.name]
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

    return intercept_times


def average_intercept_time(all_results):

    values = []

    for result in all_results:

        values.extend(
            result.values()
        )

    if not values:
        return None

    return np.mean(values)


def main():

    print("\nCASS-EW Intercept Time Evaluation")
    print("=================================")

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
            measure_sequential(seed)
        )

        results["Bayesian"].append(
            measure_bayesian(seed)
        )

        results["RL"].append(
            measure_rl(
                rl_scheduler,
                seed
            )
        )

    print(
        "Algorithm          Avg Intercept Time"
    )

    print(
        "---------------------------------------"
    )

    for name, values in results.items():

        avg_time = average_intercept_time(values)

        if avg_time is None:
            print(
                f"{name:<20} No interceptions"
            )
        else:
            print(
                f"{name:<20} {avg_time:.2f} time units"
            )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()