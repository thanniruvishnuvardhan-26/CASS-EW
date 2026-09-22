import numpy as np

from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver

from algorithms.rl_scheduler import RLScheduler


def create_environment():

    environment = RFEnvironment(num_bands=10)

    environment.add_emitter(
        Emitter(
            name="Emitter_A",
            band=2,
            behavior="intermittent",
            activity_probability=0.7
        )
    )

    environment.add_emitter(
        Emitter(
            name="Emitter_B",
            band=5,
            behavior="periodic"
        )
    )

    environment.add_emitter(
        Emitter(
            name="Emitter_C",
            band=7,
            behavior="hopping",
            activity_probability=0.8,
            hop_bands=[6, 7, 8, 9]
        )
    )

    return environment


def train_agent():

    scheduler = RLScheduler(
        num_bands=10,
        dwell_times=(1, 2, 3),
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.3
    )

    for episode in range(300):

        environment = create_environment()

        receiver = VirtualReceiver(
            num_bands=10
        )

        belief = np.ones(10) * 0.1

        for _ in range(100):

            state = scheduler.get_state(
                belief
            )

            action = scheduler.choose_action(
                state,
                training=True
            )

            band, dwell_time = (
                scheduler.action_to_parameters(
                    action
                )
            )

            detected = receiver.scan(
                environment,
                band,
                dwell_time
            )

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

            for b in range(10):

                if b != band:

                    belief[b] *= 0.99

            reward = (
                10.0 if detected else 0.0
            )

            reward -= 0.5 * dwell_time

            next_state = scheduler.get_state(
                belief
            )

            scheduler.update(
                state,
                action,
                reward,
                next_state
            )

        scheduler.decay_epsilon()

    return scheduler


def evaluate(scheduler):

    # Turn off exploration
    scheduler.epsilon = 0.0

    environment = create_environment()

    receiver = VirtualReceiver(
        num_bands=10
    )

    belief = np.ones(10) * 0.1

    total_steps = 200
    total_detections = 0
    total_dwell = 0

    print("\nRL Evaluation")
    print("-------------")

    for step in range(total_steps):

        state = scheduler.get_state(
            belief
        )

        action = scheduler.choose_action(
            state,
            training=False
        )

        band, dwell_time = (
            scheduler.action_to_parameters(
                action
            )
        )

        detected = receiver.scan(
            environment,
            band,
            dwell_time
        )

        if detected:

            total_detections += 1

            belief[band] = min(
                0.99,
                belief[band] + 0.3
            )

        else:

            belief[band] = max(
                0.01,
                belief[band] - 0.1
            )

        for b in range(10):

            if b != band:

                belief[b] *= 0.99

        total_dwell += dwell_time

        if (step + 1) % 20 == 0:

            print(
                f"Step {step + 1:03d} "
                f"| Band: {band} "
                f"| Dwell: {dwell_time} "
                f"| Detection: {detected}"
            )

    detection_rate = (
        total_detections / total_steps
    ) * 100

    scan_efficiency = (
        total_detections / total_dwell
    )

    print("\n-------------")
    print(f"Total Steps      : {total_steps}")
    print(f"Total Detections : {total_detections}")
    print(f"Total Dwell      : {total_dwell}")
    print(f"Detection Rate   : {detection_rate:.2f}%")
    print(f"Scan Efficiency  : {scan_efficiency:.4f}")


if __name__ == "__main__":

    np.random.seed(42)

    print("Training RL Agent...")

    trained_scheduler = train_agent()

    print("Training completed.")

    evaluate(trained_scheduler)