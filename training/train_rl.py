import numpy as np

from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver

from algorithms.rl_scheduler import (
    RLScheduler,
    calculate_reward
)


def create_environment():

    environment = RFEnvironment(
        num_bands=10
    )

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


def train():

    np.random.seed(42)

    scheduler = RLScheduler(
        num_bands=10,
        dwell_times=(1, 2, 3),
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.3
    )

    episodes = 200
    steps_per_episode = 100

    reward_history = []

    print("\nCASS-EW RL Training")
    print("-------------------")

    for episode in range(episodes):

        environment = create_environment()

        receiver = VirtualReceiver(
            num_bands=10
        )

        belief = np.ones(10) * 0.1

        total_reward = 0.0

        for step in range(steps_per_episode):

            # Current state
            state = scheduler.get_state(belief)

            # Choose action
            action = scheduler.choose_action(
                state,
                training=True
            )

            # Convert action to band + dwell
            band, dwell_time = scheduler.action_to_parameters(
                action
            )

            # Remember history position before scanning
            history_start = len(receiver.scan_history)

            # Scan
            detected = receiver.scan(
                environment,
                band,
                dwell_time
            )

            # Get individual observations from this dwell
            scan_results = receiver.scan_history[
                history_start:
            ]

            # Calculate reward from each observation
            reward = 0.0

            for result in scan_results:

                signal_present = result["signal_present"]
                detection = result["detected"]

                false_alarm = (
                    not signal_present
                    and detection
                )

                reward += calculate_reward(
                    detected=detection,
                    signal_present=signal_present,
                    dwell_time=1,
                    false_alarm=false_alarm
                )

            total_reward += reward

            # Update belief
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

            # Slight decay for unobserved bands
            for b in range(10):

                if b != band:
                    belief[b] *= 0.99

            # Next state
            next_state = scheduler.get_state(
                belief
            )

            # Q-learning update
            scheduler.update(
                state,
                action,
                reward,
                next_state
            )

        scheduler.decay_epsilon()

        reward_history.append(
            total_reward
        )

        if (episode + 1) % 20 == 0:

            average_reward = np.mean(
                reward_history[-20:]
            )

            print(
                f"Episode {episode + 1:03d} "
                f"| Average Reward: "
                f"{average_reward:.2f} "
                f"| Epsilon: "
                f"{scheduler.epsilon:.3f}"
            )

    print("\nTraining Complete")

    print(
        f"Final Epsilon: "
        f"{scheduler.epsilon:.3f}"
    )

    print(
        f"Learned States: "
        f"{len(scheduler.q_table)}"
    )

    return scheduler, reward_history


if __name__ == "__main__":
    train()