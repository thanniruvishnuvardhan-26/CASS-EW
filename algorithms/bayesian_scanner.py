import numpy as np

from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver
from algorithms.bayesian_scheduler import BayesianScheduler


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


if __name__ == "__main__":

    np.random.seed(42)

    environment = create_environment()

    receiver = VirtualReceiver(
        num_bands=10
    )

    scheduler = BayesianScheduler(
        num_bands=10
    )

    total_steps = 100
    total_detections = 0
    total_dwell = 0

    print("\nCASS-EW Bayesian Smart Scanner")
    print("--------------------------------")

    for step in range(total_steps):

        # Scheduler chooses band and dwell time
        band, dwell_time = scheduler.get_action()

        # Receiver scans selected band
        detected = receiver.scan(
            environment,
            band,
            dwell_time
        )

        # Update Bayesian belief
        scheduler.update(
            band,
            detected
        )

        if detected:
            total_detections += 1

        total_dwell += dwell_time

        print(
            f"Step {step + 1:03d} "
            f"| Band: {band} "
            f"| Dwell: {dwell_time} "
            f"| Detection: {detected} "
            f"| Belief: "
            f"{scheduler.belief_model.belief[band]:.3f}"
        )

    detection_rate = (
        total_detections / total_steps
    ) * 100

    scan_efficiency = (
        total_detections / total_dwell
    )

    print("\n--------------------------------")
    print(f"Total Steps      : {total_steps}")
    print(f"Total Detections : {total_detections}")
    print(f"Total Dwell      : {total_dwell}")
    print(f"Detection Rate   : {detection_rate:.2f}%")
    print(f"Scan Efficiency  : {scan_efficiency:.4f}")