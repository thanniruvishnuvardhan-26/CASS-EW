from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver


class SequentialScanner:

    def __init__(self, num_bands=10):
        self.num_bands = num_bands
        self.current_band = 0

    def get_action(self):
        band = self.current_band

        self.current_band = (
            self.current_band + 1
        ) % self.num_bands

        return band


if __name__ == "__main__":

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

    receiver = VirtualReceiver(num_bands=10)
    scanner = SequentialScanner(num_bands=10)

    total_steps = 100
    total_detections = 0

    print("\nCASS-EW Sequential Scanner")
    print("--------------------------")

    for step in range(total_steps):

        band = scanner.get_action()

        detected = receiver.scan(
            environment,
            band,
            dwell_time=1
        )

        if detected:
            total_detections += 1

        print(
            f"Step {step + 1:03d} "
            f"| Scanned Band: {band} "
            f"| Detection: {detected}"
        )

    detection_rate = (
        total_detections / total_steps
    ) * 100

    print("\n--------------------------")
    print(f"Total Steps      : {total_steps}")
    print(f"Total Detections : {total_detections}")
    print(f"Detection Rate   : {detection_rate:.2f}%")