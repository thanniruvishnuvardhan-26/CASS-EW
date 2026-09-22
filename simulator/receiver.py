import numpy as np


class VirtualReceiver:

    def __init__(
        self,
        num_bands=10,
        detection_probability=0.90,
        false_alarm_probability=0.05
    ):
        self.num_bands = num_bands
        self.detection_probability = detection_probability
        self.false_alarm_probability = false_alarm_probability
        self.scan_history = []

    def scan(self, environment, band, dwell_time=1):

        detections = []

        for _ in range(dwell_time):

            active_bands = environment.step()

            signal_present = band in active_bands

            if signal_present:
                detected = (
                    np.random.random()
                    < self.detection_probability
                )
            else:
                detected = (
                    np.random.random()
                    < self.false_alarm_probability
                )

            detections.append(detected)

            self.scan_history.append({
                "time": environment.time,
                "band": band,
                "signal_present": signal_present,
                "detected": detected
            })

        return any(detections)

    def get_last_result(self):

        if not self.scan_history:
            return None

        return self.scan_history[-1]


if __name__ == "__main__":

    from simulator.environment import RFEnvironment, Emitter

    np.random.seed(42)

    environment = RFEnvironment(10)

    environment.add_emitter(
        Emitter("Emitter_A", 2, "intermittent", 0.7)
    )

    receiver = VirtualReceiver(
        num_bands=10,
        detection_probability=0.90,
        false_alarm_probability=0.05
    )

    print("\nCASS-EW Realistic Receiver Test")
    print("--------------------------------")

    for band in [0, 2, 5, 7, 9]:

        detected = receiver.scan(
            environment,
            band,
            1
        )

        result = receiver.get_last_result()

        print(
            f"Time {result['time']:02d} | "
            f"Band: {band} | "
            f"Signal: {result['signal_present']} | "
            f"Detected: {detected}"
        )