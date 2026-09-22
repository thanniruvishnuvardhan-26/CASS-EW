import numpy as np


class Emitter:
    """
    Represents a synthetic RF emitter.

    The emitter can be:
    - intermittent
    - periodic
    - frequency-hopping
    """

    def __init__(
        self,
        name,
        band,
        behavior="intermittent",
        activity_probability=0.3,
        hop_bands=None
    ):
        self.name = name
        self.band = band
        self.behavior = behavior
        self.activity_probability = activity_probability

        self.hop_bands = hop_bands or []
        self.hop_index = 0

        self.active = False

    def step(self):
        """
        Advance the emitter by one time step.
        """

        # Intermittent emitter
        if self.behavior == "intermittent":

            self.active = (
                np.random.random()
                < self.activity_probability
            )

        # Periodic emitter
        elif self.behavior == "periodic":

            self.active = not self.active

        # Frequency-hopping emitter
        elif self.behavior == "hopping":

            self.active = (
                np.random.random()
                < self.activity_probability
            )

            if self.active and self.hop_bands:

                self.band = self.hop_bands[self.hop_index]

                self.hop_index = (
                    self.hop_index + 1
                ) % len(self.hop_bands)

        return self.active


class RFEnvironment:
    """
    Simulated electromagnetic environment.
    """

    def __init__(self, num_bands=10):

        self.num_bands = num_bands

        self.emitters = []

        self.time = 0

    def add_emitter(self, emitter):
        """
        Add an emitter to the environment.
        """

        self.emitters.append(emitter)

    def step(self):
        """
        Advance the environment by one time step.

        Returns the ground-truth active bands.
        """

        self.time += 1

        active_bands = set()

        for emitter in self.emitters:

            is_active = emitter.step()

            if is_active:

                active_bands.add(emitter.band)

        return active_bands

    def get_state(self):
        """
        Return current ground-truth environment state.
        """

        return {
            "time": self.time,
            "active_bands": [
                emitter.band
                for emitter in self.emitters
                if emitter.active
            ]
        }


if __name__ == "__main__":

    # Create environment
    environment = RFEnvironment(num_bands=10)

    # Add intermittent emitter
    emitter1 = Emitter(
        name="Emitter_A",
        band=2,
        behavior="intermittent",
        activity_probability=0.5
    )

    # Add periodic emitter
    emitter2 = Emitter(
        name="Emitter_B",
        band=5,
        behavior="periodic"
    )

    # Add frequency-hopping emitter
    emitter3 = Emitter(
        name="Emitter_C",
        band=7,
        behavior="hopping",
        activity_probability=0.7,
        hop_bands=[6, 7, 8, 9]
    )

    environment.add_emitter(emitter1)
    environment.add_emitter(emitter2)
    environment.add_emitter(emitter3)

    # Run simulation
    print("\nCASS-EW RF Environment Simulation")
    print("----------------------------------")

    for _ in range(20):

        active_bands = environment.step()

        print(
            f"Time {environment.time:02d} "
            f"| Active bands: {sorted(active_bands)}"
        )