import numpy as np


class BayesianBelief:

    def __init__(
        self,
        num_bands=10,
        initial_probability=0.1
    ):
        self.num_bands = num_bands
        self.initial_probability = initial_probability

        self.belief = (
            np.ones(num_bands)
            * initial_probability
        )

    def reset(self):
        """
        Reset beliefs to initial uniform prior.
        """
        self.belief = (
            np.ones(self.num_bands)
            * self.initial_probability
        )

    def update(self, band, detected):

        if detected:

            self.belief[band] = min(
                0.99,
                self.belief[band] + 0.25
            )

        else:

            self.belief[band] = max(
                0.01,
                self.belief[band] - 0.05
            )

        return self.belief

    def get_best_band(self):

        return int(
            np.argmax(self.belief)
        )

    def get_beliefs(self):

        return self.belief.copy()


class BayesianScheduler:

    def __init__(self, num_bands=10):

        self.num_bands = num_bands

        self.belief_model = (
            BayesianBelief(num_bands)
        )

        self.visited = np.zeros(
            num_bands,
            dtype=bool
        )

    def reset(self):
        """
        Reset scheduler state and underlying belief model.
        """
        self.belief_model.reset()
        self.visited = np.zeros(
            self.num_bands,
            dtype=bool
        )

    def get_action(self):

        # Explore every band at least once
        unvisited = np.where(
            ~self.visited
        )[0]

        if len(unvisited) > 0:

            band = int(unvisited[0])

        else:

            band = (
                self.belief_model
                .get_best_band()
            )

        self.visited[band] = True

        probability = (
            self.belief_model
            .belief[band]
        )

        if probability >= 0.6:

            dwell_time = 3

        elif probability >= 0.3:

            dwell_time = 2

        else:

            dwell_time = 1

        return band, dwell_time

    def update(
        self,
        band,
        detected
    ):

        return self.belief_model.update(
            band,
            detected
        )


if __name__ == "__main__":

    scheduler = BayesianScheduler(10)

    print(
        "\nBayesian Scheduler Test"
    )

    print(
        "-----------------------"
    )

    for step in range(15):

        band, dwell = (
            scheduler.get_action()
        )

        detected = (
            band == 5
        )

        scheduler.update(
            band,
            detected
        )

        print(
            f"Step {step + 1:02d} | "
            f"Band: {band} | "
            f"Dwell: {dwell} | "
            f"Detection: {detected}"
        )

    print(
        "\nFinal beliefs:"
    )

    print(
        scheduler.belief_model
        .get_beliefs()
    )