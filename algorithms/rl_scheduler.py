import numpy as np


class RLScheduler:

    def __init__(
        self,
        num_bands=10,
        dwell_times=(1, 2, 3),
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.2
    ):

        self.num_bands = num_bands
        self.dwell_times = list(dwell_times)

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.num_actions = (
            num_bands * len(self.dwell_times)
        )

        self.q_table = {}

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    def discretize_belief(self, belief):

        return tuple(
            np.digitize(
                belief,
                bins=[0.2, 0.5, 0.8]
            )
        )

    def get_state(self, belief):

        return self.discretize_belief(
            belief
        )

    # --------------------------------------------------------
    # Q TABLE
    # --------------------------------------------------------

    def _ensure_state(self, state):

        if state not in self.q_table:

            self.q_table[state] = np.zeros(
                self.num_actions
            )

    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    def action_to_parameters(self, action):

        band = (
            action
            // len(self.dwell_times)
        )

        dwell_index = (
            action
            % len(self.dwell_times)
        )

        dwell_time = (
            self.dwell_times[dwell_index]
        )

        return band, dwell_time

    def choose_action(
        self,
        state,
        training=True
    ):

        self._ensure_state(state)

        if (
            training
            and np.random.random()
            < self.epsilon
        ):

            action = np.random.randint(
                self.num_actions
            )

        else:

            action = int(
                np.argmax(
                    self.q_table[state]
                )
            )

        return action

    # --------------------------------------------------------
    # Q LEARNING
    # --------------------------------------------------------

    def update(
        self,
        state,
        action,
        reward,
        next_state
    ):

        self._ensure_state(state)
        self._ensure_state(next_state)

        current_q = (
            self.q_table[state][action]
        )

        best_next_q = np.max(
            self.q_table[next_state]
        )

        target = (
            reward
            + self.discount_factor
            * best_next_q
        )

        self.q_table[state][action] += (
            self.learning_rate
            * (target - current_q)
        )

    # --------------------------------------------------------
    # EXPLORATION DECAY
    # --------------------------------------------------------

    def decay_epsilon(
        self,
        minimum=0.02
    ):

        self.epsilon = max(
            minimum,
            self.epsilon * 0.995
        )


# ============================================================
# REWARD FUNCTION
# ============================================================

def calculate_reward(
    detected,
    signal_present,
    dwell_time,
    false_alarm=False
):

    reward = 0.0

    # Successful interception
    if (
        signal_present
        and detected
    ):

        reward += 10.0

    # Missed emitter
    elif (
        signal_present
        and not detected
    ):

        reward -= 3.0

    # False alarm
    if false_alarm:

        reward -= 4.0

    # Dwell-time penalty
    reward -= (
        0.5 * dwell_time
    )

    return reward