import numpy as np
import pandas as pd
import streamlit as st

from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver

from algorithms.baseline_scanner import SequentialScanner
from algorithms.bayesian_scheduler import BayesianScheduler
from algorithms.rl_scheduler import (
    RLScheduler,
    calculate_reward
)


NUM_BANDS = 10


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CASS-EW",
    page_icon="📡",
    layout="wide"
)


# ============================================================
# ENVIRONMENT
# ============================================================

def create_environment():

    environment = RFEnvironment(
        NUM_BANDS
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


# ============================================================
# RL TRAINING
# ============================================================

@st.cache_resource
def train_rl():

    np.random.seed(42)

    scheduler = RLScheduler(
        num_bands=NUM_BANDS,
        dwell_times=(1, 2, 3),
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.3
    )

    episodes = 300
    steps_per_episode = 100

    for episode in range(episodes):

        environment = create_environment()

        receiver = VirtualReceiver(
            num_bands=NUM_BANDS,
            detection_probability=0.90,
            false_alarm_probability=0.05
        )

        belief = np.ones(
            NUM_BANDS
        ) * 0.1

        for step in range(
            steps_per_episode
        ):

            # ------------------------------------------------
            # Current state
            # ------------------------------------------------

            state = scheduler.get_state(
                belief
            )

            # ------------------------------------------------
            # Choose action
            # ------------------------------------------------

            action = scheduler.choose_action(
                state,
                training=True
            )

            band, dwell = (
                scheduler.action_to_parameters(
                    action
                )
            )

            # ------------------------------------------------
            # Execute dwell
            # ------------------------------------------------

            history_start = len(
                receiver.scan_history
            )

            receiver.scan(
                environment,
                band,
                dwell
            )

            scan_results = (
                receiver.scan_history[
                    history_start:
                ]
            )

            # ------------------------------------------------
            # Improved reward
            # ------------------------------------------------

            reward = 0.0
            detections = 0

            for result in scan_results:

                signal_present = (
                    result["signal_present"]
                )

                detected = (
                    result["detected"]
                )

                false_alarm = (
                    not signal_present
                    and detected
                )

                if detected:
                    detections += 1

                reward += calculate_reward(
                    detected=detected,
                    signal_present=signal_present,
                    dwell_time=1,
                    false_alarm=false_alarm
                )

            # ------------------------------------------------
            # Belief update
            # ------------------------------------------------

            if detections > 0:

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
            for b in range(NUM_BANDS):

                if b != band:

                    belief[b] *= 0.99

            # ------------------------------------------------
            # Next state
            # ------------------------------------------------

            next_state = scheduler.get_state(
                belief
            )

            # ------------------------------------------------
            # Q-learning update
            # ------------------------------------------------

            scheduler.update(
                state,
                action,
                reward,
                next_state
            )

        # ----------------------------------------------------
        # Exploration decay
        # ----------------------------------------------------

        scheduler.decay_epsilon()

    # Greedy evaluation
    scheduler.epsilon = 0.0

    return scheduler


# ============================================================
# SESSION STATE
# ============================================================

if "initialized" not in st.session_state:

    st.session_state.initialized = True

    st.session_state.environment = (
        create_environment()
    )

    st.session_state.receiver = (
        VirtualReceiver(
            NUM_BANDS,
            detection_probability=0.90,
            false_alarm_probability=0.05
        )
    )

    st.session_state.belief = (
        np.ones(NUM_BANDS) * 0.1
    )

    st.session_state.step = 0
    st.session_state.detections = 0
    st.session_state.false_alarms = 0
    st.session_state.misses = 0
    st.session_state.total_dwell = 0

    st.session_state.history = []


environment = (
    st.session_state.environment
)

receiver = (
    st.session_state.receiver
)

belief = (
    st.session_state.belief
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ CASS-EW Control"
)

algorithm = st.sidebar.selectbox(
    "Scanning Strategy",
    [
        "RL Scheduler",
        "Bayesian Scheduler",
        "Sequential Scanner"
    ]
)

scan_count = st.sidebar.slider(
    "Automatic Scan Count",
    10,
    100,
    30,
    10
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📡 CASS-EW"
)

st.subheader(
    "Cognitive Adaptive Smart Scan for Electronic Warfare"
)

st.caption(
    "Adaptive AI-based frequency scanning and receiver scheduling"
)


# ============================================================
# CONTROL BUTTONS
# ============================================================

col1, col2 = st.columns(2)

run_button = col1.button(
    "▶ Run Next Scan",
    use_container_width=True
)

auto_button = col2.button(
    "⚡ Run Simulation",
    use_container_width=True
)


# ============================================================
# SCAN FUNCTION
# ============================================================

def execute_scan():

    global belief

    # --------------------------------------------------------
    # Select algorithm
    # --------------------------------------------------------

    if algorithm == "RL Scheduler":

        scheduler = train_rl()

        state = scheduler.get_state(
            belief
        )

        action = scheduler.choose_action(
            state,
            training=False
        )

        band, dwell = (
            scheduler.action_to_parameters(
                action
            )
        )

    elif algorithm == "Bayesian Scheduler":

        if "bayesian" not in st.session_state:

            st.session_state.bayesian = (
                BayesianScheduler(
                    NUM_BANDS
                )
            )

        scheduler = (
            st.session_state.bayesian
        )

        band, dwell = (
            scheduler.get_action()
        )

    else:

        if "sequential" not in st.session_state:

            st.session_state.sequential = (
                SequentialScanner(
                    NUM_BANDS
                )
            )

        scheduler = (
            st.session_state.sequential
        )

        band = scheduler.get_action()

        dwell = 1

    # --------------------------------------------------------
    # Receiver scan
    # --------------------------------------------------------

    history_start = len(
        receiver.scan_history
    )

    receiver.scan(
        environment,
        band,
        dwell
    )

    results = receiver.scan_history[
        history_start:
    ]

    # --------------------------------------------------------
    # Analyze scan
    # --------------------------------------------------------

    true_detections = 0
    false_alarms = 0
    misses = 0

    for result in results:

        signal = result[
            "signal_present"
        ]

        detected = result[
            "detected"
        ]

        if signal and detected:

            true_detections += 1

        elif signal and not detected:

            misses += 1

        elif not signal and detected:

            false_alarms += 1

    # --------------------------------------------------------
    # Belief update
    # --------------------------------------------------------

    detected = any(
        result["detected"]
        for result in results
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

    for b in range(NUM_BANDS):

        if b != band:

            belief[b] *= 0.99

    # --------------------------------------------------------
    # Update metrics
    # --------------------------------------------------------

    st.session_state.step += 1

    st.session_state.detections += (
        true_detections
    )

    st.session_state.false_alarms += (
        false_alarms
    )

    st.session_state.misses += (
        misses
    )

    st.session_state.total_dwell += (
        len(results)
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    st.session_state.history.append({

        "Step":
            st.session_state.step,

        "Algorithm":
            algorithm,

        "Band":
            band,

        "Dwell":
            dwell,

        "Signal":
            results[-1][
                "signal_present"
            ],

        "Detected":
            results[-1][
                "detected"
            ],

        "True Detection":
            true_detections,

        "False Alarm":
            false_alarms,

        "Miss":
            misses,

        "Belief":
            round(
                float(
                    belief[band]
                ),
                3
            )
    })

    return (
        band,
        dwell,
        detected
    )


# ============================================================
# BUTTON ACTION
# ============================================================

if run_button:

    execute_scan()

    st.rerun()


if auto_button:

    progress = st.progress(0)

    for i in range(scan_count):

        execute_scan()

        progress.progress(
            (i + 1) / scan_count
        )

    st.rerun()


# ============================================================
# TOP METRICS
# ============================================================

st.divider()

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "Scan Steps",
    st.session_state.step
)

m2.metric(
    "Detections",
    st.session_state.detections
)

m3.metric(
    "False Alarms",
    st.session_state.false_alarms
)

m4.metric(
    "Misses",
    st.session_state.misses
)

m5.metric(
    "Total Dwell",
    st.session_state.total_dwell
)


# ============================================================
# CURRENT AI DECISION
# ============================================================

st.divider()

st.header(
    "🧠 Current AI Decision"
)

if st.session_state.history:

    latest = (
        st.session_state.history[-1]
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Selected Band",
        latest["Band"]
    )

    c2.metric(
        "Dwell Time",
        latest["Dwell"]
    )

    c3.metric(
        "Detection",
        str(latest["Detected"])
    )

    c4.metric(
        "Band Belief",
        f'{latest["Belief"]:.3f}'
    )

else:

    st.info(
        "Run a scan to generate the first AI decision."
    )


# ============================================================
# RF BELIEF
# ============================================================

st.divider()

st.header(
    "📊 RF Band Belief"
)

belief_df = pd.DataFrame({

    "Band": [
        f"Band {i}"
        for i in range(NUM_BANDS)
    ],

    "Probability": [
        float(belief[i])
        for i in range(NUM_BANDS)
    ]

})

st.bar_chart(
    belief_df.set_index(
        "Band"
    )
)


# ============================================================
# ACTIVE RF ENVIRONMENT
# ============================================================

st.header(
    "📡 RF Environment"
)

active_bands = (
    environment.get_state()[
        "active_bands"
    ]
)

st.write(
    "**Currently active bands:**",
    sorted(active_bands)
)


# ============================================================
# BAND STATUS TABLE
# ============================================================

band_rows = []

for i in range(NUM_BANDS):

    band_rows.append({

        "Band":
            i,

        "AI Belief":
            round(
                float(
                    belief[i]
                ),
                3
            ),

        "Status":
            "🟢 ACTIVE"
            if i in active_bands
            else "⚪ Inactive"

    })


st.dataframe(
    band_rows,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SCAN HISTORY
# ============================================================

st.divider()

st.header(
    "📜 Scan History"
)

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df.tail(20),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Scan history will appear here."
    )


# ============================================================
# PERFORMANCE
# ============================================================

st.divider()

st.header(
    "📈 Live Performance"
)

signal_events = (
    st.session_state.detections
    + st.session_state.misses
)

if signal_events > 0:

    detection_rate = (
        st.session_state.detections
        / signal_events
        * 100
    )

else:

    detection_rate = 0


if st.session_state.step > 0:

    false_alarm_rate = (
        st.session_state.false_alarms
        / st.session_state.step
        * 100
    )

else:

    false_alarm_rate = 0


if st.session_state.total_dwell > 0:

    efficiency = (
        st.session_state.detections
        / st.session_state.total_dwell
    )

else:

    efficiency = 0


p1, p2, p3 = st.columns(3)

p1.metric(
    "Detection Rate",
    f"{detection_rate:.2f}%"
)

p2.metric(
    "False Alarm Rate",
    f"{false_alarm_rate:.2f}%"
)

p3.metric(
    "Scan Efficiency",
    f"{efficiency:.4f}"
)


# ============================================================
# SYSTEM PIPELINE
# ============================================================

st.divider()

st.header(
    "🔄 CASS-EW Decision Pipeline"
)

st.code(
    """
RF Environment
      ↓
Virtual EW Receiver
      ↓
Signal / Noise Observation
      ↓
Bayesian Belief State
      ↓
RL Decision Engine
      ↓
Band + Dwell Selection
      ↓
Receiver Scan
      ↓
Detection / Miss / False Alarm
      ↓
Reward + Belief Update
      ↺
    """,
    language="text"
)


# ============================================================
# RESET
# ============================================================

st.divider()

if st.button(
    "🔄 Reset Simulation",
    use_container_width=True
):

    for key in list(
        st.session_state.keys()
    ):

        del st.session_state[key]

    st.rerun()