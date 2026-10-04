"""
CASS-EW Centralized Configuration Module.

Provides structured, typed, and documented configuration parameters
for the RF environment, virtual receiver, reward function, schedulers,
and benchmark evaluations.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


@dataclass(frozen=True)
class EmitterSpec:
    """Specification for a synthetic RF emitter."""
    name: str
    band: int
    behavior: str  # "intermittent", "periodic", "hopping"
    activity_probability: float = 0.5
    hop_bands: Tuple[int, ...] = ()


@dataclass
class EnvironmentConfig:
    """Configuration for the RF spectrum environment."""
    num_bands: int = 10
    total_time: int = 500
    emitters: List[EmitterSpec] = field(default_factory=lambda: [
        EmitterSpec(
            name="Emitter_A",
            band=2,
            behavior="intermittent",
            activity_probability=0.7
        ),
        EmitterSpec(
            name="Emitter_B",
            band=5,
            behavior="periodic"
        ),
        EmitterSpec(
            name="Emitter_C",
            band=7,
            behavior="hopping",
            activity_probability=0.8,
            hop_bands=(6, 7, 8, 9)
        ),
    ])


@dataclass
class ReceiverConfig:
    """Configuration for the virtual EW receiver."""
    num_bands: int = 10
    detection_probability: float = 0.90
    false_alarm_probability: float = 0.05
    dwell_times: Tuple[int, ...] = (1, 2, 3)


@dataclass
class RewardConfig:
    """Configuration for the scheduling reward function."""
    detection_reward: float = 10.0
    miss_penalty: float = 3.0
    false_alarm_penalty: float = 4.0
    dwell_cost: float = 0.5


@dataclass
class RLSchedulerConfig:
    """Configuration for the Tabular Q-learning scheduler."""
    learning_rate: float = 0.1
    discount_factor: float = 0.90
    initial_epsilon: float = 0.30
    min_epsilon: float = 0.02
    epsilon_decay: float = 0.995
    belief_bins: Tuple[float, ...] = (0.2, 0.5, 0.8)
    train_episodes: int = 300
    steps_per_episode: int = 100


@dataclass
class BenchmarkConfig:
    """Configuration for reproducible common-trace benchmarking."""
    benchmark_seeds: Tuple[int, ...] = (42, 43, 44, 45, 46)
    total_time: int = 500
    training_seed: int = 42
    detector_seed_offset: int = 500


@dataclass
class Phase3SchedulerConfig:
    """Configuration for Phase 3 Adaptive Belief Scheduler."""
    initial_belief: float = 0.5
    belief_hit_update: float = 0.2
    belief_miss_update: float = 0.1
    epsilon: float = 0.1
    scheduler_seed: int = 42


@dataclass
class Phase4SchedulerConfig:
    """Configuration for Phase 4 Temporal Belief Scheduler."""
    temporal_history_size: int = 10
    minimum_detections: int = 3
    period_window_size: int = 5
    temporal_tolerance: int = 2
    belief_weight: float = 0.7
    temporal_weight: float = 0.3
    epsilon: float = 0.1

@dataclass
class CASSEWConfig:
    """Top-level unified system configuration."""
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    receiver: ReceiverConfig = field(default_factory=ReceiverConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    rl: RLSchedulerConfig = field(default_factory=RLSchedulerConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    phase3: Phase3SchedulerConfig = field(default_factory=Phase3SchedulerConfig)
    phase4: Phase4SchedulerConfig = field(default_factory=Phase4SchedulerConfig)


def get_default_config() -> CASSEWConfig:
    """Return an instance of the default CASS-EW configuration."""
    return CASSEWConfig()
