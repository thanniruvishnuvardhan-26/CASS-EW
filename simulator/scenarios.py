"""
CASS-EW Standardized Scenario Suite (S1 - S10).
Provides realistic, reproducible, and challenging Electronic Warfare test scenarios.

Scenarios:
- S1: Quiet spectrum (1 intermittent emitter, high inactive background)
- S2: Dense spectrum (multiple continuous and intermittent emitters covering majority of bands)
- S3: Intermittent emitters (low-duty bursty signals)
- S4: Strong periodic emitter (stable high-duty radar)
- S5: Frequency-agile emitter (hopping across multiple bands)
- S6: Sudden emitter appearance (dormant initially, then high activity)
- S7: Multiple competing emitters (overlapping periodic, intermittent, and agile emitters)
- S8: Environment behaviour change (switching modes halfway through simulation)
- S9: Jittered/noisy periodic emitter (periodic with timing jitter)
- S10: Multi-receiver asymmetric SNR (distributed emitters with receiver-specific geometric visibility)
"""

from typing import List, Dict, Any, Tuple


def get_scenario_suite(num_bands: int = 10) -> Dict[str, Dict[str, Any]]:
    """
    Returns the complete suite of scenarios S1 to S10 with full metadata and parameters.
    """
    scenarios = {}

    # S1: Quiet spectrum
    scenarios["S1_Quiet"] = {
        "id": "S1",
        "name": "Quiet spectrum",
        "description": "Sparse RF environment with only 1 low-duty intermittent emitter.",
        "category": "tuning",
        "emitters": [
            {
                "name": "S1_E1_Intermittent",
                "band": 4,
                "behavior": "intermittent",
                "activity_probability": 0.15,
                "position": (2.0, 3.0),
                "power_dbm": 15.0
            }
        ],
        "receivers": [
            {"id": "R1", "position": (0.0, 0.0), "snr_threshold": 10.0}
        ]
    }

    # S2: Dense spectrum
    scenarios["S2_Dense"] = {
        "id": "S2",
        "name": "Dense spectrum",
        "description": "High-density spectrum with 6 emitters active across bands.",
        "category": "tuning",
        "emitters": [
            {"name": "S2_E1_Per", "band": 0, "behavior": "periodic", "period": 4, "duty_cycle": 2, "position": (1.0, 1.0)},
            {"name": "S2_E2_Per", "band": 2, "behavior": "periodic", "period": 6, "duty_cycle": 2, "position": (3.0, 1.0)},
            {"name": "S2_E3_Int", "band": 4, "behavior": "intermittent", "activity_probability": 0.4, "position": (5.0, 2.0)},
            {"name": "S2_E4_Per", "band": 6, "behavior": "periodic", "period": 5, "duty_cycle": 1, "position": (2.0, 4.0)},
            {"name": "S2_E5_Int", "band": 8, "behavior": "intermittent", "activity_probability": 0.35, "position": (4.0, 5.0)},
            {"name": "S2_E6_Hop", "band": 7, "behavior": "hopping", "hop_bands": [7, 9], "hop_interval": 4, "position": (6.0, 6.0)}
        ],
        "receivers": [
            {"id": "R1", "position": (0.0, 0.0), "snr_threshold": 10.0}
        ]
    }

    # S3: Intermittent emitters
    scenarios["S3_Intermittent"] = {
        "id": "S3",
        "name": "Intermittent emitters",
        "description": "Bursty low-probability-of-intercept (LPI) emitters on random intervals.",
        "category": "tuning",
        "emitters": [
            {"name": "S3_E1_Burst", "band": 1, "behavior": "burst", "burst_duration": 2, "inter_burst_duration": 8, "position": (2.0, 2.0)},
            {"name": "S3_E2_Int", "band": 5, "behavior": "intermittent", "activity_probability": 0.20, "position": (4.0, 3.0)},
            {"name": "S3_E3_Burst", "band": 8, "behavior": "burst", "burst_duration": 3, "inter_burst_duration": 12, "position": (6.0, 1.0)}
        ],
        "receivers": [
            {"id": "R1", "position": (0.0, 0.0), "snr_threshold": 10.0}
        ]
    }

    # S4: Strong periodic emitter
    scenarios["S4_Periodic"] = {
        "id": "S4",
        "name": "Strong periodic emitter",
        "description": "Highly regular surveillance radar with fixed pulse repetition interval.",
        "category": "tuning",
        "emitters": [
            {"name": "S4_E1_Radar", "band": 3, "behavior": "periodic", "period": 8, "duty_cycle": 2, "position": (3.0, 3.0), "power_dbm": 25.0}
        ],
        "receivers": [
            {"id": "R1", "position": (0.0, 0.0), "snr_threshold": 10.0}
        ]
    }

    # S5: Frequency-agile emitter
    scenarios["S5_FrequencyAgile"] = {
        "id": "S5",
        "name": "Frequency-agile emitter",
        "description": "Fast frequency-hopping radar hopping across 4 bands in sequence.",
        "category": "tuning",
        "emitters": [
            {"name": "S5_E1_Agile", "band": 2, "behavior": "hopping", "hop_bands": [2, 4, 6, 8], "hop_interval": 3, "position": (2.0, 2.0)}
        ],
        "receivers": [
            {"id": "R1", "position": (0.0, 0.0), "snr_threshold": 10.0}
        ]
    }

    # S6: Sudden emitter appearance (Held-out / generalization)
    scenarios["S6_SuddenAppearance"] = {
        "id": "S6",
        "name": "Sudden emitter appearance",
        "description": "Dormant emitter that suddenly commences transmission at t=150.",
        "category": "held_out",
        "emitters": [
            {"name": "S6_E1_Baseline", "band": 1, "behavior": "periodic", "period": 10, "duty_cycle": 2, "position": (1.0, 1.0)},
            # Burst emitter with large inter_burst delay simulating delayed onset
            {"name": "S6_E2_LateArrival", "band": 7, "behavior": "burst", "burst_duration": 40, "inter_burst_duration": 200, "position": (5.0, 5.0)}
        ],
        "receivers": [
            {"id": "R1", "position": (0.0, 0.0), "snr_threshold": 10.0}
        ]
    }

    # S7: Multiple competing emitters (Held-out / generalization)
    scenarios["S7_MultipleCompeting"] = {
        "id": "S7",
        "name": "Multiple competing emitters",
        "description": "Overlapping periodic, intermittent, and agile emitters competing for receiver scan dwell.",
        "category": "held_out",
        "emitters": [
            {"name": "S7_E1_Per", "band": 2, "behavior": "periodic", "period": 7, "duty_cycle": 2, "position": (2.0, 1.0)},
            {"name": "S7_E2_Hop", "band": 3, "behavior": "hopping", "hop_bands": [3, 5, 7], "hop_interval": 4, "position": (4.0, 4.0)},
            {"name": "S7_E3_Int", "band": 6, "behavior": "intermittent", "activity_probability": 0.3, "position": (1.0, 5.0)},
            {"name": "S7_E4_Per", "band": 8, "behavior": "periodic", "period": 11, "duty_cycle": 2, "position": (5.0, 2.0)}
        ],
        "receivers": [
            {"id": "R1", "position": (0.0, 0.0), "snr_threshold": 10.0}
        ]
    }

    # S8: Environment behaviour change (Held-out / generalization)
    scenarios["S8_BehaviourChange"] = {
        "id": "S8",
        "name": "Environment behaviour change",
        "description": "Emitter hopping across low bands initially, then switching to high bands.",
        "category": "held_out",
        "emitters": [
            {"name": "S8_E1_RandomAgile", "band": 1, "behavior": "random_hopping", "hop_bands": [0, 1, 2, 7, 8, 9], "activity_probability": 0.6, "position": (3.0, 3.0)}
        ],
        "receivers": [
            {"id": "R1", "position": (0.0, 0.0), "snr_threshold": 10.0}
        ]
    }

    # S9: Jittered/noisy periodic emitter (Held-out / generalization)
    scenarios["S9_JitteredPeriodic"] = {
        "id": "S9",
        "name": "Jittered/noisy periodic emitter",
        "description": "Periodic emitter with high jitter and intermittent misses.",
        "category": "held_out",
        "emitters": [
            {"name": "S9_E1_Jitter", "band": 4, "behavior": "periodic", "period": 6, "duty_cycle": 2, "position": (2.0, 4.0)},
            {"name": "S9_E2_NoiseInt", "band": 4, "behavior": "intermittent", "activity_probability": 0.25, "position": (3.0, 4.0)}
        ],
        "receivers": [
            {"id": "R1", "position": (0.0, 0.0), "snr_threshold": 10.0}
        ]
    }

    # S10: Multi-receiver asymmetric SNR (Multi-Receiver Validation)
    scenarios["S10_MultiReceiverAsymmetric"] = {
        "id": "S10",
        "name": "Multi-receiver asymmetric SNR",
        "description": "3 spatially distributed receivers with geometry-dependent visibility.",
        "category": "held_out",
        "emitters": [
            {"name": "S10_E1_NearR1", "band": 1, "behavior": "periodic", "period": 7, "duty_cycle": 2, "position": (0.0, 1.0), "power_dbm": 18.0},
            {"name": "S10_E2_NearR2", "band": 5, "behavior": "periodic", "period": 11, "duty_cycle": 2, "position": (10.0, 1.0), "power_dbm": 18.0},
            {"name": "S10_E3_NearR3", "band": 8, "behavior": "intermittent", "activity_probability": 0.35, "position": (1.0, 10.0), "power_dbm": 18.0}
        ],
        "receivers": [
            {"id": "R1", "position": (0.0, 0.0), "snr_threshold": 15.0},
            {"id": "R2", "position": (10.0, 0.0), "snr_threshold": 15.0},
            {"id": "R3", "position": (0.0, 10.0), "snr_threshold": 15.0}
        ]
    }

    return scenarios
