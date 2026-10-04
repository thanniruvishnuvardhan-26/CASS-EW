"""
TSRD Scheduler Bridge
CASS-EW SIH Problem Statement 26055

Connects the FROZEN-CORE CASS-EW SpatialScheduler (or baseline schedulers)
to the DatasetRFEnvironment and DatasetVirtualReceiver without modifying frozen code.

Features:
- Adapts between physical frequency BandMap and discrete scheduler bands.
- Translates observation timestamps into discrete simulation time steps.
- Exposes structured next-scan decision:
    {
        "receiver": "R1",
        "selected_band": 3,
        "frequency_start_mhz": 5400.0,
        "frequency_end_mhz": 7200.0,
        "dwell_seconds": 0.05,
        "explanation": {...}
    }
- Updates scheduler state cleanly with zero label leakage.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
from data.band_map import BandMap
from simulator.dataset_rf_environment import DatasetObservation, DatasetVirtualReceiver
from algorithms.phase8_spatial_scheduler import SpatialScheduler


class TSRDSchedulerBridge:
    """
    Bridge connecting any CASS-EW scheduler to TSRD virtual receiver.
    """

    def __init__(
        self,
        scheduler: Any,
        band_map: BandMap,
        base_dwell_s: float = 0.01,  # 10ms nominal dwell
        receiver_id: str = "RX_0"
    ):
        self.scheduler = scheduler
        self.band_map = band_map
        self.base_dwell_s = float(base_dwell_s)
        self.receiver_id = receiver_id
        self.step_idx = 0

    def get_next_decision(self) -> Dict[str, Any]:
        """
        Queries scheduler for next scan action and packages structured decision contract.
        """
        if hasattr(self.scheduler, 'choose_action'):
            action = self.scheduler.choose_action(obs_time=self.step_idx)
        elif hasattr(self.scheduler, 'select_action'):
            action = self.scheduler.select_action(observation={"time": self.step_idx})
        elif hasattr(self.scheduler, 'get_action'):
            action = self.scheduler.get_action()
        else:
            raise AttributeError(f"Scheduler {type(self.scheduler)} has no action selection method")

        # Handle tuple action (receiver, band) or scalar band
        if isinstance(action, tuple) and len(action) == 2:
            rx_id, band = action
        else:
            rx_id = self.receiver_id
            band = int(action)

        band = int(band)
        freq_low, freq_high = self.band_map.band_to_range(band)

        # Allow adaptive dwell if scheduler explanation suggests it, else base_dwell_s
        dwell_s = self.base_dwell_s
        explanation_dict = {}
        if hasattr(self.scheduler, 'last_explanation') and self.scheduler.last_explanation is not None:
            expl = self.scheduler.last_explanation
            explanation_dict = expl.to_dict() if hasattr(expl, 'to_dict') else str(expl)
            if hasattr(expl, 'dwell') and expl.dwell > 1:
                dwell_s = self.base_dwell_s * expl.dwell

        return {
            "receiver": str(rx_id),
            "selected_band": band,
            "frequency_start_mhz": freq_low,
            "frequency_end_mhz": freq_high,
            "dwell_seconds": dwell_s,
            "explanation": explanation_dict
        }

    def update_observation(self, obs: DatasetObservation, decision: Dict[str, Any]):
        """
        Feeds receiver observation back to the scheduler.
        Zero leakage: passes strictly observable fields.
        """
        self.step_idx += 1
        rx_id = decision["receiver"]
        band = decision["selected_band"]

        # Adapt observation into standard dict format expected by scheduler
        obs_dict = {
            "receiver_id": rx_id,
            "band": band,
            "observation_time": self.step_idx,
            "dwell_end_time": self.step_idx,
            "detected": obs.detected,
            "signal_strength": obs.signal_strength,
            "dwell_duration": obs.effective_duration_s,
            "num_pulses": obs.num_pulses_intercepted
        }

        # Update scheduler
        if hasattr(self.scheduler, 'update'):
            from algorithms.phase8_spatial_scheduler import SpatialScheduler
            if isinstance(self.scheduler, SpatialScheduler):
                self.scheduler.update(obs_dict, (rx_id, band), obs.detected)
            else:
                # Single-receiver or baseline scheduler expecting integer band
                try:
                    self.scheduler.update(obs_dict, int(band), obs.detected)
                except TypeError:
                    self.scheduler.update(obs_dict, (rx_id, band), obs.detected)
