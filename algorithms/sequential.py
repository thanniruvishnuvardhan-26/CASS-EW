from typing import Any, Dict, Optional
from algorithms.base import BaseScheduler

class SequentialScheduler(BaseScheduler):
    """
    Phase 3 compliant Sequential Scheduler.
    Selects bands in a sequential round-robin fashion.
    """
    
    def __init__(self, num_bands: int = 10):
        self.num_bands = num_bands
        self.current_band = 0

    def reset(self, seed: Optional[int] = None) -> None:
        self.current_band = 0

    def select_action(self, observation: Optional[Any] = None) -> int:
        band = self.current_band
        self.current_band = (self.current_band + 1) % self.num_bands
        return band

    def update(self, observation: Optional[Any], action: int, result: bool) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        return {"current_band": self.current_band}
