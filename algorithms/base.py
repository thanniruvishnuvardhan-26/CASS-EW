from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseScheduler(ABC):
    """
    Formal Phase 3 Scheduler Abstraction.
    The scheduler only receives observable information.
    """
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the scheduler state, optionally seeding the RNG."""
        pass

    @abstractmethod
    def select_action(self, observation: Optional[Any] = None) -> int:
        """
        Select the next band to observe.
        Must operate ONLY on observable information.
        """
        pass

    @abstractmethod
    def update(self, observation: Optional[Any], action: int, result: bool) -> None:
        """
        Update the internal belief based on the observation result.
        
        Args:
            observation: The context/state before the action was taken.
            action: The band index chosen.
            result: Whether a signal was detected (True) or not (False).
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return the current internal state/belief of the scheduler."""
        pass
