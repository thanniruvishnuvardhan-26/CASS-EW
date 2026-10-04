"""
Time Normalization Layer for CASS-EW
CASS-EW SIH Problem Statement 26055

CRITICAL ARCHITECTURAL RULE:
NEVER mix physical microseconds (TSRD ToA) with discrete simulation steps
or floating-point seconds.
This module enforces an explicit, deterministic bi-directional conversion.
"""

from typing import Union
import numpy as np


class TimeNormalizer:
    """
    Normalizes time across external dataset units (microseconds)
    and internal CASS-EW simulation units (seconds).
    """

    US_PER_SECOND = 1_000_000.0

    @classmethod
    def us_to_seconds(cls, time_us: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
        """Converts microseconds (us) to seconds (s)."""
        return time_us / cls.US_PER_SECOND

    @classmethod
    def seconds_to_us(cls, time_s: Union[float, int, np.ndarray]) -> Union[float, np.ndarray]:
        """Converts seconds (s) to microseconds (us)."""
        return time_s * cls.US_PER_SECOND

    @classmethod
    def normalize_dwell(cls, dwell_s: float) -> float:
        """Validates and ensures dwell duration is in positive seconds."""
        dwell = float(dwell_s)
        if dwell <= 0:
            raise ValueError(f"Dwell duration must be strictly positive, got {dwell}")
        return dwell
