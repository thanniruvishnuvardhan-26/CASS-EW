"""
Physical BandMap for CASS-EW
CASS-EW SIH Problem Statement 26055

CRITICAL ARCHITECTURAL RULE:
NEVER map frequency to band using `frequency % num_bands`.
Production frequency mapping must be physically contiguous, deterministic,
and bounded by explicit lower and upper frequency boundaries (MHz).
"""

from typing import List, Tuple, Optional, Dict, Any


class BandMap:
    """
    Deterministic physical frequency-to-band mapper.
    
    Each band i is defined by a contiguous semi-open interval:
        [low_mhz, high_mhz)
    with the final band optionally inclusive of the upper boundary.
    """

    def __init__(
        self,
        boundaries: Optional[List[Tuple[float, float]]] = None,
        num_bands: int = 10,
        min_freq_mhz: float = 0.0,
        max_freq_mhz: float = 18000.0
    ):
        self.num_bands = num_bands
        self.min_freq_mhz = float(min_freq_mhz)
        self.max_freq_mhz = float(max_freq_mhz)

        if boundaries is not None:
            self.boundaries = [(float(b[0]), float(b[1])) for b in boundaries]
            self.num_bands = len(self.boundaries)
            self._validate_boundaries()
        else:
            # Linear uniform physical partition
            step = (self.max_freq_mhz - self.min_freq_mhz) / float(self.num_bands)
            self.boundaries = [
                (self.min_freq_mhz + i * step, self.min_freq_mhz + (i + 1) * step)
                for i in range(self.num_bands)
            ]

    def _validate_boundaries(self):
        for i, (low, high) in enumerate(self.boundaries):
            if low >= high:
                raise ValueError(f"Invalid band {i}: lower bound {low} >= upper bound {high}")
            if i > 0:
                prev_high = self.boundaries[i - 1][1]
                if abs(low - prev_high) > 1e-4:
                    raise ValueError(f"Band boundary discontinuity between band {i-1} ({prev_high}) and {i} ({low})")

    def frequency_to_band(self, freq_mhz: float) -> Optional[int]:
        """
        Maps a physical frequency in MHz to its discrete band index [0, num_bands - 1].
        Returns None if out of coverage.
        """
        if freq_mhz < self.boundaries[0][0] or freq_mhz > self.boundaries[-1][1]:
            return None

        # Check intervals: [low, high)
        for i, (low, high) in enumerate(self.boundaries):
            if i == self.num_bands - 1:
                # Include upper edge for final band
                if low <= freq_mhz <= high:
                    return i
            else:
                if low <= freq_mhz < high:
                    return i
        return None

    def band_to_range(self, band_idx: int) -> Tuple[float, float]:
        """
        Returns the (low_mhz, high_mhz) physical bounds for the given band index.
        """
        if not (0 <= band_idx < self.num_bands):
            raise IndexError(f"Band index {band_idx} out of range [0, {self.num_bands - 1}]")
        return self.boundaries[band_idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_bands": self.num_bands,
            "min_freq_mhz": self.boundaries[0][0],
            "max_freq_mhz": self.boundaries[-1][1],
            "bands": [
                {"band": i, "low_mhz": b[0], "high_mhz": b[1]}
                for i, b in enumerate(self.boundaries)
            ]
        }

    @classmethod
    def from_tsrd_receiver(cls, receiver_meta: Dict[str, Any], default_num_bands: int = 10) -> "BandMap":
        """
        Builds a BandMap aligned with TSRD receiver metadata.
        If freq_range_mhz is provided (e.g. [500.0, 18000.0]), uses those limits.
        """
        freq_range = receiver_meta.get('freq_range_mhz')
        if freq_range is not None and len(freq_range) == 2:
            min_f = float(freq_range[0])
            max_f = float(freq_range[1])
        else:
            min_f = 500.0
            max_f = 18000.0

        return cls(num_bands=default_num_bands, min_freq_mhz=min_f, max_freq_mhz=max_f)
