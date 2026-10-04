import numpy as np
import math
from typing import Callable, Optional
from data.emitter_truth import EmitterTruthProvider, TruthPulse

def calculate_fspl(tx_x: float, tx_y: float, rx_x: float, rx_y: float, freq_mhz: float) -> float:
    """
    REPLAY ASSUMPTION: Free Space Path Loss (FSPL).
    """
    dist_km = math.hypot(tx_x - rx_x, tx_y - rx_y)
    dist_km = max(dist_km, 0.001)  # avoid log(0)
    fspl_db = 20 * math.log10(dist_km) + 20 * math.log10(freq_mhz) + 32.44
    return fspl_db

class CausalReplayEnvironment:
    def __init__(self, truth_provider: EmitterTruthProvider, step_duration_us: float, num_bands: int = 10, band_mapping: Optional[Callable[[float], Optional[int]]] = None, seed: Optional[int] = None):
        self.truth_provider = truth_provider
        self.step_duration_us = step_duration_us
        self.num_bands = num_bands
        self.time = 0
        self.seed = seed
        self.rng = np.random.default_rng(self.seed) if self.seed is not None else np.random.default_rng()
        
        # Get receiver metadata for noise scales
        self.rx_config = self.truth_provider.transmitters[0].rx_config if self.truth_provider.transmitters else {}
        self.freq_noise_scale = float(self.rx_config.get('freq_noise_scale_mhz', 0.0))
        self.pw_noise_scale = float(self.rx_config.get('pw_noise_scale_us', 0.0))
        self.sensitivity_dbm = float(self.rx_config.get('sensitivity_dbm', -110.0))
        self.rx_gain_db = float(self.rx_config.get('gain_db', 0.0))
        
        self.band_mapping = band_mapping if band_mapping is not None else self._default_band_mapping

    @property
    def time_us(self) -> float:
        return self.time * self.step_duration_us
            
    def _default_band_mapping(self, freq_mhz: float) -> Optional[int]:
        """
        Explicit mapping layer: Maps frequency to CASS-EW band.
        Assumes 10 bands spread linearly from 0 to 18000 MHz by default,
        but can be overridden.
        """
        if freq_mhz < 0:
            return None
        band = int(freq_mhz / 1800.0)
        return min(max(band, 0), self.num_bands - 1)
        
    def reset(self, seed: Optional[int] = None):
        self.time = 0
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(self.seed)
        self.truth_provider.reset()

    def step(self):
        """
        Advances the environment by one step_duration_us.
        Returns a set of active bands where a detected pulse fell.
        """
        start_time_us = self.time * self.step_duration_us
        self.time += 1
        end_time_us = self.time * self.step_duration_us
        
        # 1. Ask the emitter-truth provider for events occurring within the requested dwell interval.
        pulses = self.truth_provider.get_ground_truth_pulses(start_time_us, end_time_us)
        
        active_bands = set()
        for p in pulses:
            # Only consider pulses that physically paint the receiver
            if not p.is_active:
                continue
                
            # REPLAY ASSUMPTION: AWGN Noise on frequency
            obs_freq = p.frequency_mhz
            if self.freq_noise_scale > 0:
                obs_freq += self.rng.normal(0, self.freq_noise_scale)
            
            # Amplitude logic
            fspl_db = calculate_fspl(p.tx_x_km, p.tx_y_km, p.rx_x_km, p.rx_y_km, p.frequency_mhz)
            tx_power_dbm = 10 * math.log10(p.power_w * 1000)
            
            # Get specific tx gain
            tx_gain = 0.0
            for tx in self.truth_provider.transmitters:
                if tx.tx_id == p.transmitter_id:
                    tx_gain = tx.gain
                    break
                    
            tx_eirp_dbm = tx_power_dbm + tx_gain
            obs_amp = tx_eirp_dbm - fspl_db + self.rx_gain_db
            
            # 2/3. Apply receiver detection semantics
            if obs_amp >= self.sensitivity_dbm:
                band = self.band_mapping(obs_freq)
                if band is not None and 0 <= band < self.num_bands:
                    active_bands.add(band)
                    
        return active_bands
