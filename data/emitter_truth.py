import h5py
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class TruthPulse:
    timestamp_us: float
    transmitter_id: str
    frequency_mhz: float
    pulse_width_us: float
    power_w: float
    tx_x_km: float
    tx_y_km: float
    rx_x_km: float
    rx_y_km: float
    is_active: bool

class TransmitterModel:
    def __init__(self, tx_id: str, config: Dict[str, Any], rx_config: Dict[str, Any]):
        self.tx_id = tx_id
        self.config = config
        self.rx_config = rx_config
        
        # Parse frequency
        self.freqs_mhz = config['frequency_config']['freqs_mhz']
        self.freq_mode = config['frequency_config'].get('freq_mode', b'FixedSingle').decode('utf-8') if isinstance(config['frequency_config'].get('freq_mode', b'FixedSingle'), bytes) else config['frequency_config'].get('freq_mode', 'FixedSingle')
        
        # Parse position
        self.start_x, self.start_y = config['position_config']['start_position_km']
        self.speed_km_s = config['position_config'].get('speed_km_s', 0.0)
        self.travel_angle_deg = config['position_config'].get('travel_angle_deg', 0.0)
        
        # Parse power
        self.power_w = config['power_config'].get('power_w', 100.0)
        self.gain = config['power_config'].get('gain', 0.0)
        
        # Parse PRI
        self.pris_us = config['pri_config']['pris_us']
        self.pri_mode = config['pri_config'].get('pri_mode', b'Fixed').decode('utf-8') if isinstance(config['pri_config'].get('pri_mode', b'Fixed'), bytes) else config['pri_config'].get('pri_mode', 'Fixed')
        
        # Parse Pulse Width
        self.pws_us = config['pulse_width_config']['pws_us']
        self.pw_mode = config['pulse_width_config'].get('pw_mode', b'Fixed').decode('utf-8') if isinstance(config['pulse_width_config'].get('pw_mode', b'Fixed'), bytes) else config['pulse_width_config'].get('pw_mode', 'Fixed')
        
        # Parse Scan
        self.scan_rate_rpm = config['scan_config'].get('scan_rate_rpm', 0.0)
        self.beam_width_deg = config['scan_config'].get('beam_width_deg', 360.0)
        self.scan_start_angle = config['scan_config'].get('scan_start_angle', 0.0)
        
        self.rx_x, self.rx_y = 0.0, 0.0
        if 'position' in rx_config:
            self.rx_x, self.rx_y = rx_config['position'][:2]
            
        self.pulse_idx = 0
        self.current_time_us = 0.0
        
    def reset(self):
        self.pulse_idx = 0
        self.current_time_us = 0.0
        
    def generate_pulses(self, end_time_us: float) -> List[TruthPulse]:
        pulses = []
        if len(self.pris_us) == 0 or all(p <= 0 for p in self.pris_us):
            return pulses
            
        if self.current_time_us >= end_time_us:
            return pulses
            
        # Fast path using numpy
        if self.pri_mode == 'Fixed':
            pri = self.pris_us[0]
            num_pulses = int((end_time_us - self.current_time_us) / pri) + 1
        else:
            # Staggered
            avg_pri = sum(self.pris_us) / len(self.pris_us)
            num_pulses = int((end_time_us - self.current_time_us) / avg_pri) + 2
            
        if num_pulses <= 0:
            return pulses
            
        # Vectorized pulse times
        if self.pri_mode == 'Fixed':
            times_us = self.current_time_us + np.arange(num_pulses) * pri
        else:
            pris = np.array(self.pris_us)
            cycles = num_pulses // len(pris) + 1
            full_pris = np.tile(pris, cycles)
            cum_pris = np.cumsum(np.insert(full_pris, 0, 0))
            times_us = self.current_time_us + cum_pris[:num_pulses]
            
        # Filter times before end_time_us
        valid = times_us < end_time_us
        times_us = times_us[valid]
        num_pulses = len(times_us)
        
        if num_pulses == 0:
            return pulses
            
        # Vectorized frequencies
        if self.freq_mode == 'FixedSingle':
            freqs = np.full(num_pulses, self.freqs_mhz[0])
        elif self.freq_mode == 'HoppingSawtooth':
            idx = (self.pulse_idx + np.arange(num_pulses)) % len(self.freqs_mhz)
            freqs = np.array(self.freqs_mhz)[idx]
        else:
            freqs = np.full(num_pulses, self.freqs_mhz[0])
            
        # Vectorized pulse widths
        if self.pw_mode == 'Fixed':
            pws = np.full(num_pulses, self.pws_us[0])
        elif self.pw_mode == 'Seq-Fixed':
            idx = (self.pulse_idx + np.arange(num_pulses)) % len(self.pws_us)
            pws = np.array(self.pws_us)[idx]
        else:
            pws = np.full(num_pulses, self.pws_us[0])
            
        # Vectorized positions
        t_s = times_us / 1e6
        rad = math.radians(self.travel_angle_deg)
        tx_x = self.start_x + self.speed_km_s * t_s * math.cos(rad)
        tx_y = self.start_y + self.speed_km_s * t_s * math.sin(rad)
        
        # Vectorized activity (beam)
        if self.beam_width_deg < 360.0:
            current_pointing_angle = (self.scan_start_angle + (self.scan_rate_rpm * 360.0 / 60.0) * t_s) % 360.0
            angle_to_rx = np.degrees(np.arctan2(self.rx_y - tx_y, self.rx_x - tx_x)) % 360.0
            diff = (current_pointing_angle - angle_to_rx + 180) % 360 - 180
            is_active = np.abs(diff) <= (self.beam_width_deg / 2.0)
        else:
            is_active = np.ones(num_pulses, dtype=bool)
            
        # FILTER ONLY ACTIVE PULSES BEFORE OBJECT ALLOCATION
        active_indices = np.nonzero(is_active)[0]
        
        # Construct objects
        for i in active_indices:
            pulses.append(TruthPulse(
                timestamp_us=float(times_us[i]),
                transmitter_id=self.tx_id,
                frequency_mhz=float(freqs[i]),
                pulse_width_us=float(pws[i]),
                power_w=self.power_w,
                tx_x_km=float(tx_x[i]),
                tx_y_km=float(tx_y[i]),
                rx_x_km=self.rx_x,
                rx_y_km=self.rx_y,
                is_active=True
            ))
            
        self.pulse_idx += num_pulses
        if self.pri_mode == 'Fixed':
            self.current_time_us = self.current_time_us + num_pulses * self.pris_us[0]
        else:
            self.current_time_us = times_us[-1] + self.pris_us[(self.pulse_idx - 1) % len(self.pris_us)]
            
        return pulses

class EmitterTruthProvider:
    def __init__(self, h5_file_path: str):
        self.h5_file_path = h5_file_path
        self.transmitters = []
        self._load_metadata()
        
    def _load_metadata(self):
        with h5py.File(self.h5_file_path, 'r') as f:
            rx_group = f['/metadata/receiver']
            rx_config = {}
            for k, v in rx_group.attrs.items():
                rx_config[k] = v.decode('utf-8') if isinstance(v, bytes) else v
            if 'position' in rx_group:
                rx_config['position'] = rx_group['position'][()]
                
            tx_group = f['/metadata/transmitters']
            for tx_id in tx_group.keys():
                tx_data = tx_group[tx_id]
                config = {}
                for key in tx_data.keys():
                    subgroup = tx_data[key]
                    config[key] = {}
                    for k, v in subgroup.attrs.items():
                        config[key][k] = v.decode('utf-8') if isinstance(v, bytes) else v
                    for ds_key in subgroup.keys():
                        config[key][ds_key] = subgroup[ds_key][()]
                config['attrs'] = {}
                for k, v in tx_data.attrs.items():
                    config['attrs'][k] = v.decode('utf-8') if isinstance(v, bytes) else v
                self.transmitters.append(TransmitterModel(tx_id, config, rx_config))
                
    def reset(self):
        for tx in self.transmitters:
            tx.reset()
            
    def get_ground_truth_pulses(self, start_time_us: float, end_time_us: float) -> List[TruthPulse]:
        all_pulses = []
        for tx in self.transmitters:
            pulses = tx.generate_pulses(end_time_us)
            for p in pulses:
                if p.timestamp_us >= start_time_us:
                    all_pulses.append(p)
        return sorted(all_pulses, key=lambda p: p.timestamp_us)
