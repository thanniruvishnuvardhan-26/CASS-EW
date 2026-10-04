import numpy as np
from simulator.environment import RFEnvironment
from simulator.receiver import VirtualReceiver

class SpatialRFEnvironment(RFEnvironment):
    """
    Environment that extends RFEnvironment with spatial coordinates for emitters.
    """
    def __init__(self, num_bands=10, seed=None):
        super().__init__(num_bands=num_bands, seed=seed)
        self.emitter_positions = {}

    def add_emitter(self, emitter, position=(0.0, 0.0)):
        super().add_emitter(emitter)
        self.emitter_positions[emitter] = np.array(position)

    def get_emitter_position(self, emitter):
        return self.emitter_positions.get(emitter, np.array([0.0, 0.0]))

class ReceiverProxyEnvironment:
    """
    A proxy environment tailored to a specific receiver. 
    It intercepts the step() call from the VirtualReceiver, 
    evaluates the spatial attenuation, and determines if the signal is present.
    """
    def __init__(self, base_env, receiver_position, snr_threshold=20.0):
        self.base_env = base_env
        self.receiver_position = np.array(receiver_position, dtype=float)
        self.snr_threshold = snr_threshold

    @property
    def time(self):
        return self.base_env.time
        
    @property
    def emitters(self):
        return self.base_env.emitters

    def step(self):
        # Advance the base environment
        self.base_env.step()
        
        detectable_bands = set()
        for e in self.base_env.emitters:
            if e.active:
                pos = self.base_env.get_emitter_position(e)
                dist = np.linalg.norm(pos - self.receiver_position)
                # Transparent synthetic spatial model:
                # Monotonic attenuation based on distance
                snr = 100.0 / (1.0 + dist)
                
                if snr >= self.snr_threshold:
                    detectable_bands.add(e.band)
                    
        return detectable_bands

class MultiReceiverSystem:
    """
    Coordinates multiple VirtualReceivers, each with its own proxy environment
    so they experience different signal strengths based on their location.
    """
    def __init__(self, base_env, receiver_configs):
        """
        receiver_configs: list of dicts: {'id': any, 'position': (x,y), 'snr_threshold': float}
        """
        self.base_env = base_env
        self.receivers = {}
        self.proxy_envs = {}
        
        for cfg in receiver_configs:
            rid = cfg['id']
            pos = cfg['position']
            snr = cfg.get('snr_threshold', 20.0)
            
            # create virtual receiver
            vr = VirtualReceiver(num_bands=base_env.num_bands, seed=base_env.seed)
            self.receivers[rid] = vr
            
            # create proxy env for this receiver
            proxy = ReceiverProxyEnvironment(base_env, pos, snr_threshold=snr)
            self.proxy_envs[rid] = proxy
            
    def scan(self, receiver_id, band, dwell_time=1):
        if receiver_id not in self.receivers:
            raise ValueError(f"Unknown receiver ID: {receiver_id}")
            
        vr = self.receivers[receiver_id]
        proxy = self.proxy_envs[receiver_id]
        
        # This will call proxy.step(), which advances the base_env time
        return vr.scan(proxy, band, dwell_time=dwell_time)
        
    def get_receiver(self, receiver_id):
        return self.receivers.get(receiver_id)
        
    def get_all_receiver_ids(self):
        return list(self.receivers.keys())
