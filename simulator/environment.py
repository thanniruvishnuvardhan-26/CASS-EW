import numpy as np

class EmitterConfig:
    def __init__(self, name, band, seed=None):
        self.name = name
        self.band = band
        self.seed = seed

class Emitter:
    def __new__(cls, name=None, band=None, behavior=None, *args, **kwargs):
        if behavior == 'intermittent':
            return object.__new__(IntermittentEmitter)
        elif behavior == 'periodic':
            return object.__new__(PeriodicEmitter)
        elif behavior == 'hopping':
            return object.__new__(HoppingEmitter)
        elif behavior == 'random_hopping':
            return object.__new__(RandomHoppingEmitter)
        elif behavior == 'persistent':
            return object.__new__(PersistentEmitter)
        elif behavior == 'burst':
            return object.__new__(BurstEmitter)
        if behavior is not None:
            raise ValueError(f"Unknown emitter behavior: {behavior}")
        return object.__new__(cls)

    def __init__(self, config=None, name=None, band=None, behavior=None, activity_probability=None, period=None, duty_cycle=None, hop_bands=None, hop_interval=None, seed=None, **kwargs):
        if isinstance(config, str):
            name = config
            config = None
        if config is None and name is not None:
            config = EmitterConfig(name, band, seed)
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.name = config.name
        self.initial_band = config.band
        self.band = config.band
        self.active = False
        self.seed = config.seed
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random.default_rng()
        self._initialized = True
            
    def reset(self, seed=None):
        self.band = self.initial_band
        self.active = False
        if seed is not None:
            self.seed = seed
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
            
    def step(self):
        raise NotImplementedError('Subclasses must implement step()')

class IntermittentEmitter(Emitter):
    def __init__(self, config=None, name=None, band=None, behavior=None, activity_probability=0.3, seed=None, **kwargs):
        if isinstance(config, str):
            name = config
            config = None
        if config is None and name is not None:
            config = EmitterConfig(name, band, seed)
        super().__init__(config)
        self.activity_probability = activity_probability

    def step(self):
        rand_val = self.rng.random()
        self.active = (rand_val < self.activity_probability)
        return self.active

class PeriodicEmitter(Emitter):
    def __init__(self, config=None, name=None, band=None, behavior=None, period=2, duty_cycle=1, seed=None, **kwargs):
        if isinstance(config, str):
            name = config
            config = None
        if config is None and name is not None:
            config = EmitterConfig(name, band, seed)
        super().__init__(config)
        self.period = period
        self.duty_cycle = duty_cycle
        self.time_in_period = 0

    def reset(self, seed=None):
        super().reset(seed)
        self.time_in_period = 0

    def step(self):
        self.active = (self.time_in_period < self.duty_cycle)
        self.time_in_period = (self.time_in_period + 1) % self.period
        return self.active

class HoppingEmitter(Emitter):
    def __init__(self, config=None, name=None, band=None, behavior=None, hop_bands=None, hop_interval=1, activity_probability=1.0, seed=None, **kwargs):
        if isinstance(config, str):
            name = config
            config = None
        if config is None and name is not None:
            config = EmitterConfig(name, hop_bands[0] if hop_bands else 0, seed)
        super().__init__(config)
        self.hop_bands = list(hop_bands) if hop_bands else []
        self.hop_interval = hop_interval
        self.activity_probability = activity_probability
        self.hop_index = 0
        self.time_on_hop = 0

    def reset(self, seed=None):
        super().reset(seed)
        self.hop_index = 0
        self.time_on_hop = 0
        if self.hop_bands:
            self.band = self.hop_bands[0]

    def step(self):
        if self.time_on_hop >= self.hop_interval:
            self.time_on_hop = 0
            if self.hop_bands:
                self.hop_index = (self.hop_index + 1) % len(self.hop_bands)
                self.band = self.hop_bands[self.hop_index]
        self.time_on_hop += 1
        rand_val = self.rng.random()
        self.active = (rand_val < self.activity_probability)
        return self.active

class RandomHoppingEmitter(Emitter):
    def __init__(self, config=None, name=None, band=None, behavior=None, hop_bands=None, activity_probability=1.0, seed=None, **kwargs):
        if isinstance(config, str):
            name = config
            config = None
        if config is None and name is not None:
            config = EmitterConfig(name, hop_bands[0] if hop_bands else 0, seed)
        super().__init__(config)
        self.hop_bands = list(hop_bands) if hop_bands else []
        self.activity_probability = activity_probability

    def reset(self, seed=None):
        super().reset(seed)
        if self.hop_bands:
            self.band = self.hop_bands[0]

    def step(self):
        if self.hop_bands:
            self.band = self.rng.choice(self.hop_bands)
        
        rand_val = self.rng.random()
        self.active = (rand_val < self.activity_probability)
        return self.active

class PersistentEmitter(Emitter):
    def __init__(self, config=None, name=None, band=None, behavior=None, seed=None, **kwargs):
        if isinstance(config, str):
            name = config
            config = None
        if config is None and name is not None:
            config = EmitterConfig(name, band, seed)
        super().__init__(config)

    def step(self):
        self.active = True
        return self.active

class BurstEmitter(Emitter):
    def __init__(self, config=None, name=None, band=None, behavior=None, burst_duration=3, inter_burst_duration=5, seed=None, **kwargs):
        if isinstance(config, str):
            name = config
            config = None
        if config is None and name is not None:
            config = EmitterConfig(name, band, seed)
        super().__init__(config)
        self.burst_duration = burst_duration
        self.inter_burst_duration = inter_burst_duration
        self.time_in_state = 0
        self.is_bursting = True

    def reset(self, seed=None):
        super().reset(seed)
        self.time_in_state = 0
        self.is_bursting = True

    def step(self):
        self.active = self.is_bursting
        self.time_in_state += 1
        
        if self.is_bursting and self.time_in_state >= self.burst_duration:
            self.is_bursting = False
            self.time_in_state = 0
        elif not self.is_bursting and self.time_in_state >= self.inter_burst_duration:
            self.is_bursting = True
            self.time_in_state = 0
            
        return self.active

class RFEnvironment:
    def __init__(self, num_bands=10, seed=None):
        self.num_bands = num_bands
        self.emitters = []
        self.time = 0
        self.seed = seed
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
        else:
            self.rng = np.random.default_rng()

    def add_emitter(self, emitter):
        self.emitters.append(emitter)

    def reset(self, seed=None):
        self.time = 0
        if seed is not None:
            self.seed = seed
        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
            
        for i, emitter in enumerate(self.emitters):
            emitter_seed = None if self.seed is None else (self.seed * 1000 + i + 1)
            emitter.reset(seed=emitter_seed)

    def step(self):
        self.time += 1
        active_bands = set()

        for emitter in self.emitters:
            is_active = emitter.step()
            if is_active:
                active_bands.add(emitter.band)

        return active_bands

    def get_state(self):
        return {
            'time': self.time,
            'active_bands': [e.band for e in self.emitters if e.active]
        }
    
    def get_ground_truth(self):
        return self.get_state()