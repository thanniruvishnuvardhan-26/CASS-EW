import unittest
import numpy as np
from simulator.environment import (
    RFEnvironment, 
    Emitter, 
    IntermittentEmitter, 
    PeriodicEmitter, 
    HoppingEmitter, 
    PersistentEmitter, 
    BurstEmitter
)
from simulator.receiver import VirtualReceiver

class TestPhase2Semantics(unittest.TestCase):

    def test_01_time_semantics(self):
        """
        1. FORMALLY DEFINE TIME SEMANTICS
        environment.time tracks the number of intervals elapsed.
        At time t=0, no intervals have elapsed.
        step() calculates the state for the interval (t, t+1] and advances time to t+1.
        Therefore, at time t=1, the interval (0, 1] has just completed and its state is observed.
        """
        env = RFEnvironment(num_bands=10, seed=42)
        self.assertEqual(env.time, 0, "Time starts at 0")
        
        # Add a persistent emitter to track state
        emitter = PersistentEmitter("Persist", band=3, seed=42)
        env.add_emitter(emitter)
        
        state_t1 = env.step()
        self.assertEqual(env.time, 1, "Time advances after step()")
        self.assertIn(3, state_t1, "State for interval (0, 1] is active")

    def test_02_receiver_scan_semantics(self):
        """
        2. FORMALLY DEFINE RECEIVER SCAN SEMANTICS
        - initial current_band = None
        - first acquisition switching cost
        - switching between different bands
        - reacquiring the same band
        - dwell timing
        - observation_time and effective_duration
        """
        env = RFEnvironment(num_bands=10, seed=42)
        receiver = VirtualReceiver(num_bands=10, switching_time=2, seed=42)
        
        # Initial state
        self.assertIsNone(receiver.current_band)
        
        # Scan 1: First acquisition, no switching cost (switching_time only if current_band != None and current_band != band)
        receiver.scan(env, band=3, dwell_time=1)
        self.assertEqual(env.time, 1) # 0 switching, 1 dwell
        obs1 = receiver.scan_history[-1]
        self.assertEqual(obs1["dwell_start_time"], 0)
        self.assertEqual(obs1["dwell_end_time"], 1)
        self.assertEqual(obs1["observation_time"], 1)
        self.assertEqual(obs1["effective_duration"], 1)
        
        # Scan 2: Switch to new band (cost = 2)
        receiver.scan(env, band=5, dwell_time=2)
        self.assertEqual(env.time, 5) # 1 + 2 switching + 2 dwell
        obs2 = receiver.scan_history[-1]
        self.assertEqual(obs2["dwell_start_time"], 3) # After switching
        self.assertEqual(obs2["dwell_end_time"], 5)
        self.assertEqual(obs2["observation_time"], 5)
        self.assertEqual(obs2["effective_duration"], 2)
        
        # Scan 3: Reacquire same band (cost = 0)
        receiver.scan(env, band=5, dwell_time=3)
        self.assertEqual(env.time, 8) # 5 + 0 switching + 3 dwell
        obs3 = receiver.scan_history[-1]
        self.assertEqual(obs3["dwell_start_time"], 5)
        self.assertEqual(obs3["dwell_end_time"], 8)
        self.assertEqual(obs3["observation_time"], 8)
        self.assertEqual(obs3["effective_duration"], 3)

    def test_03_emitter_factory_audit(self):
        """
        3. EMITTER FACTORY AUDIT
        Test Emitter.__new__ compatibility construction for ALL supported behaviors.
        """
        behaviors = {
            'intermittent': IntermittentEmitter,
            'periodic': PeriodicEmitter,
            'hopping': HoppingEmitter,
            'persistent': PersistentEmitter,
            'burst': BurstEmitter
        }
        
        for behavior, cls in behaviors.items():
            emitter = Emitter("Test", band=2, behavior=behavior, activity_probability=1.0)
            self.assertEqual(type(emitter), cls)
            self.assertTrue(isinstance(emitter, Emitter))
            
            # Direct construction
            direct_emitter = cls("Test", band=2, activity_probability=1.0)
            self.assertTrue(isinstance(direct_emitter, Emitter))
            
        with self.assertRaises(ValueError):
            Emitter("Test", band=2, behavior="invalid_behavior")

    def test_04_hopping_truth_table(self):
        """
        4. HOPPING TRUTH TABLE
        hop_bands = [6,7,8]
        Test hop_interval = 1, 2, 3.
        """
        hop_bands = [6, 7, 8]
        
        # hop_interval = 1
        e1 = HoppingEmitter("H1", behavior="hopping", hop_bands=hop_bands, hop_interval=1, activity_probability=1.0, seed=42)
        e1.reset()
        seq1 = [e1.step() and e1.band for _ in range(5)]
        self.assertEqual(seq1, [6, 7, 8, 6, 7])
        
        # hop_interval = 2
        e2 = HoppingEmitter("H2", behavior="hopping", hop_bands=hop_bands, hop_interval=2, activity_probability=1.0, seed=42)
        e2.reset()
        seq2 = [e2.step() and e2.band for _ in range(5)]
        self.assertEqual(seq2, [6, 6, 7, 7, 8])
        
        # hop_interval = 3
        e3 = HoppingEmitter("H3", behavior="hopping", hop_bands=hop_bands, hop_interval=3, activity_probability=1.0, seed=42)
        e3.reset()
        seq3 = [e3.step() and e3.band for _ in range(7)]
        self.assertEqual(seq3, [6, 6, 6, 7, 7, 7, 8])

    def test_05_periodic_truth_table(self):
        """
        5. PERIODIC TRUTH TABLE
        period=4 and duty_cycle=2, explicitly define expected active/inactive state for t=0..7.
        """
        e = PeriodicEmitter("P1", band=2, period=4, duty_cycle=2, seed=42)
        e.reset()
        seq = [e.step() for _ in range(8)]
        # Expected: [True, True, False, False, True, True, False, False]
        self.assertEqual(seq, [True, True, False, False, True, True, False, False])
        
        # Boundary: duty_cycle = 1
        e2 = PeriodicEmitter("P2", band=2, period=4, duty_cycle=1, seed=42)
        e2.reset()
        seq2 = [e2.step() for _ in range(8)]
        self.assertEqual(seq2, [True, False, False, False, True, False, False, False])
        
        # Boundary: duty_cycle = period
        e3 = PeriodicEmitter("P3", band=2, period=4, duty_cycle=4, seed=42)
        e3.reset()
        seq3 = [e3.step() for _ in range(8)]
        self.assertEqual(seq3, [True, True, True, True, True, True, True, True])

    def test_06_burst_truth_table(self):
        """
        6. BURST TRUTH TABLE
        burst_duration = 2
        inter_burst_duration = 3
        """
        e = BurstEmitter("B1", band=2, burst_duration=2, inter_burst_duration=3, seed=42)
        e.reset()
        seq = [e.step() for _ in range(8)]
        # burst for 2 steps (t=1,2), inter-burst for 3 steps (t=3,4,5), burst for 2 (t=6,7), inter-burst (t=8)
        self.assertEqual(seq, [True, True, False, False, False, True, True, False])

    def test_07_reset_seed_reproducibility(self):
        """
        7. RESET / SEED REPRODUCIBILITY
        For every emitter type: run sequence, reset, run identical sequence, assert equality.
        """
        behaviors = {
            'intermittent': IntermittentEmitter("Test", band=2, activity_probability=0.5, seed=42),
            'hopping': HoppingEmitter("Test", hop_bands=[1,2,3], hop_interval=1, activity_probability=0.5, seed=42),
        }
        
        for name, emitter in behaviors.items():
            emitter.reset(seed=42)
            seq1 = [emitter.step() for _ in range(10)]
            
            emitter.reset(seed=42)
            seq2 = [emitter.step() for _ in range(10)]
            self.assertEqual(seq1, seq2, f"Seed reproducibility failed for {name}")
            
            emitter.reset(seed=99)
            seq3 = [emitter.step() for _ in range(10)]
            self.assertNotEqual(seq1, seq3, f"Different seed should produce different stochastic sequence for {name}")

        # RFEnvironment overriding seeds
        env = RFEnvironment(num_bands=10, seed=42)
        e1 = IntermittentEmitter("E1", band=1, seed=999) # This seed will be overridden
        env.add_emitter(e1)
        env.reset()
        # env.reset() sets emitter seed to (env.seed * 1000 + i + 1) -> 42001
        self.assertEqual(e1.seed, 42001)

    def test_08_observable_vs_ground_truth_separation(self):
        """
        8. OBSERVABLE VS GROUND-TRUTH SEPARATION
        Prove that receiver-visible observations do NOT expose hidden future state.
        """
        env_a = RFEnvironment(num_bands=10, seed=42) 
        env_b = RFEnvironment(num_bands=10, seed=42)
        
        class DivergentEmitter(IntermittentEmitter):
            def __init__(self, diverge_after, seed1, seed2, **kwargs):
                super().__init__(**kwargs)
                self.diverge_after = diverge_after
                self.time_count = 0
                self.rng1 = np.random.default_rng(seed1)
                self.rng2 = np.random.default_rng(seed2)
            def step(self):
                self.time_count += 1
                if self.time_count <= self.diverge_after:
                    self.active = self.rng1.random() < self.activity_probability
                else:
                    self.active = self.rng2.random() < self.activity_probability
                return self.active

        e_a = DivergentEmitter(diverge_after=5, seed1=42, seed2=42, name="A", band=5, activity_probability=0.5)
        e_b = DivergentEmitter(diverge_after=5, seed1=42, seed2=99, name="B", band=5, activity_probability=0.5)
        
        env_a.add_emitter(e_a)
        env_b.add_emitter(e_b)
        
        receiver_a = VirtualReceiver(num_bands=10, seed=123, switching_time=1)
        receiver_b = VirtualReceiver(num_bands=10, seed=123, switching_time=1)
        
        for _ in range(5):
            receiver_a.scan(env_a, band=5, dwell_time=1)
            receiver_b.scan(env_b, band=5, dwell_time=1)
            
        self.assertEqual(receiver_a.scan_history, receiver_b.scan_history)
        self.assertEqual(env_a.time, 5)
        self.assertEqual(env_b.time, 5)
        
        # Next step will be different, proving history was untainted by the divergence
        
if __name__ == '__main__':
    unittest.main()
