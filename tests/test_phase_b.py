import unittest
from web_app import app, sim_state

class TestPhaseB(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_scenario_switching(self):
        # Default scenario
        res = self.client.post('/api/reset', json={'seed': 42})
        self.assertEqual(res.status_code, 200)
        default_emitters = [e.name for e in sim_state['emitters']]
        
        # Periodic scenario
        res2 = self.client.post('/api/reset', json={'seed': 42, 'scenario': 'Periodic'})
        self.assertEqual(res2.status_code, 200)
        periodic_emitters = [e.name for e in sim_state['emitters']]
        
        self.assertNotEqual(default_emitters, periodic_emitters)
        self.assertIn('E1_Per', periodic_emitters)

    def test_algorithm_switching(self):
        # Default spatial
        res = self.client.post('/api/reset', json={'seed': 42, 'algorithm': 'Phase 8 Spatial Integrated'})
        scheduler_spatial = sim_state['scheduler']
        self.assertEqual(scheduler_spatial.__class__.__name__, 'SpatialScheduler')
        
        # Random algorithm adapter
        res2 = self.client.post('/api/reset', json={'seed': 42, 'algorithm': 'Random'})
        scheduler_random = sim_state['scheduler']
        self.assertEqual(scheduler_random.__class__.__name__, 'MultiReceiverAdapter')
        
        # Step and ensure it works
        step_res = self.client.post('/api/step', json={'steps': 1})
        self.assertEqual(step_res.status_code, 200)

    def test_reproducibility(self):
        self.client.post('/api/reset', json={'seed': 123, 'scenario': 'Deterministic Hopping', 'algorithm': 'Random'})
        s1 = self.client.post('/api/step', json={'steps': 1}).get_json()
        s2 = self.client.post('/api/step', json={'steps': 1}).get_json()
        
        self.client.post('/api/reset', json={'seed': 123, 'scenario': 'Deterministic Hopping', 'algorithm': 'Random'})
        s3 = self.client.post('/api/step', json={'steps': 1}).get_json()
        s4 = self.client.post('/api/step', json={'steps': 1}).get_json()
        
        self.assertEqual(s1['ground_truth']['active_bands'], s3['ground_truth']['active_bands'])
        self.assertEqual(s2['ground_truth']['active_bands'], s4['ground_truth']['active_bands'])

    def test_stationary(self):
        self.client.post('/api/reset', json={'seed': 42, 'scenario': 'Stationary / Fixed'})
        emitter = sim_state['emitters'][0]
        initial_band = emitter.band
        for _ in range(5):
            emitter.step()
            self.assertEqual(emitter.band, initial_band)
            self.assertTrue(emitter.active)

    def test_periodic(self):
        self.client.post('/api/reset', json={'seed': 42, 'scenario': 'Periodic'})
        emitter = sim_state['emitters'][0] # E1_Per, period=5, duty_cycle=2
        actives = [emitter.step() for _ in range(5)]
        self.assertEqual(actives, [True, True, False, False, False])
        
    def test_deterministic_hopping(self):
        self.client.post('/api/reset', json={'seed': 42, 'scenario': 'Deterministic Hopping'})
        emitter = sim_state['emitters'][0] # E1_Hop, bands [1,3,5], interval 5
        bands = []
        for _ in range(6):
            emitter.step()
            bands.append(emitter.band)
        self.assertEqual(bands, [1, 1, 1, 1, 1, 3])

    def test_frequency_agile(self):
        self.client.post('/api/reset', json={'seed': 42, 'scenario': 'Frequency Agile'})
        emitter = sim_state['emitters'][0] # bands [0,3,6,9], interval 3
        bands = []
        for _ in range(4):
            emitter.step()
            bands.append(emitter.band)
        self.assertEqual(bands, [0, 0, 0, 3])

    def test_random_hopping(self):
        self.client.post('/api/reset', json={'seed': 42, 'scenario': 'Random Hopping'})
        emitter = sim_state['emitters'][0] 
        bands = set()
        for _ in range(20):
            emitter.step()
            bands.add(emitter.band)
        self.assertTrue(len(bands) > 1, "Random hopping should change bands")

    def test_mixed_environment(self):
        self.client.post('/api/reset', json={'seed': 42, 'scenario': 'Mixed Environment'})
        classes = [e.__class__.__name__ for e in sim_state['emitters']]
        self.assertIn('PeriodicEmitter', classes)
        self.assertIn('IntermittentEmitter', classes)

if __name__ == '__main__':
    unittest.main()
