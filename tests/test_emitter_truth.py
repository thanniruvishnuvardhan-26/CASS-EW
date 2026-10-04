import unittest
import os
import tempfile
import h5py
import numpy as np

from data.emitter_truth import EmitterTruthProvider

class TestEmitterTruthProvider(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.valid_h5_path = os.path.join(self.temp_dir.name, "truth_fixture.h5")
        self._create_fixture(self.valid_h5_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_fixture(self, path):
        with h5py.File(path, 'w') as f:
            # Create minimum required metadata
            meta = f.create_group('/metadata')
            rx = meta.create_group('receiver')
            rx.attrs['position'] = np.array([0.0, 0.0, 0.0])
            rx.attrs['sensitivity_dbm'] = -110.0
            
            tx = meta.create_group('transmitters')
            
            # Tx 0: Fixed freq, fixed pri, fixed pw, always active
            tx0 = tx.create_group('transmitters_0')
            fc0 = tx0.create_group('frequency_config')
            fc0.create_dataset('freqs_mhz', data=np.array([1000.0]))
            fc0.attrs['freq_mode'] = 'FixedSingle'
            
            pc0 = tx0.create_group('position_config')
            pc0.create_dataset('start_position_km', data=np.array([10.0, 0.0]))
            pc0.attrs['speed_km_s'] = 0.0
            
            pwr0 = tx0.create_group('power_config')
            pwr0.attrs['power_w'] = 1000.0
            pwr0.attrs['gain'] = 30.0
            
            pri0 = tx0.create_group('pri_config')
            pri0.create_dataset('pris_us', data=np.array([1000.0]))
            pri0.attrs['pri_mode'] = 'Fixed'
            
            pw0 = tx0.create_group('pulse_width_config')
            pw0.create_dataset('pws_us', data=np.array([10.0]))
            pw0.attrs['pw_mode'] = 'Fixed'
            
            sc0 = tx0.create_group('scan_config')
            sc0.attrs['beam_width_deg'] = 360.0
            sc0.attrs['scan_rate_rpm'] = 0.0
            sc0.attrs['scan_start_angle'] = 0.0
            
            # Tx 1: Hopping freq
            tx1 = tx.create_group('transmitters_1')
            fc1 = tx1.create_group('frequency_config')
            fc1.create_dataset('freqs_mhz', data=np.array([2000.0, 3000.0]))
            fc1.attrs['freq_mode'] = 'HoppingSawtooth'
            
            pc1 = tx1.create_group('position_config')
            pc1.create_dataset('start_position_km', data=np.array([20.0, 0.0]))
            
            pwr1 = tx1.create_group('power_config')
            pwr1.attrs['power_w'] = 1000.0
            
            pri1 = tx1.create_group('pri_config')
            pri1.create_dataset('pris_us', data=np.array([500.0]))
            pri1.attrs['pri_mode'] = 'Fixed'
            
            pw1 = tx1.create_group('pulse_width_config')
            pw1.create_dataset('pws_us', data=np.array([5.0]))
            pw1.attrs['pw_mode'] = 'Fixed'
            
            sc1 = tx1.create_group('scan_config')
            sc1.attrs['beam_width_deg'] = 360.0

    def test_provider_initialization(self):
        provider = EmitterTruthProvider(self.valid_h5_path)
        self.assertEqual(len(provider.transmitters), 2)

    def test_deterministic_emitter_reconstruction(self):
        provider = EmitterTruthProvider(self.valid_h5_path)
        pulses = provider.get_ground_truth_pulses(0.0, 3000.0)
        
        # Tx0 (PRI=1000) should have pulses at 0, 1000, 2000
        # Tx1 (PRI=500) should have pulses at 0, 500, 1000, 1500, 2000, 2500
        self.assertEqual(len(pulses), 9)
        
        tx0_pulses = [p for p in pulses if p.transmitter_id == 'transmitters_0']
        self.assertEqual(len(tx0_pulses), 3)
        self.assertEqual(tx0_pulses[0].timestamp_us, 0.0)
        self.assertEqual(tx0_pulses[1].timestamp_us, 1000.0)
        self.assertEqual(tx0_pulses[2].timestamp_us, 2000.0)
        
        tx1_pulses = [p for p in pulses if p.transmitter_id == 'transmitters_1']
        self.assertEqual(len(tx1_pulses), 6)
        # Check hopping frequencies
        self.assertEqual(tx1_pulses[0].frequency_mhz, 2000.0)
        self.assertEqual(tx1_pulses[1].frequency_mhz, 3000.0)
        self.assertEqual(tx1_pulses[2].frequency_mhz, 2000.0)

    def test_sequential_access(self):
        provider = EmitterTruthProvider(self.valid_h5_path)
        pulses_1 = provider.get_ground_truth_pulses(0.0, 1000.0)
        # Tx0: 0, Tx1: 0, 500
        self.assertEqual(len(pulses_1), 3)
        
        pulses_2 = provider.get_ground_truth_pulses(1000.0, 2000.0)
        # Tx0: 1000, Tx1: 1000, 1500
        self.assertEqual(len(pulses_2), 3)

    def test_same_metadata_produces_same_truth(self):
        p1 = EmitterTruthProvider(self.valid_h5_path)
        p2 = EmitterTruthProvider(self.valid_h5_path)
        
        pulses1 = p1.get_ground_truth_pulses(0.0, 5000.0)
        pulses2 = p2.get_ground_truth_pulses(0.0, 5000.0)
        
        self.assertEqual(len(pulses1), len(pulses2))
        for p_a, p_b in zip(pulses1, pulses2):
            self.assertEqual(p_a.timestamp_us, p_b.timestamp_us)
            self.assertEqual(p_a.frequency_mhz, p_b.frequency_mhz)

if __name__ == '__main__':
    unittest.main()
