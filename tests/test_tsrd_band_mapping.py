"""
Unit Tests for BandMap & Frequency Partitioning
CASS-EW SIH Problem Statement 26055
"""

import unittest
from data.band_map import BandMap


class TestBandMap(unittest.TestCase):
    def test_uniform_physical_partition(self):
        bm = BandMap(num_bands=10, min_freq_mhz=0.0, max_freq_mhz=10000.0)
        self.assertEqual(bm.num_bands, 10)
        self.assertEqual(bm.band_to_range(0), (0.0, 1000.0))
        self.assertEqual(bm.band_to_range(9), (9000.0, 10000.0))

    def test_deterministic_mapping(self):
        bm = BandMap(num_bands=10, min_freq_mhz=0.0, max_freq_mhz=10000.0)
        # 500 MHz should be in Band 0: [0, 1000)
        self.assertEqual(bm.frequency_to_band(500.0), 0)
        # 1500 MHz in Band 1
        self.assertEqual(bm.frequency_to_band(1500.0), 1)
        # 9500 MHz in Band 9
        self.assertEqual(bm.frequency_to_band(9500.0), 9)

    def test_boundary_conditions(self):
        bm = BandMap(num_bands=5, min_freq_mhz=0.0, max_freq_mhz=5000.0)
        # Lower edge of band 1: 1000.0 belongs to band 1
        self.assertEqual(bm.frequency_to_band(1000.0), 1)
        # Just below band 1: 999.999 belongs to band 0
        self.assertEqual(bm.frequency_to_band(999.999), 0)
        # Exactly upper limit: 5000.0 belongs to final band (band 4)
        self.assertEqual(bm.frequency_to_band(5000.0), 4)

    def test_comprehensive_boundary_conditions(self):
        # 5 bands: [500, 1000), [1000, 1500), [1500, 2000), [2000, 2500), [2500, 3000]
        bm = BandMap(num_bands=5, min_freq_mhz=500.0, max_freq_mhz=3000.0)

        # 1. Below first band
        self.assertIsNone(bm.frequency_to_band(499.99))
        self.assertIsNone(bm.frequency_to_band(0.0))

        # 2. Exact lower boundary of first band
        self.assertEqual(bm.frequency_to_band(500.0), 0)

        # 3. Exact lower boundary of internal band (Band 1)
        self.assertEqual(bm.frequency_to_band(1000.0), 1)

        # 4. Between bands / just below boundary (999.9999 MHz in Band 0)
        self.assertEqual(bm.frequency_to_band(999.9999), 0)

        # 5. Exact boundary of Band 2 (1500.0 MHz in Band 2)
        self.assertEqual(bm.frequency_to_band(1500.0), 2)

        # 6. Exact upper boundary of final band (Band 4)
        self.assertEqual(bm.frequency_to_band(3000.0), 4)

        # 7. Above final band
        self.assertIsNone(bm.frequency_to_band(3000.01))
        self.assertIsNone(bm.frequency_to_band(5000.0))


if __name__ == "__main__":
    unittest.main()
