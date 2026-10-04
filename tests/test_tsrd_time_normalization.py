"""
Unit Tests for Time Normalization Layer
CASS-EW SIH Problem Statement 26055
"""

import unittest
import numpy as np
from data.time_normalization import TimeNormalizer


class TestTimeNormalization(unittest.TestCase):
    def test_conversion_precision(self):
        # 1,500,000 us = 1.5 s
        self.assertEqual(TimeNormalizer.us_to_seconds(1_500_000.0), 1.5)
        self.assertEqual(TimeNormalizer.seconds_to_us(1.5), 1_500_000.0)

    def test_round_trip(self):
        original_us = 1234567.89
        s = TimeNormalizer.us_to_seconds(original_us)
        recovered_us = TimeNormalizer.seconds_to_us(s)
        self.assertAlmostEqual(original_us, recovered_us, places=5)

    def test_array_conversion(self):
        arr_us = np.array([100.0, 1000.0, 10000.0])
        arr_s = TimeNormalizer.us_to_seconds(arr_us)
        np.testing.assert_allclose(arr_s, [0.0001, 0.001, 0.01])

    def test_dwell_validation(self):
        self.assertEqual(TimeNormalizer.normalize_dwell(0.01), 0.01)
        with self.assertRaises(ValueError):
            TimeNormalizer.normalize_dwell(0.0)
        with self.assertRaises(ValueError):
            TimeNormalizer.normalize_dwell(-0.05)


if __name__ == "__main__":
    unittest.main()
