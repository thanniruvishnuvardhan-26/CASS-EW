import unittest
import os
import tempfile
import h5py
import numpy as np

from data.h5_pdws import H5PDWAdapter, PDWRecord

class TestH5PDWAdapter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.valid_h5_path = os.path.join(self.temp_dir.name, "valid.h5")
        self._create_valid_fixture(self.valid_h5_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_valid_fixture(self, path, n_records=10):
        with h5py.File(path, 'w') as f:
            # Data: N x 5 (ToA, Freq, PW, AoA, Amp)
            # ToA monotonically increasing
            data = np.zeros((n_records, 5), dtype=np.float32)
            data[:, 0] = np.arange(n_records) * 100.0  # ToA
            data[:, 1] = np.random.uniform(100, 2000, n_records) # Freq
            data[:, 2] = np.random.uniform(1, 10, n_records)     # PW
            data[:, 3] = np.random.uniform(0, 360, n_records)    # AoA
            data[:, 4] = np.random.uniform(-100, 0, n_records)   # Amp
            f.create_dataset('/data', data=data)

            # Labels
            labels = np.array([f"emitter_{i % 3}".encode('utf-8') for i in range(n_records)])
            f.create_dataset('/labels', data=labels)

            # Metadata
            meta = f.create_group('/metadata')
            feature_names = [b"ToA", b"Frequency", b"PulseWidth", b"AoA", b"Amplitude"]
            meta.create_dataset('feature_names', data=np.array(feature_names))

            rx = meta.create_group('receiver')
            rx.attrs['position'] = np.array([0.0, 0.0, 0.0])
            rx.attrs['sensitivity'] = -100.0

            tx = meta.create_group('transmitters')
            tx.attrs['count'] = 3

    def test_valid_file_loads(self):
        adapter = H5PDWAdapter(self.valid_h5_path)
        summary = adapter.get_summary()
        self.assertEqual(summary.record_count, 10)
        self.assertEqual(summary.label_count, 3) # 3 unique emitters

    def test_correct_feature_schema(self):
        adapter = H5PDWAdapter(self.valid_h5_path)
        summary = adapter.get_summary()
        self.assertEqual(summary.feature_names, ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"])

    def test_correct_record_count(self):
        adapter = H5PDWAdapter(self.valid_h5_path)
        self.assertEqual(adapter.get_record_count(), 10)

    def test_label_count_matches_data_rows(self):
        adapter = H5PDWAdapter(self.valid_h5_path)
        self.assertEqual(adapter.get_record_count(), adapter.get_label_count())

    def test_toa_monotonicity_validation(self):
        bad_path = os.path.join(self.temp_dir.name, "bad_toa.h5")
        self._create_valid_fixture(bad_path)
        with h5py.File(bad_path, 'r+') as f:
            data = f['/data'][:]
            data[1, 0] = 5000.0 # break monotonicity
            f['/data'][...] = data
            
        with self.assertRaisesRegex(ValueError, "ToA is not monotonically increasing"):
            H5PDWAdapter(bad_path)

    def test_nan_inf_rejection(self):
        bad_path = os.path.join(self.temp_dir.name, "bad_nan.h5")
        self._create_valid_fixture(bad_path)
        with h5py.File(bad_path, 'r+') as f:
            data = f['/data'][:]
            data[2, 1] = np.nan
            f['/data'][...] = data
            
        with self.assertRaisesRegex(ValueError, "Dataset contains NaN or Inf values"):
            H5PDWAdapter(bad_path)

    def test_malformed_shape_rejection(self):
        bad_path = os.path.join(self.temp_dir.name, "bad_shape.h5")
        self._create_valid_fixture(bad_path)
        with h5py.File(bad_path, 'r+') as f:
            del f['/data']
            f.create_dataset('/data', data=np.zeros((10, 4))) # only 4 features
            
        with self.assertRaisesRegex(ValueError, "Malformed /data shape"):
            H5PDWAdapter(bad_path)

    def test_observable_records_exclude_labels(self):
        adapter = H5PDWAdapter(self.valid_h5_path)
        for record in adapter.iter_observations():
            self.assertIsInstance(record, PDWRecord)
            # Make sure it lacks label attribute
            self.assertFalse(hasattr(record, 'label'))
            self.assertFalse(hasattr(record, 'transmitter_id'))

    def test_ground_truth_is_available_separately(self):
        adapter = H5PDWAdapter(self.valid_h5_path)
        gt_list = list(adapter.iter_ground_truth())
        self.assertEqual(len(gt_list), 10)
        self.assertTrue(all(isinstance(gt, str) for gt in gt_list))

    def test_original_chronological_order_is_preserved(self):
        adapter = H5PDWAdapter(self.valid_h5_path)
        last_time = -1
        for record in adapter.iter_observations():
            self.assertTrue(record.timestamp >= last_time)
            last_time = record.timestamp

    def test_receiver_metadata_can_be_inspected(self):
        adapter = H5PDWAdapter(self.valid_h5_path)
        meta = adapter.get_receiver_metadata()
        self.assertIn('position', meta)
        self.assertIn('sensitivity', meta)

    def test_transmitter_metadata_can_be_inspected(self):
        adapter = H5PDWAdapter(self.valid_h5_path)
        meta = adapter.get_transmitter_metadata()
        self.assertIn('count', meta)
        self.assertEqual(meta['count'], 3)

    def test_empty_dataset_rejection(self):
        bad_path = os.path.join(self.temp_dir.name, "bad_empty.h5")
        self._create_valid_fixture(bad_path, n_records=0)
        with self.assertRaisesRegex(ValueError, "Dataset is empty"):
            H5PDWAdapter(bad_path)

    def test_missing_required_hdf5_object_rejection(self):
        bad_path = os.path.join(self.temp_dir.name, "bad_missing.h5")
        self._create_valid_fixture(bad_path)
        with h5py.File(bad_path, 'r+') as f:
            del f['/metadata/feature_names']
            
        with self.assertRaisesRegex(ValueError, "Missing required HDF5 object"):
            H5PDWAdapter(bad_path)

if __name__ == '__main__':
    unittest.main()
