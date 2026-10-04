"""
HDF5 PDW Adapter for External Synthetic Datasets
CASS-EW SIH 26055
"""

import sys
from dataclasses import dataclass
from typing import Iterator, Tuple, List, Dict, Any
import numpy as np
import h5py

@dataclass(frozen=True)
class PDWRecord:
    index: int
    timestamp: float
    frequency: float
    pulse_width: float
    angle_of_arrival: float
    amplitude: float

@dataclass(frozen=True)
class H5DatasetSummary:
    file_path: str
    record_count: int
    label_count: int
    feature_names: List[str]
    toa_range: Tuple[float, float]
    frequency_range: Tuple[float, float]
    pulse_width_range: Tuple[float, float]
    aoa_range: Tuple[float, float]
    amplitude_range: Tuple[float, float]
    is_monotonic: bool
    receiver_metadata_keys: List[str]
    transmitter_count: int

class H5PDWAdapter:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._validate_and_load()
        
    def _validate_and_load(self):
        with h5py.File(self.file_path, 'r') as f:
            # Schema validation
            required_paths = [
                ('/data', "dataset"),
                ('/labels', "dataset"),
                ('/metadata', "group"),
                ('/metadata/feature_names', "dataset"),
                ('/metadata/receiver', "group_or_dataset"),
                ('/metadata/transmitters', "group_or_dataset")
            ]
            
            for path, ptype in required_paths:
                if path not in f:
                    raise ValueError(f"Missing required HDF5 object: {path}")
            
            data_dset = f['/data']
            labels_dset = f['/labels']
            
            self._data_shape = data_dset.shape
            self._labels_shape = labels_dset.shape
            
            if len(self._data_shape) != 2 or self._data_shape[1] != 5:
                raise ValueError(f"Malformed /data shape: {self._data_shape}. Expected (N, 5).")
            
            if self._data_shape[0] != self._labels_shape[0]:
                raise ValueError(f"Record count mismatch: data has {self._data_shape[0]} rows, labels has {self._labels_shape[0]} rows.")
            
            if self._data_shape[0] == 0:
                raise ValueError("Dataset is empty.")
                
            # Read feature names
            feature_names_raw = f['/metadata/feature_names'][:]
            self._feature_names = [fn.decode('utf-8') if isinstance(fn, bytes) else str(fn) for fn in feature_names_raw]
            
            expected_features = ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"]
            if self._feature_names != expected_features:
                raise ValueError(f"Feature names mismatch. Expected {expected_features}, got {self._feature_names}")
                
            # Check for NaN / Inf
            # Read in chunks or entirely. If dataset is huge, this could OOM, but requirements say to reject malformed files clearly.
            # Assuming files fit in memory for this validation phase.
            data_array = data_dset[:]
            if np.isnan(data_array).any() or np.isinf(data_array).any():
                raise ValueError("Dataset contains NaN or Inf values.")
                
            # Monotonicity check on ToA (column 0)
            toas = data_array[:, 0]
            if not np.all(np.diff(toas) >= 0):
                raise ValueError("ToA is not monotonically increasing. Chronological ordering violated.")

            self._toa_range = (float(np.min(toas)), float(np.max(toas)))
            self._freq_range = (float(np.min(data_array[:, 1])), float(np.max(data_array[:, 1])))
            self._pw_range = (float(np.min(data_array[:, 2])), float(np.max(data_array[:, 2])))
            self._aoa_range = (float(np.min(data_array[:, 3])), float(np.max(data_array[:, 3])))
            self._amp_range = (float(np.min(data_array[:, 4])), float(np.max(data_array[:, 4])))

            rx_meta = f['/metadata/receiver']
            self._receiver_keys = list(rx_meta.keys()) if isinstance(rx_meta, h5py.Group) else list(rx_meta.attrs.keys())
            
            tx_meta = f['/metadata/transmitters']
            self._tx_count = len(tx_meta.keys()) if isinstance(tx_meta, h5py.Group) else self._labels_shape[0] # Fallback
            # Better counting of unique labels:
            labels_array = labels_dset[:]
            self._unique_label_count = len(np.unique(labels_array))

    def get_record_count(self) -> int:
        return self._data_shape[0]
        
    def get_label_count(self) -> int:
        return self._labels_shape[0]
        
    def get_summary(self) -> H5DatasetSummary:
        return H5DatasetSummary(
            file_path=self.file_path,
            record_count=self.get_record_count(),
            label_count=self._unique_label_count,
            feature_names=self._feature_names,
            toa_range=self._toa_range,
            frequency_range=self._freq_range,
            pulse_width_range=self._pw_range,
            aoa_range=self._aoa_range,
            amplitude_range=self._amp_range,
            is_monotonic=True,
            receiver_metadata_keys=self._receiver_keys,
            transmitter_count=self._tx_count
        )
        
    def iter_observations(self) -> Iterator[PDWRecord]:
        """
        Yields observable PDW records. 
        MANDATORY: Does NOT include labels to prevent leakage.
        """
        with h5py.File(self.file_path, 'r') as f:
            data = f['/data']
            # Iterating by yielding to avoid loading entire object arrays into memory
            for i in range(data.shape[0]):
                row = data[i]
                yield PDWRecord(
                    index=i,
                    timestamp=float(row[0]),
                    frequency=float(row[1]),
                    pulse_width=float(row[2]),
                    angle_of_arrival=float(row[3]),
                    amplitude=float(row[4])
                )
                
    def iter_ground_truth(self) -> Iterator[Any]:
        """
        Yields ground truth labels separately.
        MUST NEVER be passed to the scheduler.
        """
        with h5py.File(self.file_path, 'r') as f:
            labels = f['/labels']
            for i in range(labels.shape[0]):
                label = labels[i]
                if isinstance(label, bytes):
                    yield label.decode('utf-8')
                else:
                    yield label

    def get_receiver_metadata(self) -> Dict[str, Any]:
        return self._extract_metadata('/metadata/receiver')

    def get_transmitter_metadata(self) -> Dict[str, Any]:
        return self._extract_metadata('/metadata/transmitters')
        
    def _extract_metadata(self, path: str) -> Dict[str, Any]:
        with h5py.File(self.file_path, 'r') as f:
            meta_obj = f[path]
            meta = {}
            if isinstance(meta_obj, h5py.Group):
                for k, v in meta_obj.items():
                    if isinstance(v, h5py.Dataset):
                        val = v[()]
                        meta[k] = val.decode('utf-8') if isinstance(val, bytes) else val
                    else:
                        meta[k] = f"Group: {k}"
            elif isinstance(meta_obj, h5py.Dataset):
                val = meta_obj[()]
                if isinstance(val, np.ndarray):
                    meta['value'] = val.tolist()
                else:
                    meta['value'] = val.decode('utf-8') if isinstance(val, bytes) else val
            
            for k, v in meta_obj.attrs.items():
                meta[k] = v.decode('utf-8') if isinstance(v, bytes) else v
            return meta

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data/h5_pdws.py <path-to-h5>")
        sys.exit(1)
        
    adapter = H5PDWAdapter(sys.argv[1])
    summary = adapter.get_summary()
    
    print("=== HDF5 PDW ADAPTER SMOKE TEST ===")
    print(f"File: {summary.file_path}")
    print(f"Records: {summary.record_count}")
    print(f"Unique Labels: {summary.label_count}")
    print(f"Features: {summary.feature_names}")
    print(f"ToA Range: {summary.toa_range}")
    print(f"Freq Range: {summary.frequency_range}")
    print(f"PW Range: {summary.pulse_width_range}")
    print(f"AoA Range: {summary.aoa_range}")
    print(f"Amp Range: {summary.amplitude_range}")
    print(f"Monotonic: {summary.is_monotonic}")
    print(f"Receiver Metadata Keys: {summary.receiver_metadata_keys}")
    print(f"Transmitter Count: {summary.transmitter_count}")
    
    print("\n--- Observable Record Preview (First 3) ---")
    obs_iter = adapter.iter_observations()
    for _ in range(min(3, summary.record_count)):
        print(next(obs_iter))
        
    print("\n--- Ground Truth Preview (First 3) ---")
    gt_iter = adapter.iter_ground_truth()
    for _ in range(min(3, summary.record_count)):
        print(next(gt_iter))
        
    print("\n--- Receiver Metadata ---")
    print(adapter.get_receiver_metadata())
