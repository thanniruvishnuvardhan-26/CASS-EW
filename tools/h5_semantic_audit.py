import h5py
import numpy as np
import json
import sys
import os

def analyze_h5_file(filepath):
    print(f"Analyzing {filepath}...")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    results = {}
    with h5py.File(filepath, 'r') as f:
        # Check provenance/global attributes
        results['global_attributes'] = {k: str(v) for k, v in f.attrs.items()}

        # Feature Data
        if 'data' in f:
            data = f['data'][:]
            results['num_records'] = data.shape[0]
            # Assuming ['ToA', 'Frequency', 'PulseWidth', 'AoA', 'Amplitude'] based on adapter
            features = ['ToA', 'Frequency', 'PulseWidth', 'AoA', 'Amplitude']
            results['features'] = {}
            for i, feat in enumerate(features):
                col = data[:, i]
                results['features'][feat] = {
                    'min': float(np.min(col)),
                    'max': float(np.max(col)),
                    'mean': float(np.mean(col)),
                    'median': float(np.median(col)),
                    'dtype': str(col.dtype),
                    'num_unique': len(np.unique(col))
                }
            
            # ToA specific
            toa = data[:, 0]
            diffs = np.diff(toa)
            results['toa_diffs'] = {
                'min': float(np.min(diffs)),
                'max': float(np.max(diffs)),
                'median': float(np.median(diffs)),
                'mean': float(np.mean(diffs))
            }

        # Labels
        if 'labels' in f:
            labels = f['labels'][:]
            results['labels'] = {
                'num_labels': labels.shape[0],
                'unique_count': len(np.unique(labels)),
                'unique_values': np.unique(labels).tolist()[:20],  # first 20
                'dtype': str(labels.dtype)
            }

        # Receiver Metadata
        results['receiver'] = {}
        if 'metadata/receiver' in f:
            rx_group = f['metadata/receiver']
            results['receiver']['attributes'] = {k: str(v) for k, v in rx_group.attrs.items()}
            for k in rx_group.keys():
                val = rx_group[k][()]
                if isinstance(val, np.ndarray):
                    results['receiver'][k] = {
                        'type': 'array',
                        'shape': val.shape,
                        'dtype': str(val.dtype),
                        'values': val.tolist()[:10]  # sample
                    }
                else:
                    results['receiver'][k] = {
                        'type': 'scalar',
                        'value': str(val)
                    }

        # Transmitter Metadata
        results['transmitters'] = {}
        if 'metadata/transmitters' in f:
            tx_group = f['metadata/transmitters']
            results['transmitters']['count'] = len(tx_group.keys())
            tx_keys = list(tx_group.keys())
            if tx_keys:
                sample_tx = tx_group[tx_keys[0]]
                sample_schema = {}
                for k in sample_tx.keys():
                    item = sample_tx[k]
                    if isinstance(item, h5py.Dataset):
                        val = item[()]
                        if isinstance(val, np.ndarray):
                            sample_schema[k] = f"array, shape={val.shape}, dtype={val.dtype}, sample={val.tolist()[:5]}"
                        else:
                            sample_schema[k] = f"scalar, val={val}, dtype={type(val)}"
                    elif isinstance(item, h5py.Group):
                        grp_data = {}
                        for sub_k in item.keys():
                            sub_item = item[sub_k]
                            if isinstance(sub_item, h5py.Dataset):
                                val = sub_item[()]
                                if isinstance(val, np.ndarray):
                                    grp_data[sub_k] = f"array, shape={val.shape}, sample={val.tolist()[:5]}"
                                else:
                                    grp_data[sub_k] = f"scalar, val={val}"
                        sample_schema[k] = f"group, contents={grp_data}"
                results['transmitters']['sample_schema'] = sample_schema

    return results

def main():
    base_dir = r"C:\Anti Gravity\Datasets\scan\train_scan"
    files_to_check = [f"config_{i}.h5" for i in range(5)]
    
    all_results = {}
    for fname in files_to_check:
        path = os.path.join(base_dir, fname)
        res = analyze_h5_file(path)
        if res:
            all_results[fname] = res

    with open("h5_audit_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Results saved to h5_audit_results.json")

if __name__ == "__main__":
    main()
