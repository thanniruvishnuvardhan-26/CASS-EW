import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from evaluation.tsrd_benchmark import TSRDBenchmarkSuite

seeds = [42, 43, 44, 45, 46]
suite = TSRDBenchmarkSuite('data/tsrd_fixtures/sample_stare.h5', max_steps=100)

metrics = {
    'Sequential Scan': {'eff': [], 'pfa': [], 'latency': []},
    'Random Scan': {'eff': [], 'pfa': [], 'latency': []},
    'UCB1 Baseline': {'eff': [], 'pfa': [], 'latency': []},
    'CASS-EW Cognitive Adaptive': {'eff': [], 'pfa': [], 'latency': []},
    'Experimental Q-Learning': {'eff': [], 'pfa': [], 'latency': []}
}

for s in seeds:
    res = suite.run_all_baselines(trace_seed=s)
    for name, r in res.items():
        metrics[name]['eff'].append(r.scan_efficiency)
        metrics[name]['pfa'].append(r.receiver_pfa)
        metrics[name]['latency'].append(r.mean_intercept_time_s)

print("=== MULTI-SEED REPRODUCIBILITY REPORT (SEEDS 42..46) ===")
for name, m in metrics.items():
    print(f"[{name}]")
    print(f"  Scan Efficiency : mean={np.mean(m['eff']):.3f} | std={np.std(m['eff']):.3f} | min={np.min(m['eff']):.3f} | max={np.max(m['eff']):.3f}")
    print(f"  Receiver Pfa    : mean={np.mean(m['pfa']):.3f} | std={np.std(m['pfa']):.3f} | min={np.min(m['pfa']):.3f} | max={np.max(m['pfa']):.3f}")
    print(f"  Mean Latency (s): mean={np.mean(m['latency']):.4f} | std={np.std(m['latency']):.4f} | min={np.min(m['latency']):.4f} | max={np.max(m['latency']):.4f}")
