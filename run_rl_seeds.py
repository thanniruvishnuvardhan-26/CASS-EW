import sys
sys.path.insert(0, '.')
from evaluation.final_benchmark import *
import numpy as np

print('Seed | Interception | False Alarm | Miss | Efficiency')
print('-' * 60)
results = []
for seed in [42, 43, 44, 45, 46]:
    np.random.seed(seed) # Set global seed for reproducibility
    trace = generate_trace(seed, TOTAL_TIME)
    det, fa = generate_detector_trace(seed + 500)
    scheduler = train_rl() 
    metrics = evaluate_rl(scheduler, trace, det, fa)
    results.append(metrics)
    print(f"{seed:4d} | {metrics['interception_rate']:12.2f} | {metrics['false_alarm_rate']:11.2f} | {metrics['miss_rate']:4.2f} | {metrics['efficiency']:10.4f}")

ints = [m['interception_rate'] for m in results]
fas = [m['false_alarm_rate'] for m in results]
misses = [m['miss_rate'] for m in results]
effs = [m['efficiency'] for m in results]

print('-' * 60)
print(f"Mean | {np.mean(ints):12.2f} | {np.mean(fas):11.2f} | {np.mean(misses):4.2f} | {np.mean(effs):10.4f}")
print(f"Std  | {np.std(ints):12.2f} | {np.std(fas):11.2f} | {np.std(misses):4.2f} | {np.std(effs):10.4f}")
print(f"Min  | {np.min(ints):12.2f} | {np.min(fas):11.2f} | {np.min(misses):4.2f} | {np.min(effs):10.4f}")
print(f"Max  | {np.max(ints):12.2f} | {np.max(fas):11.2f} | {np.max(misses):4.2f} | {np.max(effs):10.4f}")
