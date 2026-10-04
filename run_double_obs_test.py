import sys
sys.path.insert(0, '.')
from evaluation.final_benchmark import *
import numpy as np

def train_rl_fixed():
    scheduler = RLScheduler(NUM_BANDS, (1, 2, 3), 0.1, 0.9, 0.3)
    for episode in range(300):
        trace = generate_trace(10000 + episode, 300)
        detection_random, false_alarm_random = generate_detector_trace(20000 + episode)
        belief = np.ones(NUM_BANDS) * 0.1
        time_index = 0
        while time_index < len(trace):
            state = scheduler.get_state(belief)
            action = scheduler.choose_action(state, training=True)
            band, dwell = scheduler.action_to_parameters(action)
            
            dwell_start = time_index
            detections = 0
            reward = 0.0
            
            for _ in range(dwell):
                if time_index >= len(trace):
                    break
                signal, detected = observe(trace, detection_random, false_alarm_random, time_index, band)
                if detected:
                    detections += 1
                
                false_alarm = (not signal and detected)
                reward += calculate_reward(detected=detected, signal_present=signal, dwell_time=1, false_alarm=false_alarm)
                time_index += 1

            if detections > 0:
                belief[band] = min(0.99, belief[band] + 0.3)
            else:
                belief[band] = max(0.01, belief[band] - 0.1)

            for b in range(NUM_BANDS):
                if b != band:
                    belief[b] *= 0.99

            next_state = scheduler.get_state(belief)
            scheduler.update(state, action, reward, next_state)
        scheduler.decay_epsilon()
    scheduler.epsilon = 0.0
    return scheduler

print('Double Observation Bug Verification')
print('-' * 60)
seed = 42

# Original Buggy Behavior
np.random.seed(seed)
trace = generate_trace(seed, TOTAL_TIME)
det, fa = generate_detector_trace(seed + 500)
scheduler_buggy = train_rl()
metrics_buggy = evaluate_rl(scheduler_buggy, trace, det, fa)
print(f'Buggy   | Int: {metrics_buggy["interception_rate"]:.2f} | FA: {metrics_buggy["false_alarm_rate"]:.2f}')

# Fixed Behavior
np.random.seed(seed)
trace = generate_trace(seed, TOTAL_TIME)
det, fa = generate_detector_trace(seed + 500)
scheduler_fixed = train_rl_fixed()
metrics_fixed = evaluate_rl(scheduler_fixed, trace, det, fa)
print(f'Fixed   | Int: {metrics_fixed["interception_rate"]:.2f} | FA: {metrics_fixed["false_alarm_rate"]:.2f}')

