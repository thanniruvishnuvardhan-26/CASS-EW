import sys
sys.path.insert(0, '.')
import numpy as np

from algorithms.bayesian_scheduler import BayesianScheduler

def demo_bayesian_starvation():
    print("Bayesian Starvation Verification")
    print("-" * 40)
    scheduler = BayesianScheduler(num_bands=10)
    
    # 1. Initialization: Explore all 10 bands once (0 to 9)
    # Band 2 happens to have an intermittent emitter active right now
    for i in range(10):
        band, dwell = scheduler.get_action()
        # Simulate signal presence only on band 2
        detected = (band == 2)
        scheduler.update(band, detected)
        
    print(f"After initialization, beliefs: {scheduler.belief_model.get_beliefs()}")
    print("Band 2 has higher belief due to detection.")

    # 2. Steady State
    # Now that band 2 is the highest belief, the BayesianScheduler (which is purely greedy after initialization)
    # will perpetually select band 2, as long as it occasionally detects something to keep its belief high.
    
    band_selections = []
    
    for step in range(20):
        band, dwell = scheduler.get_action()
        band_selections.append(band)
        
        # Simulate periodic emitter on band 5 that appears every other step
        # but the scheduler will only query band 2!
        if band == 2:
            # Band 2 emitter is intermittent, say it's there 70% of the time, so we detect it often enough
            detected = True
        else:
            detected = False
            
        scheduler.update(band, detected)
        
    print(f"Band selections for the next 20 steps: {band_selections}")
    
    # Verify starvation
    starved = (5 not in band_selections)
    print(f"Was band 5 starved? {'YES' if starved else 'NO'}")

if __name__ == '__main__':
    demo_bayesian_starvation()
