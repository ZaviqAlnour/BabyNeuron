
import sys
import os
import numpy as np
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.digital_baby import DigitalBaby
from src.environment import WhiteRoomEnvironment

def run_simulation(steps=1000):
    print(f"Initializing Digital Baby Simulation ({steps} steps)...")
    
    baby = DigitalBaby()
    env = WhiteRoomEnvironment()
    
    history = {
        'free_energy': [],
        'actions': [],
        'dopamine': [],
        'norepinephrine': []
    }
    
    start_time = time.time()
    
    for t in range(steps):
        # Run step
        result = baby.step(env)
        
        # Record metrics
        history['free_energy'].append(result['free_energy'])
        history['actions'].append(result['action'])
        history['dopamine'].append(result['neuromodulators']['dopamine'])
        history['norepinephrine'].append(result['neuromodulators']['norepinephrine'])
        
        # Periodic reporting
        if (t + 1) % 100 == 0:
            avg_fe = np.mean(history['free_energy'][-100:])
            avg_da = np.mean(history['dopamine'][-100:])
            action_counts = np.bincount(history['actions'][-100:], minlength=9)
            
            print(f"Step {t+1}/{steps} | "
                  f"Avg Free Energy: {avg_fe:.4f} | "
                  f"Avg Dopamine: {avg_da:.4f} | "
                  f"Actions: {action_counts}")
            
    elapsed = time.time() - start_time
    print(f"\nSimulation complete in {elapsed:.2f}s ({steps/elapsed:.2f} steps/s)")
    
    # Analysis
    first_100_fe = np.mean(history['free_energy'][:100])
    last_100_fe = np.mean(history['free_energy'][-100:])
    print(f"\nHabituation Analysis:")
    print(f"Initial Free Energy (0-100): {first_100_fe:.4f}")
    print(f"Final Free Energy ({steps-100}-{steps}): {last_100_fe:.4f}")
    
    if last_100_fe < first_100_fe:
        print("SUCCESS: Free energy decreased (Habituation observed)")
    else:
        print("WARNING: Free energy did not decrease")
        
    return history

if __name__ == '__main__':
    run_simulation()
