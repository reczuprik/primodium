# In your main simulation runner script (e.g., main.py)

import csv
import argparse
import os
import numpy as np
from primordium_engine.world import World
import primordium_engine.config as cfg

def save_chronicle(chronicle_data: np.ndarray, run_number: int):
    """Saves the final chronicle data to a CSV file."""
    os.makedirs("chronicles", exist_ok=True)
    filename = os.path.join("chronicles", f"run_{run_number}_chronicle.csv")
    
    # Using the best-practice of getting the keys from config.
    header = ['tick','eid','action','x1','y1','x2','y2','extra'] + cfg.GENOME_KEYS
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(chronicle_data)
    print(f"Successfully saved chronicle to {filename}")

def main(num_evolutions: int):
    """
    Runs a specified number of separate, independent evolution simulations.
    """
    print(f"--- Preparing to run {num_evolutions} evolution(s). ---")
    
    

    # --- The restored batch-run loop ---
    for run_number in range(1, num_evolutions + 1):
        print(f"\n--- Starting Evolution #{run_number}/{num_evolutions} ---")
        
        # 1. Initialize a completely new world for each evolution.
        world = World()

        # 2. Run the ENTIRE simulation with one function call, passing run_number for terrain saving.
        final_chronicle = world.run(run_number)

        # 3. Save the results.
        if len(final_chronicle) > 0:
            save_chronicle(final_chronicle, run_number)
        else:
            print("Simulation resulted in an empty chronicle. Nothing to save.")

        print(f"--- Evolution #{run_number} Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Primordium simulation harness.")
    # --- MODIFIED: The parameter now defines HOW MANY runs. ---
    parser.add_argument("num_evolutions", type=int, nargs='?', default=1)
    args = parser.parse_args()
    main(args.num_evolutions)