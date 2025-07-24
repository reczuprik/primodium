# primordium_engine/world.py

import numpy as np
import os

from . import config as cfg # Relative imports
from . import engine
from .terrain import generate_terrain, get_terrain_solar_efficiency
# ==============================================================================
# PART 3: THE PYTHON WORLD CLASS (FIXED WITH TERRAIN)
# ==============================================================================
class World:
    """Prepares initial conditions and launches the Numba simulation engine - FIXED VERSION with terrain."""
    def __init__(self, seed=None):
        self.width, self.height = cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT
        self.energy_grid = np.zeros((self.width, self.height), dtype=np.float32)
        
        # Generate terrain with optional seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        self.terrain_grid = generate_terrain(self.width, self.height)
        
        self.entity_data = np.zeros((cfg.MAX_POPULATION, 9), dtype=np.float32)
        self.entities_genome = np.zeros((cfg.MAX_POPULATION, 15), dtype=np.float32)
        self.next_available_uid = 0
        self._seed_world()

    def _seed_world(self):
        """FIXED: Proper initial conditions with terrain-aware seeding."""
        print("Generating world with terrain...")
        
        # TERRAIN FIX: Seed initial energy preferentially on good terrain
        suitable_locations = []
        for x in range(self.width):
            for y in range(self.height):
                terrain_type = self.terrain_grid[x, y]
                if terrain_type != 0:  # Not water
                    efficiency = get_terrain_solar_efficiency(terrain_type)
                    # Higher efficiency terrain gets more initial energy locations
                    weight = int(efficiency * 10)
                    for _ in range(weight):
                        suitable_locations.append((x, y))
        
        # Place initial population on suitable terrain
        if len(suitable_locations) >= cfg.INITIAL_POPULATION:
            spawn_indices = np.random.choice(len(suitable_locations), cfg.INITIAL_POPULATION, replace=False)
            spawn_locations = [suitable_locations[i] for i in spawn_indices]
            xs = np.array([loc[0] for loc in spawn_locations])
            ys = np.array([loc[1] for loc in spawn_locations])
        else:
            # Fallback if not enough suitable terrain
            xs = np.random.randint(0, self.width, cfg.INITIAL_POPULATION)
            ys = np.random.randint(0, self.height, cfg.INITIAL_POPULATION)
        
        self.energy_grid[xs, ys] = 1.0
        
        # Add additional energy across the world, terrain-weighted
        total_energy_target = int(self.width * self.height * cfg.INITIAL_ENERGY_DENSITY)
        remaining_energy_to_place = total_energy_target - cfg.INITIAL_POPULATION
        
        if remaining_energy_to_place > 0 and len(suitable_locations) > cfg.INITIAL_POPULATION:
            # Remove already used spawn locations
            available_locations = []
            spawn_coords = set(zip(xs, ys))
            for loc in suitable_locations:
                if loc not in spawn_coords:
                    available_locations.append(loc)
            
            if len(available_locations) >= remaining_energy_to_place:
                energy_indices = np.random.choice(len(available_locations), remaining_energy_to_place, replace=False)
                energy_locations = [available_locations[i] for i in energy_indices]
                rx = np.array([loc[0] for loc in energy_locations])
                ry = np.array([loc[1] for loc in energy_locations])
                self.energy_grid[rx, ry] += 1.0
        
        # FIXED: Create entities with proper starting conditions
        for i in range(cfg.INITIAL_POPULATION):
            genome = np.array([
                np.random.uniform(cfg.GENE_WEIGHT_MIN, cfg.GENE_WEIGHT_MAX), 
                np.random.uniform(cfg.GENE_WEIGHT_MIN, cfg.GENE_WEIGHT_MAX),
                np.random.uniform(cfg.GENE_WEIGHT_MIN, cfg.GENE_WEIGHT_MAX), 
                np.random.uniform(cfg.GENE_WEIGHT_MIN, cfg.GENE_WEIGHT_MAX),
                np.random.uniform(cfg.GENE_STRESS_THRESHOLD_MIN, cfg.GENE_STRESS_THRESHOLD_MAX),
                np.random.uniform(cfg.GENE_SURPLUS_THRESHOLD_MIN, cfg.GENE_SURPLUS_THRESHOLD_MAX),
                np.random.uniform(cfg.GENE_MUTATION_RATE_MIN, cfg.GENE_MUTATION_RATE_MAX),
                np.random.uniform(cfg.GENE_SOCIAL_WEIGHT_MIN, cfg.GENE_SOCIAL_WEIGHT_MAX),
                np.random.uniform(cfg.GENE_SOCIAL_WEIGHT_MIN, cfg.GENE_SOCIAL_WEIGHT_MAX),
                np.random.uniform(cfg.GENE_MAX_AGE_MIN, cfg.GENE_MAX_AGE_MAX),
                np.random.rand(),
                np.random.uniform(cfg.GENE_MEMORY_WEIGHT_MIN, cfg.GENE_MEMORY_WEIGHT_MAX),
                float(np.random.randint(cfg.GENE_SENSE_RANGE_MIN, cfg.GENE_SENSE_RANGE_MAX + 1)),
                np.random.uniform(cfg.GENE_BODY_STRENGTH_MIN, cfg.GENE_BODY_STRENGTH_MAX),
                np.random.uniform(cfg.GENE_BODY_SPEED_MIN, cfg.GENE_BODY_SPEED_MAX),
            ], dtype=np.float32)
            self.entities_genome[i] = genome
            uid = self.next_available_uid
            self.next_available_uid += 1
            
            # REPRODUCTION EXPLOSION FIX: Start with energy well below surplus threshold
            # This prevents immediate reproduction and forces entities to forage first
            starting_energy = genome[4] * cfg.INITIAL_ENERGY_MULTIPLIER  # Start at stress threshold * multiplier
            
            # Entity data: [x, y, energy, alive, age, memory_strength, memory_target, last_reproduction_tick, uid]
            self.entity_data[i] = [xs[i], ys[i], starting_energy, 1.0, 0.0, 0.0, -1.0, -cfg.REPRODUCTION_COOLDOWN, uid]
        
        print(f"World seeded: {cfg.INITIAL_POPULATION} entities on terrain with {np.sum(self.energy_grid)} energy units")
        
        # Print terrain statistics
        unique, counts = np.unique(self.terrain_grid, return_counts=True)
        terrain_names = ["Water", "Plains", "Forest", "Mountains"]
        for terrain_type, count in zip(unique, counts):
            percentage = (count / (self.width * self.height)) * 100
            print(f"  {terrain_names[terrain_type]}: {count} cells ({percentage:.1f}%)")

    def run(self, run_number=1) -> np.ndarray:
        """Launches the high-performance Numba simulation engine with terrain support."""
        print("--- Fixed Numba Core with Terrain Engaged: Forging universe from initial conditions... ---")

        self.save_terrain(run_number)

        final_chronicle = engine.run_universe(
            self.entity_data, self.entities_genome, self.energy_grid, self.terrain_grid,
            self.next_available_uid,
            cfg.MAX_TICKS_PER_RUN,
            cfg.SOLAR_FLUX_INTERVAL,
            cfg.SOLAR_FLUX_AMOUNT,
            self.width, self.height,
            cfg.TICKER_INTERVAL
        )
        print(f"--- Simulation Complete: {len(final_chronicle)} events logged. Chronicle returned to Python world. ---")
        return final_chronicle

    def get_terrain_grid(self):
        """Return the terrain grid for visualization."""
        return self.terrain_grid
    
    def save_terrain(self, run_number):
        """Save the terrain grid for visualization access."""
        os.makedirs("chronicles", exist_ok=True)
        terrain_file = os.path.join("chronicles", f"run_{run_number}_terrain.npy")
        np.save(terrain_file, self.terrain_grid)
        print(f"Terrain saved to {terrain_file}")
