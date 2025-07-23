# --- terrain.py - NEW FILE: Procedural Terrain Generation ---

import numpy as np
import numba

# Terrain types as described in project description
TERRAIN_WATER = 0
TERRAIN_PLAINS = 1  
TERRAIN_FOREST = 2
TERRAIN_MOUNTAINS = 3

# Terrain movement cost multipliers
TERRAIN_MOVEMENT_COSTS = {
    TERRAIN_WATER: 3.0,      # Expensive to move through water
    TERRAIN_PLAINS: 1.0,     # Base movement cost
    TERRAIN_FOREST: 2.0,     # Dense terrain, harder to move
    TERRAIN_MOUNTAINS: 2.5   # Rocky terrain, expensive movement
}

# Solar energy efficiency by terrain type
TERRAIN_SOLAR_EFFICIENCY = {
    TERRAIN_WATER: 0.3,      # Low energy availability in water
    TERRAIN_PLAINS: 1.0,     # Base solar efficiency  
    TERRAIN_FOREST: 0.7,     # Partial shade reduces solar energy
    TERRAIN_MOUNTAINS: 0.9   # High altitude, good sun but harsh
}

@numba.njit
def generate_noise(width, height, scale=0.1, octaves=4):
    """Generate Perlin-like noise for terrain generation."""
    noise = np.zeros((width, height), dtype=np.float32)
    
    for octave in range(octaves):
        frequency = scale * (2 ** octave)
        amplitude = 1.0 / (2 ** octave)
        
        for x in range(width):
            for y in range(height):
                # Simple noise function - not true Perlin but good enough
                sample_x = x * frequency
                sample_y = y * frequency
                
                # Create pseudo-random values based on position
                noise_val = np.sin(sample_x * 12.9898 + sample_y * 78.233) * 43758.5453
                noise_val = noise_val - np.floor(noise_val)  # Get fractional part
                
                noise[x, y] += noise_val * amplitude
    
    return noise

@numba.njit  
def generate_terrain(width, height, seed=42):
    """Generate procedural terrain with distinct biomes."""
    np.random.seed(seed)
    
    # Generate base elevation map
    elevation = generate_noise(width, height, scale=0.02, octaves=6)
    
    # Generate moisture map
    moisture = generate_noise(width, height, scale=0.015, octaves=4)
    
    # Generate temperature map (latitude-based with noise)
    temperature = np.zeros((width, height), dtype=np.float32)
    for x in range(width):
        for y in range(height):
            # Base temperature decreases toward edges (poles)
            lat_temp = 1.0 - abs(y - height/2) / (height/2)
            temp_noise = generate_noise(1, 1, scale=0.01, octaves=2)[0, 0]
            temperature[x, y] = lat_temp + temp_noise * 0.3
    
    # Classify terrain based on elevation, moisture, and temperature
    terrain = np.zeros((width, height), dtype=np.int32)
    
    for x in range(width):
        for y in range(height):
            elev = elevation[x, y]
            moist = moisture[x, y]
            temp = temperature[x, y]
            
            # Water: low elevation
            if elev < 0.3:
                terrain[x, y] = TERRAIN_WATER
            # Mountains: high elevation
            elif elev > 0.7:
                terrain[x, y] = TERRAIN_MOUNTAINS
            # Forest: moderate elevation, high moisture
            elif moist > 0.5 and temp > 0.3:
                terrain[x, y] = TERRAIN_FOREST
            # Plains: everything else
            else:
                terrain[x, y] = TERRAIN_PLAINS
    
    return terrain

@numba.njit
def get_terrain_movement_cost(terrain_type):
    """Get movement cost multiplier for terrain type."""
    if terrain_type == TERRAIN_WATER:
        return 3.0
    elif terrain_type == TERRAIN_FOREST:
        return 2.0
    elif terrain_type == TERRAIN_MOUNTAINS:
        return 2.5
    else:  # TERRAIN_PLAINS
        return 1.0

@numba.njit
def get_terrain_solar_efficiency(terrain_type):
    """Get solar energy efficiency for terrain type."""
    if terrain_type == TERRAIN_WATER:
        return 0.3
    elif terrain_type == TERRAIN_FOREST:
        return 0.7
    elif terrain_type == TERRAIN_MOUNTAINS:
        return 0.9
    else:  # TERRAIN_PLAINS
        return 1.0

@numba.njit
def apply_terrain_solar_infusion(energy_grid, terrain_grid, width, height, base_flux_amount):
    """Apply solar infusion with terrain-based efficiency modifiers."""
    base_cells_to_add = int(width * height * base_flux_amount)
    
    for _ in range(base_cells_to_add):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        
        # Apply terrain efficiency
        terrain_type = terrain_grid[x, y]
        efficiency = get_terrain_solar_efficiency(terrain_type)
        
        # Only add energy based on terrain efficiency
        if np.random.rand() < efficiency:
            energy_grid[x, y] += 1.0

def get_terrain_color(terrain_type):
    """Get color for terrain visualization (for playback system)."""
    colors = {
        TERRAIN_WATER: '#1e3a8a',     # Deep blue
        TERRAIN_PLAINS: '#16a34a',    # Green  
        TERRAIN_FOREST: '#15803d',    # Dark green
        TERRAIN_MOUNTAINS: '#78716c'  # Gray-brown
    }
    return colors.get(terrain_type, '#000000')