# --- THE FIXED, REBALANCED CONSTANTS FOR PREDATOR SURVIVAL ---

# World & Performance Parameters
WORLD_WIDTH = 256; WORLD_HEIGHT = 256
INITIAL_POPULATION = 200
MAX_POPULATION = 15000 

# Seeding and Ecosystem
INITIAL_ENERGY_DENSITY = 0.35  # INCREASED - More initial energy
SOLAR_FLUX_INTERVAL = 8        # REDUCED - More frequent energy
SOLAR_FLUX_AMOUNT = 0.015      # INCREASED - More energy per flux

# --- PREDATOR-FRIENDLY PHYSICS COSTS ---
C_E_BASE = 0.6       # REDUCED Base Cost of Existence (was 0.8)
C_S_BASE = 0.12      # REDUCED Base Cost of Sensation (was 0.15)
C_M_BASE = 1.5       # REDUCED Base Cost of Movement (was 2.0)
C_R = 130.0          # REDUCED Reproduction cost (was 150.0)
ENERGY_PER_QUANTUM = 100.0     # INCREASED Energy per food (was 80.0)
PREDATION_ENERGY_YIELD = 0.95  # INCREASED - Better predation rewards (was 0.75)
REPRODUCTION_ENERGY_INHERITANCE = 0.7  # INCREASED inheritance (was 0.5)

# --- REPRODUCTION EXPLOSION FIXES ---
REPRODUCTION_COOLDOWN = 80          # REDUCED cooldown (was 100)
MIN_REPRODUCTION_AGE = 40           # REDUCED min age (was 50)
INITIAL_ENERGY_MULTIPLIER = 0.9     # INCREASED starting energy (was 0.8)

# --- EPOCH 3: THE AGE OF THE BODY ---
C_MEMORY = 0.15      # REDUCED memory cost (was 0.2)
MEMORY_DECAY = 0.985 # INCREASED memory persistence (was 0.98)

# Genetic Bounds (15 Genes Total) - PREDATOR FRIENDLY
GENE_WEIGHT_MIN = 0.4; GENE_WEIGHT_MAX = 1.6  # Wider range
GENE_STRESS_THRESHOLD_MIN = 15.0; GENE_STRESS_THRESHOLD_MAX = 100.0  # Lower stress
GENE_SURPLUS_THRESHOLD_MIN = 120.0; GENE_SURPLUS_THRESHOLD_MAX = 280.0  # Lower surplus needed
GENE_MUTATION_RATE_MIN = 0.008; GENE_MUTATION_RATE_MAX = 0.04  # Slightly reduced
GENE_SOCIAL_WEIGHT_MIN = -1.2; GENE_SOCIAL_WEIGHT_MAX = 1.2   # Wider range
GENE_MAX_AGE_MIN = 180.0; GENE_MAX_AGE_MAX = 900.0            # Longer potential lifespan
GENE_MEMORY_WEIGHT_MIN = -1.2; GENE_MEMORY_WEIGHT_MAX = 1.2   # Wider range
GENE_SENSE_RANGE_MIN = 1; GENE_SENSE_RANGE_MAX = 6            # INCREASED max sense range
# PHYSICAL GENES - PREDATOR FRIENDLY
GENE_BODY_STRENGTH_MIN = 0.4; GENE_BODY_STRENGTH_MAX = 2.2    # Stronger predators possible
GENE_BODY_SPEED_MIN = 0.4; GENE_BODY_SPEED_MAX = 2.2          # Faster predators possible

# Experiment Harness & Chronicle
MAX_TICKS_PER_RUN = 5000
MAX_SIMULATION_RUNS = 1
LOG_ONLY_SUCCESSFUL_RUNS = True
TICKER_INTERVAL = 500

ACTION_SPAWN = 0.0; ACTION_MOVE = 1.0; ACTION_REPRODUCE = 2.0
ACTION_DEATH_STARVATION = 3.0; ACTION_DEATH_AGE = 4.0; ACTION_PREDATE = 5.0
ACTION_ENERGY_SPAWN = 6.0; ACTION_SOLAR_INFUSION = 7.0; ACTION_ENERGY_CONSUMED = 8.0

# Visualization - ADJUSTED THRESHOLDS
PREDATION_THRESHOLD = 0.08; ANXIETY_THRESHOLD = 60.0  # Lower thresholds for easier classification

# --- THE CANONICAL GENOME STRUCTURE ---
GENOME_KEYS = [
    'w_north', 'w_east', 'w_south', 'w_west',
    'energy_stress_threshold', 'energy_surplus_threshold', 'mutation_rate',
    'w_peace', 'w_predate','max_age', 'kin_tag', 'w_flee', 'sense_range',
    'body_strength', 'body_speed'
]