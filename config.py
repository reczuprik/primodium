# --- THE FINAL, CANONICAL CONSTANTS ---

# World & Performance Parameters
WORLD_WIDTH = 256; WORLD_HEIGHT = 256
INITIAL_POPULATION = 200
MAX_POPULATION = 15000 

# Seeding and Ecosystem
INITIAL_ENERGY_DENSITY = 0.30
SOLAR_FLUX_INTERVAL = 10
SOLAR_FLUX_AMOUNT = 0.01

# --- NEW: BASE PHYSICS COSTS ---
# These are the baseline costs for a creature with a gene value of 1.0
C_E_BASE = 0.8       # Base Cost of Existence
C_S_BASE = 0.25      # Base Cost of Sensation
C_M_BASE = 4.0       # Base Cost of Movement
C_R = 120.0          # Reproduction cost is absolute
ENERGY_PER_QUANTUM = 300.0
PREDATION_ENERGY_YIELD = 0.75
REPRODUCTION_ENERGY_INHERITANCE = 0.5 # Child inherits 50% of the parent's post-cost energy

# --- EPOCH 3: THE AGE OF THE BODY ---
C_MEMORY = 0.2
MEMORY_DECAY = 0.98

# Genetic Bounds (15 Genes Total)
GENE_WEIGHT_MIN = 0.5; GENE_WEIGHT_MAX = 1.5
GENE_STRESS_THRESHOLD_MIN = 20.0; GENE_STRESS_THRESHOLD_MAX = 110.0
GENE_SURPLUS_THRESHOLD_MIN = 130.0; GENE_SURPLUS_THRESHOLD_MAX = 250.0
GENE_MUTATION_RATE_MIN = 0.01; GENE_MUTATION_RATE_MAX = 0.10
GENE_SOCIAL_WEIGHT_MIN = -1.0; GENE_SOCIAL_WEIGHT_MAX = 1.0
GENE_MAX_AGE_MIN = 200.0; GENE_MAX_AGE_MAX = 800.0
GENE_MEMORY_WEIGHT_MIN = -1.0; GENE_MEMORY_WEIGHT_MAX = 1.0
GENE_SENSE_RANGE_MIN = 1; GENE_SENSE_RANGE_MAX = 5
# NEW PHYSICAL GENES
GENE_BODY_STRENGTH_MIN = 0.5; GENE_BODY_STRENGTH_MAX = 2.0
GENE_BODY_SPEED_MIN = 0.5; GENE_BODY_SPEED_MAX = 2.0

# Experiment Harness & Chronicle
MAX_TICKS_PER_RUN = 5000
MAX_SIMULATION_RUNS = 1
LOG_ONLY_SUCCESSFUL_RUNS = True
TICKER_INTERVAL = 500

ACTION_SPAWN = 0.0; ACTION_MOVE = 1.0; ACTION_REPRODUCE = 2.0
ACTION_DEATH_STARVATION = 3.0; ACTION_DEATH_AGE = 4.0; ACTION_PREDATE = 5.0
ACTION_ENERGY_SPAWN = 6.0; ACTION_SOLAR_INFUSION = 7.0; ACTION_ENERGY_CONSUMED = 8.0

# Visualization
PREDATION_THRESHOLD = 0.1; ANXIETY_THRESHOLD = 70.0

# --- ADD THIS LIST ---
# The canonical list of gene names, defining the genome's structure.
GENOME_KEYS = [
    'w_north', 'w_east', 'w_south', 'w_west',
    'energy_stress_threshold', 'energy_surplus_threshold', 'mutation_rate',
    'w_peace', 'w_predate','max_age', 'kin_tag', 'w_flee', 'sense_range',
    'body_strength', 'body_speed'
]

