# --- world.py (Complete, Corrected, and Final) ---

import numpy as np
import numba
from numba.typed import List

import config as cfg

# ==============================================================================
# PART 1: THE CORE LOGIC - This is the complete and correct tick_logic
# ==============================================================================
@numba.njit
def tick_logic(width, height, energy_grid, entity_data, entities_genome, next_available_uid):
    """
    Executes a single tick of the simulation for all living entities using the
    "Graveyard" memory model.
    """
    EVENT_LOG_WIDTH = 8 + len(entities_genome[0])
    event_log = np.zeros((len(entity_data) * 4, EVENT_LOG_WIDTH), dtype=np.float32)
    log_idx = 0
    mutations, movements = 0, 0
    
    occupancy_map = np.full((width, height), -1, dtype=np.int32)
    live_indices = np.where(entity_data[:, 3] > 0)[0]
    for i in live_indices:
        occupancy_map[int(entity_data[i, 0]), int(entity_data[i, 1])] = i
        
    if np.all(entity_data[live_indices, 4] == 0):
        for i in live_indices:
            uid = entity_data[i, 8]
            ox, oy = int(entity_data[i,0]), int(entity_data[i,1])
            log_entry = np.zeros(EVENT_LOG_WIDTH, dtype=np.float32)
            log_entry[:8] = [0, uid, cfg.ACTION_SPAWN, ox, oy, ox, oy, uid]
            log_entry[8:] = entities_genome[i]
            event_log[log_idx] = log_entry; log_idx += 1

    for i in live_indices:
        state, genome = entity_data[i], entities_genome[i]
        uid = state[8]
        ox, oy = int(state[0]), int(state[1])

        strength, speed = genome[13], genome[14]
        mass = strength + speed
        metabolic_tax = (cfg.C_E_BASE * mass) * (1.0 + (state[4] / genome[9]))
        metabolic_tax += state[5] * cfg.C_MEMORY
        state[2] -= metabolic_tax
        state[4] += 1
        state[5] *= cfg.MEMORY_DECAY
        if state[5] < 0.01: state[5] = 0.0; state[6] = -1.0
        
        if state[2] <= 0 or state[4] >= genome[9]:
            state[3] = 0
            death_cause = cfg.ACTION_DEATH_AGE if state[4] >= genome[9] else cfg.ACTION_DEATH_STARVATION
            event_log[log_idx, :8] = [0, uid, death_cause, ox, oy, 0, 0, 0]; log_idx += 1
            continue

        is_in_surplus = state[2] > genome[5]
        if is_in_surplus and state[2] > cfg.C_R:
            next_entity_idx = -1
            for grave_idx in range(len(entity_data)):
                if entity_data[grave_idx, 3] == 0:
                    next_entity_idx = grave_idx
                    break
            if next_entity_idx != -1:
                spawn_site = (-1, -1)
                neighbors = [(ox, (oy + 1) % height), ((ox + 1) % width, oy), (ox, (oy - 1 + height) % height), ((ox - 1 + width) % width, oy)]
                for nx, ny in neighbors:
                    if occupancy_map[nx, ny] == -1: spawn_site = (nx, ny); break
                if spawn_site[0] != -1:
                    # --- THE FIX IS HERE ---
                    # Parent pays the reproduction cost. The remaining energy is the
                    # "parental investment". The child inherits a fraction of this investment.
                    parental_investment = state[2] - cfg.C_R
                    daughter_energy = parental_investment * cfg.REPRODUCTION_ENERGY_INHERITANCE
                    
                    # The parent is left with the other half of the investment.
                    state[2] = parental_investment * (1.0 - cfg.REPRODUCTION_ENERGY_INHERITANCE)
                    # ----------------------------------------
                    
                    # Mutate the parent's genome to create the child's genome.
                    new_genome = genome.copy()
                    for g_idx in range(len(genome) - 2):
                        if np.random.rand() < genome[6]: new_genome[g_idx] *= np.random.uniform(0.9, 1.1)
                    if np.random.rand() < genome[6]: new_genome[10] += np.random.normal(0, 0.01)
                    if np.random.rand() < genome[6]:
                        new_genome[12] += np.random.choice(np.array([-1, 1], dtype=np.float32))
                        if new_genome[12] < cfg.GENE_SENSE_RANGE_MIN: new_genome[12] = float(cfg.GENE_SENSE_RANGE_MIN)
                        elif new_genome[12] > cfg.GENE_SENSE_RANGE_MAX: new_genome[12] = float(cfg.GENE_SENSE_RANGE_MAX)
                    mutations += 1
                    child_uid = next_available_uid
                    next_available_uid += 1
                    log_entry = np.zeros(EVENT_LOG_WIDTH, dtype=np.float32)
                    log_entry[:8] = [0, uid, cfg.ACTION_REPRODUCE, ox, oy, spawn_site[0], spawn_site[1], child_uid]
                    log_entry[8:] = new_genome
                    event_log[log_idx] = log_entry; log_idx += 1
                    entity_data[next_entity_idx] = [spawn_site[0], spawn_site[1], daughter_energy, 1.0, 0.0, 0.0, -1.0, 0.0, child_uid]
                    entities_genome[next_entity_idx] = new_genome
                    occupancy_map[spawn_site[0], spawn_site[1]] = next_entity_idx
        
        # --- Survival Drive is now correctly placed INSIDE the main loop ---
        is_in_crisis = state[2] < genome[4]
        is_in_surplus_after_repro = state[2] > genome[5]
        
        if is_in_crisis or not is_in_surplus_after_repro:
            sense_range = int(genome[12])
            sense_cost = (cfg.C_S_BASE * mass) * (sense_range**2)
            state[2] -= sense_cost
            if state[2] <= 0: state[3] = 0; continue

            sensed_threats = np.zeros((4,2), dtype=np.float32); sensed_food = np.zeros((4,2), dtype=np.float32)
            for r in range(1, sense_range + 1):
                coords_to_check = [(ox, (oy + r) % height), ((ox + r) % width, oy), (ox, (oy - r + height) % height), ((ox - r + width) % width, oy)]
                for dir_idx, (cx, cy) in enumerate(coords_to_check):
                    if sensed_food[dir_idx, 0] == 0 and energy_grid[cx, cy] > 0:
                        sensed_food[dir_idx, 0] = energy_grid[cx, cy]; sensed_food[dir_idx, 1] = float(r)
                    target_idx = occupancy_map[cx, cy]
                    if sensed_threats[dir_idx, 0] == 0 and target_idx != -1 and target_idx != i:
                        target_genome = entities_genome[target_idx]
                        if target_genome[8] > cfg.PREDATION_THRESHOLD:
                            sensed_threats[dir_idx, 0] = target_genome[10]; sensed_threats[dir_idx, 1] = float(r)
            
            desire = np.zeros(4, dtype=np.float32)
            w_flee = genome[11]
            mem_strength, mem_target_kin_tag = state[5], state[6]
            for dir_idx in range(4):
                if sensed_food[dir_idx, 1] > 0: desire[dir_idx] += (sensed_food[dir_idx, 0] * genome[dir_idx]) / sensed_food[dir_idx, 1]
                if sensed_threats[dir_idx, 1] > 0:
                    threat_kin_tag, threat_dist = sensed_threats[dir_idx, 0], sensed_threats[dir_idx, 1]
                    desire[dir_idx] -= (w_flee * 50) / threat_dist
                    if mem_strength > 0.1 and threat_kin_tag == mem_target_kin_tag:
                        desire[dir_idx] -= (mem_strength * w_flee * 100) / threat_dist

            if np.sum(desire) > 0:
                action = np.argmax(desire); tx, ty = ox, oy
                if action == 0: ty = (oy + 1) % height
                elif action == 1: tx = (ox + 1) % width
                elif action == 2: ty = (oy - 1 + height) % height
                elif action == 3: tx = (ox - 1 + width) % width
                target_idx = occupancy_map[tx, ty]
                if target_idx == -1:
                    move_cost = cfg.C_M_BASE * speed
                    state[2] -= move_cost
                    movements += 1
                    if state[2] > 0:
                        state[0], state[1] = tx, ty
                        if energy_grid[tx, ty] > 0:
                            energy_grid[tx, ty] -= 1
                            state[2] += cfg.ENERGY_PER_QUANTUM
                            event_log[log_idx, :8] = [0, uid, cfg.ACTION_ENERGY_CONSUMED, tx, ty, 0, 0, -1]; log_idx += 1
                        event_log[log_idx, :8] = [0, uid, cfg.ACTION_MOVE, ox, oy, tx, ty, -1]; log_idx += 1
                elif target_idx != i:
                    my_strength, their_strength = genome[13], entities_genome[target_idx][13]
                    my_power = state[2] * my_strength
                    their_power = entity_data[target_idx][2] * their_strength
                    combat_advantage = my_power - their_power
                    w_peace, w_predate, w_retaliate = genome[7], genome[8], genome[11]
                    grudge_factor = 0.0
                    if state[5] > 0.1 and entities_genome[target_idx][10] == state[6]:
                        grudge_factor = state[5] * w_retaliate
                    desire_predate = (w_predate * combat_advantage) + grudge_factor
                    if desire_predate > w_peace and my_power > their_power:
                        predator_state = state; prey_state = entity_data[target_idx]
                        predator_genome = genome; prey_genome = entities_genome[target_idx]
                        predator_chase_cost = cfg.C_M_BASE * predator_genome[14] * 2.0
                        prey_chase_cost = cfg.C_M_BASE * prey_genome[14] * 2.0
                        predator_state[2] -= predator_chase_cost
                        prey_state[2] -= prey_chase_cost
                        if predator_state[2] > 0 and prey_state[2] > 0:
                            catch_probability = 0.5 + (predator_genome[14] - prey_genome[14]) / 4.0
                            stamina_modifier = (predator_state[2] / my_power) - (prey_state[2] / their_power)
                            catch_probability += stamina_modifier
                            if np.random.rand() < catch_probability:
                                predator_state[2] += prey_state[2] * cfg.PREDATION_ENERGY_YIELD
                                prey_state[3] = 0
                                predator_state[0], predator_state[1] = tx, ty
                                predator_state[5], predator_state[6] = 1.0, prey_genome[10]
                                victim_uid = prey_state[8] # Get victim's UID for the log
                                event_log[log_idx, :8] = [0, uid, cfg.ACTION_PREDATE, ox, oy, tx, ty, victim_uid]; log_idx += 1
                            else:
                                prey_state[5], prey_state[6] = 1.0, predator_genome[10]
                        elif predator_state[2] <= 0: predator_state[3] = 0
                        elif prey_state[2] <= 0: prey_state[3] = 0
            elif is_in_crisis:
                move_cost = cfg.C_M_BASE * speed
                state[2] -= move_cost; movements += 1
                if state[2] > 0:
                    action = np.random.randint(0, 4); tx, ty = ox, oy
                    if action == 0: ty = (oy + 1) % height
                    elif action == 1: tx = (ox + 1) % width
                    elif action == 2: ty = (oy - 1 + height) % height
                    elif action == 3: tx = (ox - 1 + width) % width
                    if occupancy_map[tx, ty] == -1: state[0], state[1] = tx, ty; event_log[log_idx, :8] = [0, uid, cfg.ACTION_MOVE, ox, oy, tx, ty, -1]; log_idx += 1
    
    return event_log[:log_idx], mutations, movements, next_available_uid

# ==============================================================================
# PART 2: THE HEADLESS UNIVERSE RUNNER
# ==============================================================================
@numba.njit
def _apply_solar_infusion_numba(energy_grid, width, height, flux_amount):
    """A Numba-compatible version of the solar infusion logic."""
    num_to_add = int(width * height * flux_amount)
    for _ in range(num_to_add):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        energy_grid[x, y] += 1.0

@numba.njit
def run_universe(
    entity_data, entities_genome, energy_grid, initial_next_uid,
    max_ticks, solar_flux_interval, solar_flux_amount, width, height,
    ticker_interval
):
    """The monolithic, headless simulation engine."""
    chronicle_list = List()
    next_available_uid = initial_next_uid
    for tick in range(1, max_ticks + 1):
        tick_event_log, _, _, next_available_uid = tick_logic(
            width, height, energy_grid, entity_data, entities_genome, next_available_uid
        )
        if len(tick_event_log) > 0:
            tick_event_log[:, 0] = tick
            for row in tick_event_log:
                chronicle_list.append(row)
        if tick % solar_flux_interval == 0:
            _apply_solar_infusion_numba(energy_grid, width, height, solar_flux_amount)
        # --- NEW: TICKER TAPE FEEDBACK ---
        # Purpose: Periodically print the simulation status to the console.
        if tick % ticker_interval == 0:
            pop_count = np.sum(entity_data[:, 3] > 0)
            # Numba's print is simple but effective. We can't use f-strings.
            print("> Tick:", tick, "/", max_ticks, "| Population:", pop_count)
        # --------------------------------
        if np.sum(entity_data[:, 3] > 0) == 0:
            break
    if len(chronicle_list) > 0:
        final_chronicle = np.empty((len(chronicle_list), chronicle_list[0].shape[0]), dtype=np.float32)
        for i, row in enumerate(chronicle_list):
            final_chronicle[i] = row
        return final_chronicle
    else:
        return np.empty((0, 8 + entities_genome.shape[1]), dtype=np.float32)

# ==============================================================================
# PART 3: THE PYTHON WORLD CLASS (CLEANED UP SETUP HELPER)
# ==============================================================================
class World:
    """Prepares initial conditions and launches the Numba simulation engine."""
    def __init__(self):
        self.width, self.height = cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT
        self.energy_grid = np.zeros((self.width, self.height), dtype=np.float32)
        self.entity_data = np.zeros((cfg.MAX_POPULATION, 9), dtype=np.float32)
        self.entities_genome = np.zeros((cfg.MAX_POPULATION, 15), dtype=np.float32)
        self.next_available_uid = 0
        self._seed_world()

    def _seed_world(self):
        # This function seeds the initial population and energy.
        xs = np.random.randint(0, self.width, cfg.INITIAL_POPULATION)
        ys = np.random.randint(0, self.height, cfg.INITIAL_POPULATION)
        self.energy_grid[xs, ys] = 1.0
        total_energy_target = int(self.width * self.height * cfg.INITIAL_ENERGY_DENSITY)
        remaining_energy_to_place = total_energy_target - cfg.INITIAL_POPULATION
        if remaining_energy_to_place > 0:
            all_coords = np.arange(self.width * self.height)
            birth_coords_flat = xs * self.height + ys
            unique_birth_coords = np.unique(birth_coords_flat)
            empty_coords = np.setdiff1d(all_coords, unique_birth_coords, assume_unique=True)
            if len(empty_coords) >= remaining_energy_to_place:
                random_indices = np.random.choice(empty_coords, size=remaining_energy_to_place, replace=False)
                rx, ry = random_indices // self.height, random_indices % self.height
                self.energy_grid[rx, ry] += 1.0
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
            
            # Instead of starting with a massive surplus, let's start them in a "calm" state,
            # right in the middle of their stress and surplus thresholds.
            # This forces them to find food before they can reproduce.
            calm_energy_level = (genome[4] + genome[5]) / 2.0
            
            # MODIFIED: Use the new calm_energy_level for starting energy.
            self.entity_data[i] = [xs[i], ys[i], calm_energy_level, 1.0, 0.0, 0.0, -1.0, 0.0, uid]

    
    #        self.entity_data[i] = [xs[i], ys[i], genome[5] + cfg.ENERGY_PER_QUANTUM, 1.0, 0.0, 0.0, -1.0, 0.0, uid]

    def run(self) -> np.ndarray:
        """Launches the high-performance Numba simulation engine."""
        print("--- Numba Core Engaged: Forging universe from initial conditions... ---")
        final_chronicle = run_universe(
            self.entity_data, self.entities_genome, self.energy_grid,
            self.next_available_uid,
            cfg.MAX_TICKS_PER_RUN,
            cfg.SOLAR_FLUX_INTERVAL,
            cfg.SOLAR_FLUX_AMOUNT,
            self.width, self.height,
            cfg.TICKER_INTERVAL
        )
        print(f"--- Simulation Complete: {len(final_chronicle)} events logged. Chronicle returned to Python world. ---")
        return final_chronicle