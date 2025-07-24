# primordium_engine/engine.py

import numpy as np
import numba
from numba.typed import List

from . import config as cfg # Use relative import within the package
from .terrain import get_terrain_movement_cost, get_terrain_solar_efficiency

# ==============================================================================
# PART 1: THE CORE LOGIC - FIXED VERSION WITH TERRAIN
# ==============================================================================
@numba.njit
def tick_logic(width, height, energy_grid, terrain_grid, entity_data, entities_genome, next_available_uid, current_tick):
    """
    Executes a single tick of the simulation for all living entities using the
    "Graveyard" memory model. FIXED VERSION with terrain-aware movement costs.
    """
    EVENT_LOG_WIDTH = 8 + len(entities_genome[0])
    event_log = np.zeros((len(entity_data) * 4, EVENT_LOG_WIDTH), dtype=np.float32)
    log_idx = 0
    mutations, movements = 0, 0
    
    # PERFORMANCE FIX: Build occupancy map more efficiently
    occupancy_map = np.full((width, height), -1, dtype=np.int32)
    live_indices = np.where(entity_data[:, 3] > 0)[0]
    for i in live_indices:
        occupancy_map[int(entity_data[i, 0]), int(entity_data[i, 1])] = i
        
    # Handle initial spawning events (first tick only)
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

        # Apply metabolism and aging
        strength, speed = genome[13], genome[14]
        mass = strength + speed
        metabolic_tax = (cfg.C_E_BASE * mass) * (1.0 + (state[4] / genome[9]))
        metabolic_tax += state[5] * cfg.C_MEMORY
        state[2] -= metabolic_tax
        state[4] += 1
        state[5] *= cfg.MEMORY_DECAY
        if state[5] < 0.01: state[5] = 0.0; state[6] = -1.0
        
        # Handle death from starvation or old age
        if state[2] <= 0 or state[4] >= genome[9]:
            state[3] = 0
            death_cause = cfg.ACTION_DEATH_AGE if state[4] >= genome[9] else cfg.ACTION_DEATH_STARVATION
            event_log[log_idx, :8] = [0, uid, death_cause, ox, oy, 0, 0, 0]; log_idx += 1
            continue

        # REPRODUCTION FIX: Add cooldown and stricter conditions
        reproduction_cooldown_remaining = max(0, cfg.REPRODUCTION_COOLDOWN - (current_tick - state[7]))
        is_in_surplus = state[2] > genome[5]
        can_afford_reproduction = state[2] > cfg.C_R
        is_old_enough = state[4] >= cfg.MIN_REPRODUCTION_AGE
        reproduction_ready = (reproduction_cooldown_remaining == 0 and 
                            is_in_surplus and can_afford_reproduction and is_old_enough)
        
        if reproduction_ready:
            next_entity_idx = -1
            for grave_idx in range(len(entity_data)):
                if entity_data[grave_idx, 3] == 0:
                    next_entity_idx = grave_idx
                    break
            if next_entity_idx != -1:
                spawn_site = (-1, -1)
                neighbors = [(ox, (oy + 1) % height), ((ox + 1) % width, oy), 
                           (ox, (oy - 1 + height) % height), ((ox - 1 + width) % width, oy)]
                for nx, ny in neighbors:
                    if occupancy_map[nx, ny] == -1: spawn_site = (nx, ny); break
                if spawn_site[0] != -1:
                    # REPRODUCTION FIX: Balanced energy inheritance
                    parental_investment = state[2] - cfg.C_R
                    daughter_energy = parental_investment * cfg.REPRODUCTION_ENERGY_INHERITANCE
                    state[2] = parental_investment * (1.0 - cfg.REPRODUCTION_ENERGY_INHERITANCE)
                    state[7] = current_tick  # Set reproduction timestamp for cooldown
                    
                    # Mutate the parent's genome to create the child's genome.
                    new_genome = genome.copy()
                    for g_idx in range(len(genome) - 2):
                        if np.random.rand() < genome[6]: 
                            # MUTATION FIX: Smaller, bounded mutations
                            mutation_factor = np.random.normal(1.0, 0.05)  # 5% std dev instead of 10%
                            new_genome[g_idx] *= mutation_factor
                            # Apply bounds checking
                            if g_idx < 4:  # Movement weights
                                new_genome[g_idx] = max(cfg.GENE_WEIGHT_MIN, min(cfg.GENE_WEIGHT_MAX, new_genome[g_idx]))
                            elif g_idx == 4:  # Stress threshold
                                new_genome[g_idx] = max(cfg.GENE_STRESS_THRESHOLD_MIN, min(cfg.GENE_STRESS_THRESHOLD_MAX, new_genome[g_idx]))
                            elif g_idx == 5:  # Surplus threshold
                                new_genome[g_idx] = max(cfg.GENE_SURPLUS_THRESHOLD_MIN, min(cfg.GENE_SURPLUS_THRESHOLD_MAX, new_genome[g_idx]))
                            elif g_idx == 13:  # Body strength
                                new_genome[g_idx] = max(cfg.GENE_BODY_STRENGTH_MIN, min(cfg.GENE_BODY_STRENGTH_MAX, new_genome[g_idx]))
                            elif g_idx == 14:  # Body speed
                                new_genome[g_idx] = max(cfg.GENE_BODY_SPEED_MIN, min(cfg.GENE_BODY_SPEED_MAX, new_genome[g_idx]))
                    
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
                    entity_data[next_entity_idx] = [spawn_site[0], spawn_site[1], daughter_energy, 1.0, 0.0, 0.0, -1.0, -cfg.REPRODUCTION_COOLDOWN, child_uid]
                    entities_genome[next_entity_idx] = new_genome
                    occupancy_map[spawn_site[0], spawn_site[1]] = next_entity_idx
        
        # SURVIVAL DRIVE FIX: Check affordability BEFORE spending energy
        is_in_crisis = state[2] < genome[4]
        is_in_surplus_after_repro = state[2] > genome[5]
        
        if is_in_crisis or not is_in_surplus_after_repro:
            sense_range = int(genome[12])
            sense_cost = (cfg.C_S_BASE * mass) * (sense_range**2)
            
            # FIX: Check if entity can afford sensing
            if state[2] > sense_cost:
                # Can afford sensing - do intelligent behavior
                state[2] -= sense_cost
                
                sensed_threats = np.zeros((4,2), dtype=np.float32)
                sensed_food = np.zeros((4,2), dtype=np.float32)
                for r in range(1, sense_range + 1):
                    coords_to_check = [(ox, (oy + r) % height), ((ox + r) % width, oy), 
                                     (ox, (oy - r + height) % height), ((ox - r + width) % width, oy)]
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
                    if sensed_food[dir_idx, 1] > 0: 
                        desire[dir_idx] += (sensed_food[dir_idx, 0] * genome[dir_idx]) / sensed_food[dir_idx, 1]
                    if sensed_threats[dir_idx, 1] > 0:
                        threat_kin_tag, threat_dist = sensed_threats[dir_idx, 0], sensed_threats[dir_idx, 1]
                        desire[dir_idx] -= (w_flee * 50) / threat_dist
                        if mem_strength > 0.1 and abs(threat_kin_tag - mem_target_kin_tag) < 0.001:  # FIX: Proper float comparison
                            desire[dir_idx] -= (mem_strength * w_flee * 100) / threat_dist

                # Act on desires if any exist
                if np.sum(desire) > 0:
                    action = np.argmax(desire)
                    tx, ty = ox, oy
                    if action == 0: ty = (oy + 1) % height
                    elif action == 1: tx = (ox + 1) % width
                    elif action == 2: ty = (oy - 1 + height) % height
                    elif action == 3: tx = (ox - 1 + width) % width
                    
                    target_idx = occupancy_map[tx, ty]
                    if target_idx == -1:
                        # TERRAIN FIX: Calculate terrain-aware movement cost
                        terrain_type = terrain_grid[tx, ty]
                        terrain_multiplier = get_terrain_movement_cost(terrain_type)
                        move_cost = cfg.C_M_BASE * speed * terrain_multiplier
                        
                        if state[2] > move_cost:  # Check affordability
                            state[2] -= move_cost
                            movements += 1
                            state[0], state[1] = tx, ty
                            if energy_grid[tx, ty] > 0:
                                energy_grid[tx, ty] -= 1
                                state[2] += cfg.ENERGY_PER_QUANTUM
                                event_log[log_idx, :8] = [0, uid, cfg.ACTION_ENERGY_CONSUMED, tx, ty, 0, 0, -1]; log_idx += 1
                            event_log[log_idx, :8] = [0, uid, cfg.ACTION_MOVE, ox, oy, tx, ty, terrain_type]; log_idx += 1  # Include terrain in log
                    elif target_idx != i:
                        # Combat logic (simplified and more reliable)
                        my_strength, their_strength = genome[13], entities_genome[target_idx][13]
                        my_power = state[2] * my_strength
                        their_power = entity_data[target_idx][2] * their_strength
                        combat_advantage = my_power - their_power
                        w_peace, w_predate, w_retaliate = genome[7], genome[8], genome[11]
                        grudge_factor = 0.0
                        if state[5] > 0.1 and abs(entities_genome[target_idx][10] - state[6]) < 0.001:
                            grudge_factor = state[5] * w_retaliate
                        desire_predate = (w_predate * combat_advantage) + grudge_factor
                        if desire_predate > w_peace and my_power > their_power:
                            # Engage in combat with terrain-aware chase costs
                            predator_state = state; prey_state = entity_data[target_idx]
                            predator_genome = genome; prey_genome = entities_genome[target_idx]
                            terrain_type = terrain_grid[tx, ty]
                            terrain_multiplier = get_terrain_movement_cost(terrain_type)
                            predator_chase_cost = cfg.C_M_BASE * predator_genome[14] * 2.0 * terrain_multiplier
                            prey_chase_cost = cfg.C_M_BASE * prey_genome[14] * 2.0 * terrain_multiplier
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
                                    victim_uid = prey_state[8]
                                    event_log[log_idx, :8] = [0, uid, cfg.ACTION_PREDATE, ox, oy, tx, ty, victim_uid]; log_idx += 1
                                else:
                                    prey_state[5], prey_state[6] = 1.0, predator_genome[10]
                            elif predator_state[2] <= 0: predator_state[3] = 0
                            elif prey_state[2] <= 0: prey_state[3] = 0
            else:
                # Cannot afford sensing - desperate random movement
                action = np.random.randint(0, 4)
                tx, ty = ox, oy
                if action == 0: ty = (oy + 1) % height
                elif action == 1: tx = (ox + 1) % width
                elif action == 2: ty = (oy - 1 + height) % height
                elif action == 3: tx = (ox - 1 + width) % width
                
                if occupancy_map[tx, ty] == -1:
                    # TERRAIN FIX: Apply terrain costs even to desperate movement
                    terrain_type = terrain_grid[tx, ty]
                    terrain_multiplier = get_terrain_movement_cost(terrain_type)
                    move_cost = cfg.C_M_BASE * speed * terrain_multiplier
                    
                    if state[2] > move_cost:
                        state[2] -= move_cost
                        movements += 1
                        state[0], state[1] = tx, ty
                        # Check for food at new location
                        if energy_grid[tx, ty] > 0:
                            energy_grid[tx, ty] -= 1
                            state[2] += cfg.ENERGY_PER_QUANTUM
                            event_log[log_idx, :8] = [0, uid, cfg.ACTION_ENERGY_CONSUMED, tx, ty, 0, 0, -1]; log_idx += 1
                        event_log[log_idx, :8] = [0, uid, cfg.ACTION_MOVE, ox, oy, tx, ty, terrain_type]; log_idx += 1
    
    return event_log[:log_idx], mutations, movements, next_available_uid

# ==============================================================================
# PART 2: TERRAIN-AWARE SOLAR FLUX WITH PROPER EVENT LOGGING (FIXED)
# ==============================================================================
@numba.njit
def apply_terrain_solar_infusion_with_logging(energy_grid, terrain_grid, width, height, flux_amount, event_log, log_idx):
    """
    FIX: Terrain-aware solar infusion that properly logs events for playback reconstruction.
    """
    base_cells = int(width * height * flux_amount)
    for _ in range(base_cells):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        
        # Apply terrain efficiency
        terrain_type = terrain_grid[x, y]
        efficiency = get_terrain_solar_efficiency(terrain_type)
        
        # Only add energy based on terrain efficiency
        if np.random.rand() < efficiency:
            energy_grid[x, y] += 1.0
            # LOG THE SOLAR INFUSION EVENT
            event_log[log_idx, :8] = [0, -1, cfg.ACTION_SOLAR_INFUSION, x, y, 0, 0, terrain_type]
            log_idx += 1
    return log_idx

@numba.njit
def run_universe(
    entity_data, entities_genome, energy_grid, terrain_grid, initial_next_uid,
    max_ticks, solar_flux_interval, solar_flux_amount, width, height,
    ticker_interval
):
    """The monolithic, headless simulation engine - FIXED VERSION with terrain."""
    # FIX: Pre-allocate chronicle with space for solar events
    max_events_per_tick = len(entity_data) * 4 + int(width * height * solar_flux_amount) + 100
    max_total_events = max_events_per_tick * max_ticks
    chronicle = np.zeros((max_total_events, 8 + len(entities_genome[0])), dtype=np.float32)
    chronicle_idx = 0
    next_available_uid = initial_next_uid
    
    for tick in range(1, max_ticks + 1):
        # Handle terrain-aware solar flux with proper logging
        if tick % solar_flux_interval == 0:
            chronicle_idx = apply_terrain_solar_infusion_with_logging(
                energy_grid, terrain_grid, width, height, solar_flux_amount, chronicle, chronicle_idx
            )
        
        # Run main tick logic with terrain
        tick_event_log, _, _, next_available_uid = tick_logic(
            width, height, energy_grid, terrain_grid, entity_data, entities_genome, next_available_uid, tick
        )
        
        # Add tick events to chronicle
        if len(tick_event_log) > 0:
            for row in tick_event_log:
                row[0] = tick  # Set correct tick number
                chronicle[chronicle_idx] = row
                chronicle_idx += 1
        
        # Ticker feedback
        if tick % ticker_interval == 0:
            pop_count = np.sum(entity_data[:, 3] > 0)
            print("> Tick:", tick, "/", max_ticks, "| Population:", pop_count)
        
        # Check for extinction
        if np.sum(entity_data[:, 3] > 0) == 0:
            print("EXTINCTION at tick", tick)
            break
    
    return chronicle[:chronicle_idx]
