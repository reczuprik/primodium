import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
import matplotlib.cm as cm
import argparse
import config as cfg
import os
import csv
from matplotlib.lines import Line2D # The missing import

# --- HELPER FUNCTIONS (The missing pieces) ---

def load_chronicle(run_number: int):
    """Loads and parses the chronicle file for a given run."""
    filename = os.path.join("chronicles", f"run_{run_number}_chronicle.csv")
    print(f"Loading chronicle from {filename}...")
    try:
        with open(filename, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            # Convert all data to float, handling potential empty rows
            return [[float(val) for val in row if val] for row in reader if row]
    except (FileNotFoundError, IOError):
        print(f"Error: Chronicle file '{filename}' not found or is empty."); return None
    except ValueError:
        print(f"Error: Could not parse data in {filename}. The file may be corrupted."); return None

def process_events(chronicle_data: list) -> dict:
    """Groups all events from the chronicle by their tick number."""
    events_by_tick = {}
    for event in chronicle_data:
        tick = int(event[0])
        if tick not in events_by_tick: events_by_tick[tick] = []
        events_by_tick[tick].append(event)
    return events_by_tick

def get_species_archetype(genome: np.ndarray):
    """Classifies an entity's species based on its genome."""
    w_predate, stress, kin_tag, w_flee, sense_range = genome[8], genome[4], genome[10], genome[11], genome[12]
    
    prefix = ""
    if sense_range > 3: prefix = "Far-Seeing "
    if w_flee > 0.5: prefix = "Fearful "

    is_predator = w_predate > cfg.PREDATION_THRESHOLD
    is_anxious = stress > cfg.ANXIETY_THRESHOLD
    
    if is_predator: return prefix + ("Vadállat" if is_anxious else "Orvvadász"), ("red" if is_anxious else "orange")
    else: return prefix + ("Ijedős Növényevő" if is_anxious else "Békés Legelő"), ("yellow" if is_anxious else "cyan")


# --- THE ORRERY CLASS (The Observatory) ---

class Orrery:
    """The interactive playback and analysis tool for Project Primordium."""
# In orrery.py, inside the Orrery class

    def __init__(self, run_number, events_by_tick):
        self.run_number = run_number
        self.events_by_tick = events_by_tick
        self.max_tick = max(events_by_tick.keys()) if events_by_tick else 0

        self.fig = plt.figure(figsize=(16, 9))
        self.fig.patch.set_facecolor('#1a1a1a')
        gs = self.fig.add_gridspec(10, 5)
        self.ax_main = self.fig.add_subplot(gs[:, 0:4])
        self.ax_info = self.fig.add_subplot(gs[0:8, 4])
        ax_slider = self.fig.add_subplot(gs[8, 4])
        self.btn_play_ax = plt.axes([0.85, 0.05, 0.05, 0.04])

        self.ax_main.set_facecolor('black'); self.ax_main.set_xlim(0, cfg.WORLD_WIDTH); self.ax_main.set_ylim(0, cfg.WORLD_HEIGHT)
        self.ax_main.set_xticks([]); self.ax_main.set_yticks([])
        self.ax_main.set_title(f"Orrery Playback: Run #{run_number}", color='white', fontsize=16)

        self.ax_info.set_facecolor('#1a1a1a'); self.ax_info.set_xticks([]); self.ax_info.set_yticks([])
        self.stats_text = self.ax_info.text(0.05, 0.95, '', color='white', fontsize=10.5, verticalalignment='top', fontfamily='monospace')

        self.energy_grid = np.zeros((cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT))
        self.energy_map = self.ax_main.imshow(self.energy_grid.T, cmap='plasma', vmin=0, vmax=5, zorder=1, interpolation='nearest', origin='lower')
        self.entity_dots = self.ax_main.scatter([], [], s=12, zorder=2, cmap='hsv')
        
        self.slider = Slider(ax=ax_slider, label='Tick', valmin=1, valmax=self.max_tick, valinit=1, valstep=1, color='#555')
        self.slider.on_changed(self.update)
        
        self.btn_play = Button(self.btn_play_ax, 'Play')
        self.btn_play.on_clicked(self.toggle_play)
        self.is_playing = False

        # --- MODIFIED: THE CORRECTED TIMER SETUP ---
        # Create the timer object ONCE and store it. Don't start it yet.
        self.animation_timer = self.fig.canvas.new_timer(interval=30)
        self.animation_timer.add_callback(self.play_step)
        # -------------------------------------------

        self.entity_states = {}
        self.current_tick = 0

        self.update(1)

    def toggle_play(self, event):
        if self.is_playing:
            # --- PAUSE LOGIC ---
            self.is_playing = False
            self.btn_play.label.set_text('Play')
            self.animation_timer.stop() # Simply stop the existing timer
        else:
            # --- PLAY LOGIC ---
            self.is_playing = True
            self.btn_play.label.set_text('Pause')
            self.animation_timer.start() # Simply start the existing timer
    def play_step(self):
        if self.is_playing:
            current_val = self.slider.val
            if current_val < self.max_tick:
                self.slider.set_val(current_val + 1)
                self.animation_timer.start()
            else:
                self.is_playing = False
                self.btn_play.label.set_text('Play')

# In orrery.py, replace the entire update method with this new version.

    def update(self, val):
        target_tick = int(self.slider.val)

        # If the user drags the slider backward, we must reset and resimulate from scratch.
        if target_tick < self.current_tick:
            self.entity_states = {}
            self.energy_grid.fill(0)
            start_tick = 1 # Start over from the beginning
        else:
            # Otherwise, we are moving forward, so we start from where we left off.
            start_tick = self.current_tick + 1

        # This loop is now much shorter. During playback, it only runs ONCE per frame.
        for tick in range(start_tick, target_tick + 1):
            # Apply metabolism and aging to the entities we already have in memory
            for eid in list(self.entity_states.keys()):
                state = self.entity_states[eid]
                genome = state['genome']
                mass = genome[13] + genome[14]
                metabolic_tax = (cfg.C_E_BASE * mass) * (1.0 + (state['age'] / genome[9]))
                metabolic_tax += state.get('mem_strength', 0.0) * cfg.C_MEMORY
                state['energy'] -= metabolic_tax
                state['age'] += 1
                state['mem_strength'] = state.get('mem_strength', 0.0) * cfg.MEMORY_DECAY
                if state['energy'] <= 0 or state['age'] >= genome[9]:
                    del self.entity_states[eid]

            # Apply the new events for this specific tick
            if tick in self.events_by_tick:
                for event in self.events_by_tick[tick]:
                    _, eid, action, x1, y1, x2, y2, extra = event[:8]
                    eid, action = int(eid), int(action)
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    if action == cfg.ACTION_ENERGY_SPAWN or action == cfg.ACTION_SOLAR_INFUSION: self.energy_grid[x1, y1] += 1
                    elif action == cfg.ACTION_ENERGY_CONSUMED:
                        if eid in self.entity_states: self.entity_states[eid]['energy'] += cfg.ENERGY_PER_QUANTUM
                        if self.energy_grid[x1, y1] > 0: self.energy_grid[x1, y1] -= 1
                    
                    elif action == cfg.ACTION_SPAWN or action == cfg.ACTION_REPRODUCE:
                        child_id = int(extra); genome = np.array(event[8:])
                        archetype, color = get_species_archetype(genome)

                        if action == cfg.ACTION_REPRODUCE and eid in self.entity_states:
                            parent_energy = self.entity_states[eid]['energy']
                            daughter_energy = (parent_energy - cfg.C_R) / 2
                            self.entity_states[eid]['energy'] = daughter_energy
                            self.entity_states[child_id] = {'pos': [x2, y2], 'genome': genome, 'age': 0, 'energy': daughter_energy, 'archetype': archetype, 'color': color, 'mem_strength': 0.0}
                        
                        elif action == cfg.ACTION_SPAWN:
                            initial_energy = genome[5] + cfg.ENERGY_PER_QUANTUM
                            self.entity_states[child_id] = {'pos': [x2, y2], 'genome': genome, 'age': 0, 'energy': initial_energy, 'archetype': archetype, 'color': color, 'mem_strength': 0.0}
                    
                    elif action == cfg.ACTION_MOVE:
                        if eid in self.entity_states: self.entity_states[eid]['pos'] = [x2, y2]
                    elif action == cfg.ACTION_PREDATE:
                        victim_id = int(extra)
                        if eid in self.entity_states and victim_id in self.entity_states:
                            self.entity_states[eid]['energy'] += self.entity_states[victim_id]['energy'] * cfg.PREDATION_ENERGY_YIELD
                            self.entity_states[eid]['pos'] = [x2, y2]
                            self.entity_states[eid]['mem_strength'] = 1.0
                            del self.entity_states[victim_id]
                    elif action == cfg.ACTION_DEATH_AGE or action == cfg.ACTION_DEATH_STARVATION:
                        if eid in self.entity_states: del self.entity_states[eid]
        
        # Update the current tick counter
        self.current_tick = target_tick

        # --- Update Visualization (this part is unchanged) ---
        if not self.entity_states: 
            self.entity_dots.set_offsets(np.empty((0, 2)))
        else:
            coords = np.array([s['pos'] for s in self.entity_states.values()])
            colors = [s['color'] for s in self.entity_states.values()] 
            self.entity_dots.set_offsets(coords)
            self.entity_dots.set_color(colors)
        
        self.energy_map.set_data(self.energy_grid.T)
        self.update_stats_panel()
        self.fig.canvas.draw_idle()
    def update_stats_panel(self):
        if not self.entity_states:
            self.stats_text.set_text(f"TICK: {int(self.slider.val)}/{self.max_tick}\n\nEXTINCTION EVENT"); return
        
        pop_count = len(self.entity_states)
        genomes = np.array([s['genome'] for s in self.entity_states.values()])
        ages = np.array([s['age'] for s in self.entity_states.values()])
        energies = np.array([s['energy'] for s in self.entity_states.values()])
        
        stats_string = (
            f"TICK: {int(self.slider.val)}/{self.max_tick}\n\n"
            f"[ECOSYSTEM]\n"
            f"Population:         {pop_count}\n"
            f"Environmental Energy: {int(np.sum(self.energy_grid))}\n\n"
            f"[BODY - AVG]\n"
            f"Age:                {np.mean(ages):.1f}\n"
            f"Energy:             {np.mean(energies):.1f}\n"
            f"Strength:           {np.mean(genomes[:, 13]):.2f}\n"
            f"Speed:              {np.mean(genomes[:, 14]):.2f}\n\n"
            f"[MIND - AVG]\n"
            f"Sense Range:        {np.mean(genomes[:, 12]):.2f}\n"
            f"Predation Gene:     {np.mean(genomes[:, 8]):.2f}\n"
            f"Fear Gene (Flee):   {np.mean(genomes[:, 11]):.2f}\n\n"
            f"[EXTREMES]\n"
            f"Oldest:             {np.max(ages) if len(ages) > 0 else 0}\n"
            f"Richest:            {np.max(energies) if len(energies) > 0 else 0:.1f}\n"
            f"Strongest:          {np.max(genomes[:, 13]) if len(genomes) > 0 else 0:.2f}\n"
            f"Fastest:            {np.max(genomes[:, 14]) if len(genomes) > 0 else 0:.2f}"
        )
        self.stats_text.set_text(stats_string)

    def run(self):
        legend_elements = [
            Line2D([0], [0], marker='o', color='#1a1a1a', label='Vadállat', markerfacecolor='red', markersize=10),
            Line2D([0], [0], marker='o', color='#1a1a1a', label='Orvvadász', markerfacecolor='orange', markersize=10),
            Line2D([0], [0], marker='o', color='#1a1a1a', label='Ijedős Növényevő', markerfacecolor='yellow', markersize=10),
            Line2D([0], [0], marker='o', color='#1a1a1a', label='Békés Legelő', markerfacecolor='cyan', markersize=10),
            Line2D([0], [0], marker='o', color='#1a1a1a', mfc='none', mew=1.5, label='Emlék Aura', markersize=10, mec='white'),
        ]
        self.ax_info.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.05, 0.05), labelcolor='white', facecolor='#2b2b2b', edgecolor='gray')
        self.fig.tight_layout(rect=[0, 0.0, 0.85, 1])
        plt.show()


def main(run_number):
    chronicle = load_chronicle(run_number)
    if chronicle:
        events = process_events(chronicle)
        orrery = Orrery(run_number, events)
        orrery.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Watcher's Orrery: An interactive simulation playback tool.")
    parser.add_argument("run_number", type=int, help="The run number of the chronicle to analyze.")
    args = parser.parse_args()
    main(args.run_number)