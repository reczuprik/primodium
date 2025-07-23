import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
import argparse
import config as cfg
import os
import csv
from matplotlib.lines import Line2D
from terrain import get_terrain_color, TERRAIN_WATER, TERRAIN_PLAINS, TERRAIN_FOREST, TERRAIN_MOUNTAINS

# Enhanced visualization constants
ENTITY_SIZE_BASE = 8
ENERGY_ALPHA = 0.7
TERRAIN_ALPHA = 0.6
ANIMATION_SPEED = 50  # milliseconds between frames

def load_chronicle(run_number: int):
    """Loads and parses the chronicle file for a given run."""
    filename = os.path.join("chronicles", f"run_{run_number}_chronicle.csv")
    print(f"Loading chronicle from {filename}...")
    try:
        with open(filename, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            return [[float(val) for val in row if val] for row in reader if row]
    except (FileNotFoundError, IOError):
        print(f"Error: Chronicle file '{filename}' not found or is empty.")
        return None
    except ValueError:
        print(f"Error: Could not parse data in {filename}. The file may be corrupted.")
        return None

def load_terrain(run_number: int):
    """Load or generate terrain for visualization."""
    try:
        # Try to load saved terrain first
        terrain_file = os.path.join("chronicles", f"run_{run_number}_terrain.npy")
        if os.path.exists(terrain_file):
            return np.load(terrain_file)
        else:
            # Generate terrain if not saved
            from terrain import generate_terrain
            terrain = generate_terrain(cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT)
            # Save for future use
            os.makedirs("chronicles", exist_ok=True)
            np.save(terrain_file, terrain)
            return terrain
    except Exception as e:
        print(f"Warning: Could not load terrain: {e}")
        # Fallback: generate new terrain
        from terrain import generate_terrain
        return generate_terrain(cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT)

def process_events(chronicle_data: list) -> dict:
    """Groups all events from the chronicle by their tick number."""
    events_by_tick = {}
    for event in chronicle_data:
        tick = int(event[0])
        if tick not in events_by_tick:
            events_by_tick[tick] = []
        events_by_tick[tick].append(event)
    return events_by_tick

def get_species_info(genome: np.ndarray):
    """Enhanced species classification with size and color information."""
    w_predate, stress, w_flee, sense_range = genome[8], genome[4], genome[11], genome[12]
    strength, speed = genome[13], genome[14]
    
    # Determine species type
    is_predator = w_predate > cfg.PREDATION_THRESHOLD
    is_anxious = stress > cfg.ANXIETY_THRESHOLD
    
    # Size based on body mass
    size_multiplier = (strength + speed) / 2.0
    
    # Color and name classification
    if is_predator:
        if is_anxious:
            return "Aggressive Hunter", "#FF4444", size_multiplier
        else:
            return "Calm Predator", "#FF8800", size_multiplier
    else:
        if is_anxious:
            return "Anxious Forager", "#FFFF44", size_multiplier
        else:
            return "Peaceful Grazer", "#44FFFF", size_multiplier

class EnhancedOrrery:
    """Enhanced interactive playback system with beautiful terrain and energy visualization."""
    
    def __init__(self, run_number, events_by_tick, terrain_grid):
        self.run_number = run_number
        self.events_by_tick = events_by_tick
        self.terrain_grid = terrain_grid
        self.max_tick = max(events_by_tick.keys()) if events_by_tick else 0
        self.current_tick = 0
        self.entity_states = {}
        self.is_playing = False
        
        # Setup the enhanced UI
        self._setup_ui()
        self._setup_terrain_visualization()
        self._setup_controls()
        self._setup_stats_panel()
        
        # Initialize visualization
        self.update(1)
    
    def _setup_ui(self):
        """Setup the main UI layout."""
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Create sophisticated grid layout
        gs = self.fig.add_gridspec(12, 8, hspace=0.3, wspace=0.2)
        
        # Main simulation view (larger)
        self.ax_main = self.fig.add_subplot(gs[0:10, 0:6])
        
        # Stats panel (right side)
        self.ax_stats = self.fig.add_subplot(gs[0:6, 6:8])
        
        # Species legend (right side, bottom)
        self.ax_legend = self.fig.add_subplot(gs[6:8, 6:8])
        
        # Event log (bottom strip)
        self.ax_events = self.fig.add_subplot(gs[8:10, 6:8])
        
        # Controls (bottom)
        self.ax_slider = self.fig.add_subplot(gs[10, 0:6])
        self.ax_play_btn = self.fig.add_subplot(gs[11, 0:1])
        self.ax_speed_btn = self.fig.add_subplot(gs[11, 1:2])
        
    def _setup_terrain_visualization(self):
        """Setup beautiful terrain visualization."""
        # Configure main display
        self.ax_main.set_facecolor('#000511')
        self.ax_main.set_xlim(0, cfg.WORLD_WIDTH)
        self.ax_main.set_ylim(0, cfg.WORLD_HEIGHT)
        self.ax_main.set_xticks([])
        self.ax_main.set_yticks([])
        self.ax_main.set_title(f'Project Primordium - Evolution #{self.run_number}', 
                              color='white', fontsize=16, fontweight='bold')
        
        # Create terrain colormap
        terrain_colors = ['#1e40af', '#22c55e', '#166534', '#78716c']  # Water, Plains, Forest, Mountains
        terrain_cmap = ListedColormap(terrain_colors)
        terrain_norm = BoundaryNorm([0, 1, 2, 3, 4], terrain_cmap.N)
        
        # Display terrain as base layer
        self.terrain_display = self.ax_main.imshow(
            self.terrain_grid.T, 
            cmap=terrain_cmap, 
            norm=terrain_norm,
            alpha=TERRAIN_ALPHA, 
            zorder=1, 
            interpolation='nearest', 
            origin='lower'
        )
        
        # Energy visualization layer
        self.energy_grid = np.zeros((cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT))
        self.energy_display = self.ax_main.imshow(
            self.energy_grid.T, 
            cmap='plasma', 
            vmin=0, 
            vmax=8, 
            alpha=ENERGY_ALPHA, 
            zorder=2, 
            interpolation='nearest', 
            origin='lower'
        )
        
        # Entity scatter plot
        self.entity_scatter = self.ax_main.scatter(
            [], [], 
            s=[], 
            c=[], 
            alpha=0.9, 
            zorder=3, 
            edgecolors='white', 
            linewidths=0.5
        )
        
        # Add terrain legend
        terrain_legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='#1e40af', label='Water'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#22c55e', label='Plains'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#166534', label='Forest'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#78716c', label='Mountains')
        ]
        
        terrain_legend = self.ax_main.legend(
            handles=terrain_legend_elements,
            loc='upper left',
            bbox_to_anchor=(0.02, 0.98),
            frameon=True,
            facecolor='black',
            edgecolor='gray',
            fontsize=10
        )
        terrain_legend.get_frame().set_alpha(0.8)
        for text in terrain_legend.get_texts():
            text.set_color('white')
    
    def _setup_controls(self):
        """Setup playback controls."""
        # Time slider
        self.slider = Slider(
            ax=self.ax_slider, 
            label='Tick', 
            valmin=1, 
            valmax=self.max_tick, 
            valinit=1, 
            valstep=1, 
            color='#4CAF50'
        )
        self.slider.on_changed(self.update)
        
        # Play/Pause button
        self.btn_play = Button(self.ax_play_btn, 'Play', color='#4CAF50')
        self.btn_play.on_clicked(self.toggle_play)
        
        # Speed control
        self.btn_speed = Button(self.ax_speed_btn, '1x', color='#2196F3')
        self.btn_speed.on_clicked(self.cycle_speed)
        self.speeds = [30, 60, 120, 200]  # milliseconds
        self.current_speed_idx = 0
        
        # Setup animation timer
        self.animation_timer = self.fig.canvas.new_timer(interval=self.speeds[0])
        self.animation_timer.add_callback(self.play_step)
    
    def _setup_stats_panel(self):
        """Setup the statistics and information panels."""
        # Stats panel
        self.ax_stats.set_facecolor('#0a0a0a')
        self.ax_stats.set_xticks([])
        self.ax_stats.set_yticks([])
        self.ax_stats.set_title('Ecosystem Statistics', color='white', fontweight='bold')
        
        self.stats_text = self.ax_stats.text(
            0.05, 0.95, '', 
            color='white', 
            fontsize=11, 
            fontfamily='monospace',
            verticalalignment='top',
            transform=self.ax_stats.transAxes
        )
        
        # Species legend panel
        self.ax_legend.set_facecolor('#0a0a0a')
        self.ax_legend.set_xticks([])
        self.ax_legend.set_yticks([])
        self.ax_legend.set_title('Species Types', color='white', fontweight='bold')
        
        # Events panel
        self.ax_events.set_facecolor('#0a0a0a')
        self.ax_events.set_xticks([])
        self.ax_events.set_yticks([])
        self.ax_events.set_title('Recent Events', color='white', fontweight='bold')
        
        self.events_text = self.ax_events.text(
            0.05, 0.95, '', 
            color='white', 
            fontsize=9, 
            fontfamily='monospace',
            verticalalignment='top',
            transform=self.ax_events.transAxes
        )
    
    def toggle_play(self, event):
        """Toggle play/pause state."""
        if self.is_playing:
            self.is_playing = False
            self.btn_play.label.set_text('Play')
            self.animation_timer.stop()
        else:
            self.is_playing = True
            self.btn_play.label.set_text('Pause')
            self.animation_timer.start()
    
    def cycle_speed(self, event):
        """Cycle through playback speeds."""
        if self.is_playing:
            self.animation_timer.stop()
        
        self.current_speed_idx = (self.current_speed_idx + 1) % len(self.speeds)
        speed = self.speeds[self.current_speed_idx]
        speed_labels = ['1x', '2x', '4x', '8x']
        
        self.btn_speed.label.set_text(speed_labels[self.current_speed_idx])
        self.animation_timer.interval = speed
        
        if self.is_playing:
            self.animation_timer.start()
    
    def play_step(self):
        """Advance one tick during playback."""
        if self.is_playing and self.slider.val < self.max_tick:
            self.slider.set_val(self.slider.val + 1)
        else:
            self.is_playing = False
            self.btn_play.label.set_text('Play')
            self.animation_timer.stop()
    
    def update(self, val):
        """Update visualization for the given tick."""
        target_tick = int(self.slider.val)
        
        # Reset if going backward
        if target_tick < self.current_tick:
            self.entity_states = {}
            self.energy_grid.fill(0)
            start_tick = 1
        else:
            start_tick = self.current_tick + 1
        
        recent_events = []
        
        # Process events from start_tick to target_tick
        for tick in range(start_tick, target_tick + 1):
            # Apply metabolism and aging
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
            
            # Process tick events
            if tick in self.events_by_tick:
                for event in self.events_by_tick[tick]:
                    self._process_event(event, recent_events)
        
        self.current_tick = target_tick
        self._update_visualization()
        self._update_stats_panel()
        self._update_events_panel(recent_events[-10:])  # Show last 10 events
        self.fig.canvas.draw_idle()
    
    def _process_event(self, event, recent_events):
        """Process a single event and update state."""
        _, eid, action, x1, y1, x2, y2, extra = event[:8]
        eid, action = int(eid), int(action)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Log significant events
        event_descriptions = {
            cfg.ACTION_SPAWN: f"Entity {eid} spawned",
            cfg.ACTION_REPRODUCE: f"Entity {eid} reproduced",
            cfg.ACTION_PREDATE: f"Entity {eid} predated",
            cfg.ACTION_DEATH_STARVATION: f"Entity {eid} starved",
            cfg.ACTION_DEATH_AGE: f"Entity {eid} died of age",
            cfg.ACTION_SOLAR_INFUSION: "Solar energy added"
        }
        
        if action in event_descriptions and action != cfg.ACTION_SOLAR_INFUSION:
            recent_events.append(event_descriptions[action])
        
        # Process the event
        if action == cfg.ACTION_SOLAR_INFUSION:
            self.energy_grid[x1, y1] += 1
        elif action == cfg.ACTION_ENERGY_CONSUMED:
            if eid in self.entity_states:
                self.entity_states[eid]['energy'] += cfg.ENERGY_PER_QUANTUM
            if self.energy_grid[x1, y1] > 0:
                self.energy_grid[x1, y1] -= 1
        elif action in [cfg.ACTION_SPAWN, cfg.ACTION_REPRODUCE]:
            child_id = int(extra) if action == cfg.ACTION_REPRODUCE else eid
            genome = np.array(event[8:])
            species_name, color, size_mult = get_species_info(genome)
            
            if action == cfg.ACTION_REPRODUCE and eid in self.entity_states:
                parent_energy = self.entity_states[eid]['energy']
                daughter_energy = (parent_energy - cfg.C_R) * cfg.REPRODUCTION_ENERGY_INHERITANCE
                self.entity_states[eid]['energy'] = (parent_energy - cfg.C_R) * (1 - cfg.REPRODUCTION_ENERGY_INHERITANCE)
                initial_energy = daughter_energy
            else:
                initial_energy = genome[5] + cfg.ENERGY_PER_QUANTUM
            
            self.entity_states[child_id] = {
                'pos': [x2, y2],
                'genome': genome,
                'age': 0,
                'energy': initial_energy,
                'species': species_name,
                'color': color,
                'size_mult': size_mult,
                'mem_strength': 0.0
            }
        elif action == cfg.ACTION_MOVE:
            if eid in self.entity_states:
                self.entity_states[eid]['pos'] = [x2, y2]
        elif action == cfg.ACTION_PREDATE:
            victim_id = int(extra)
            if eid in self.entity_states and victim_id in self.entity_states:
                self.entity_states[eid]['energy'] += self.entity_states[victim_id]['energy'] * cfg.PREDATION_ENERGY_YIELD
                self.entity_states[eid]['pos'] = [x2, y2]
                self.entity_states[eid]['mem_strength'] = 1.0
                del self.entity_states[victim_id]
        elif action in [cfg.ACTION_DEATH_AGE, cfg.ACTION_DEATH_STARVATION]:
            if eid in self.entity_states:
                del self.entity_states[eid]
    
    def _update_visualization(self):
        """Update the main visualization."""
        if not self.entity_states:
            self.entity_scatter.set_offsets(np.empty((0, 2)))
            self.entity_scatter.set_sizes([])
            self.entity_scatter.set_color([])
        else:
            # Prepare entity data
            positions = np.array([s['pos'] for s in self.entity_states.values()])
            colors = [s['color'] for s in self.entity_states.values()]
            sizes = [ENTITY_SIZE_BASE * s['size_mult'] for s in self.entity_states.values()]
            
            # Update scatter plot
            self.entity_scatter.set_offsets(positions)
            self.entity_scatter.set_sizes(sizes)
            self.entity_scatter.set_color(colors)
        
        # Update energy display
        self.energy_display.set_data(self.energy_grid.T)
        self.energy_display.set_clim(0, max(1, np.max(self.energy_grid)))
    
    def _update_stats_panel(self):
        """Update the statistics panel."""
        if not self.entity_states:
            self.stats_text.set_text(f"TICK: {self.current_tick}/{self.max_tick}\n\nEXTINCTION EVENT")
            return
        
        # Calculate statistics
        pop_count = len(self.entity_states)
        genomes = np.array([s['genome'] for s in self.entity_states.values()])
        ages = np.array([s['age'] for s in self.entity_states.values()])
        energies = np.array([s['energy'] for s in self.entity_states.values()])
        
        # Species counts
        species_counts = {}
        for state in self.entity_states.values():
            species = state['species']
            species_counts[species] = species_counts.get(species, 0) + 1
        
        # Build stats string
        stats_lines = [
            f"TICK: {self.current_tick}/{self.max_tick}",
            "",
            "[ECOSYSTEM]",
            f"Population:         {pop_count}",
            f"Environmental Energy: {int(np.sum(self.energy_grid))}",
            "",
            "[SPECIES DISTRIBUTION]"
        ]
        
        for species, count in sorted(species_counts.items()):
            percentage = (count / pop_count) * 100
            stats_lines.append(f"{species}: {count} ({percentage:.1f}%)")
        
        stats_lines.extend([
            "",
            "[AVERAGES]",
            f"Age:                {np.mean(ages):.1f}",
            f"Energy:             {np.mean(energies):.1f}",
            f"Body Strength:      {np.mean(genomes[:, 13]):.2f}",
            f"Body Speed:         {np.mean(genomes[:, 14]):.2f}",
            f"Sense Range:        {np.mean(genomes[:, 12]):.2f}",
            "",
            "[EXTREMES]",
            f"Oldest:             {np.max(ages)}",
            f"Richest Energy:     {np.max(energies):.1f}",
            f"Strongest:          {np.max(genomes[:, 13]):.2f}",
            f"Fastest:            {np.max(genomes[:, 14]):.2f}"
        ])
        
        self.stats_text.set_text('\n'.join(stats_lines))
    
    def _update_events_panel(self, recent_events):
        """Update the events panel."""
        if recent_events:
            events_text = '\n'.join(recent_events)
        else:
            events_text = "No recent events"
        
        self.events_text.set_text(events_text)
    
    def run(self):
        """Start the enhanced visualization."""
        # Create species legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4444', markersize=10, label='Aggressive Hunter'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF8800', markersize=10, label='Calm Predator'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFFF44', markersize=10, label='Anxious Forager'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#44FFFF', markersize=10, label='Peaceful Grazer')
        ]
        
        legend = self.ax_legend.legend(
            handles=legend_elements, 
            loc='center',
            frameon=True,
            facecolor='#1a1a1a',
            edgecolor='gray'
        )
        
        for text in legend.get_texts():
            text.set_color('white')
        
        plt.tight_layout()
        plt.show()

def main(run_number):
    """Main function to launch enhanced playback."""
    print(f"Loading enhanced visualization for run {run_number}...")
    
    chronicle = load_chronicle(run_number)
    if not chronicle:
        print("Failed to load chronicle data.")
        return
    
    terrain = load_terrain(run_number)
    events = process_events(chronicle)
    
    print(f"Loaded {len(chronicle)} events across {len(events)} ticks")
    print("Starting enhanced Orrery visualization...")
    
    orrery = EnhancedOrrery(run_number, events, terrain)
    orrery.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Project Primordium Visualization")
    parser.add_argument("run_number", type=int, help="The run number to visualize")
    args = parser.parse_args()
    main(args.run_number)