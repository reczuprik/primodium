import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
import argparse
import primordium_engine.config as cfg
import os
import csv
from matplotlib.lines import Line2D
from primordium_engine.terrain import get_terrain_color, TERRAIN_WATER, TERRAIN_PLAINS, TERRAIN_FOREST, TERRAIN_MOUNTAINS, generate_terrain

# Enhanced visualization constants
ENTITY_SIZE_BASE = 12
ENERGY_ALPHA = 0.6
TERRAIN_ALPHA = 0.5
ANIMATION_SPEED = 50

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
    """Load terrain for visualization with improved fallback handling."""
    terrain_file = os.path.join("chronicles", f"run_{run_number}_terrain.npy")
    
    try:
        if os.path.exists(terrain_file):
            print(f"Loading terrain from {terrain_file}")
            terrain = np.load(terrain_file)
            
            # Verify terrain distribution
            unique, counts = np.unique(terrain, return_counts=True)
            terrain_names = ["Water", "Plains", "Forest", "Mountains"]
            print("Terrain distribution:")
            for terrain_type, count in zip(unique, counts):
                percentage = (count / terrain.size) * 100
                print(f"  {terrain_names[terrain_type]}: {count} cells ({percentage:.1f}%)")
            
            return terrain
    except Exception as e:
        print(f"Warning: Could not load terrain file: {e}")
    
    # Fallback: generate new terrain
    print("Generating new terrain for visualization...")
    terrain = generate_terrain(cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT, seed=run_number * 42)
    
    # Try to save the generated terrain for future use
    try:
        os.makedirs("chronicles", exist_ok=True)
        np.save(terrain_file, terrain)
        print(f"Generated terrain saved to {terrain_file}")
    except Exception as e:
        print(f"Warning: Could not save generated terrain: {e}")
    
    return terrain

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
    
    # Size based on body mass (more visible differences)
    size_multiplier = 0.8 + (strength + speed) / 3.0
    
    # Color and name classification
    if is_predator:
        if is_anxious:
            return "Aggressive Hunter", "#FF2222", size_multiplier
        else:
            return "Calm Predator", "#FF6600", size_multiplier
    else:
        if is_anxious:
            return "Anxious Forager", "#FFDD00", size_multiplier
        else:
            return "Peaceful Grazer", "#00DDDD", size_multiplier

class EvolutionAnalyzer:
    """Analyzer to track species evolution over time."""
    
    def __init__(self):
        self.species_history = {
            "Aggressive Hunter": [],
            "Calm Predator": [],
            "Anxious Forager": [],
            "Peaceful Grazer": []
        }
        self.tick_history = []
        self.population_history = []
        self.energy_history = []
    
    def record_tick(self, tick, entity_states, energy_grid):
        """Record data for this tick."""
        self.tick_history.append(tick)
        self.population_history.append(len(entity_states))
        self.energy_history.append(np.sum(energy_grid))
        
        # Count species
        species_counts = {species: 0 for species in self.species_history.keys()}
        for state in entity_states.values():
            species = state['species']
            if species in species_counts:
                species_counts[species] += 1
        
        # Record counts
        for species, count in species_counts.items():
            self.species_history[species].append(count)

class EnhancedOrrery:
    """Enhanced interactive playback system with evolutionary analysis."""
    
    def __init__(self, run_number, events_by_tick, terrain_grid):
        self.run_number = run_number
        self.events_by_tick = events_by_tick
        self.terrain_grid = terrain_grid
        self.max_tick = max(events_by_tick.keys()) if events_by_tick else 0
        self.current_tick = 0
        self.entity_states = {}
        self.is_playing = False
        self.analyzer = EvolutionAnalyzer()
        
        # Setup the enhanced UI
        self._setup_ui()
        self._setup_terrain_visualization()
        self._setup_controls()
        self._setup_stats_panel()
        self._setup_evolution_charts()
        
        # Initialize visualization
        self.update(1)
    
    def _setup_ui(self):
        """Setup the main UI layout with better spacing."""
        self.fig = plt.figure(figsize=(24, 14))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Create sophisticated grid layout - FIXED overlapping
        gs = self.fig.add_gridspec(14, 12, hspace=0.4, wspace=0.3)
        
        # Main simulation view (larger, left side)
        self.ax_main = self.fig.add_subplot(gs[0:10, 0:7])
        
        # Stats panel (top right)
        self.ax_stats = self.fig.add_subplot(gs[0:5, 7:10])
        
        # Species legend (middle right)
        self.ax_legend = self.fig.add_subplot(gs[5:7, 7:10])
        
        # Event log (bottom right)
        self.ax_events = self.fig.add_subplot(gs[7:10, 7:10])
        
        # Evolution charts (right side, large)
        self.ax_species_chart = self.fig.add_subplot(gs[0:7, 10:12])
        self.ax_ecosystem_chart = self.fig.add_subplot(gs[7:10, 10:12])
        
        # Controls (bottom, spanning width)
        self.ax_slider = self.fig.add_subplot(gs[10, 0:10])
        self.ax_play_btn = self.fig.add_subplot(gs[11, 0:1])
        self.ax_speed_btn = self.fig.add_subplot(gs[11, 1:2])
        self.ax_reset_btn = self.fig.add_subplot(gs[11, 2:3])
        
    def _setup_terrain_visualization(self):
        """Setup beautiful terrain visualization."""
        # Configure main display
        self.ax_main.set_facecolor('#000511')
        self.ax_main.set_xlim(0, cfg.WORLD_WIDTH)
        self.ax_main.set_ylim(0, cfg.WORLD_HEIGHT)
        self.ax_main.set_xticks([])
        self.ax_main.set_yticks([])
        self.ax_main.set_title(f'Project Primordium - Evolution #{self.run_number}', 
                              color='white', fontsize=18, fontweight='bold', pad=20)
        
        # Create terrain colormap with better colors
        terrain_colors = ['#1e40af', '#22c55e', '#15803d', '#78716c']  # Water, Plains, Forest, Mountains
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
            interpolation='bilinear', 
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
            linewidths=0.8
        )
        
        # Add terrain legend to main plot
        terrain_legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='#1e40af', label='Water'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#22c55e', label='Plains'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#15803d', label='Forest'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#78716c', label='Mountains')
        ]
        
        terrain_legend = self.ax_main.legend(
            handles=terrain_legend_elements,
            loc='upper left',
            bbox_to_anchor=(0.02, 0.98),
            frameon=True,
            facecolor='black',
            edgecolor='gray',
            fontsize=11
        )
        terrain_legend.get_frame().set_alpha(0.9)
        for text in terrain_legend.get_texts():
            text.set_color('white')
    
    def _setup_evolution_charts(self):
        """Setup evolutionary tracking charts."""
        # Species evolution chart
        self.ax_species_chart.set_facecolor('#0a0a0a')
        self.ax_species_chart.set_title('Species Evolution Over Time', color='white', fontweight='bold')
        self.ax_species_chart.set_xlabel('Tick', color='white')
        self.ax_species_chart.set_ylabel('Population', color='white')
        self.ax_species_chart.tick_params(colors='white')
        
        # Ecosystem health chart  
        self.ax_ecosystem_chart.set_facecolor('#0a0a0a')
        self.ax_ecosystem_chart.set_title('Ecosystem Health', color='white', fontweight='bold')
        self.ax_ecosystem_chart.set_xlabel('Tick', color='white')
        self.ax_ecosystem_chart.set_ylabel('Total Energy', color='white')
        self.ax_ecosystem_chart.tick_params(colors='white')
        
        # Initialize empty plots
        self.species_lines = {}
        colors = ['#FF2222', '#FF6600', '#FFDD00', '#00DDDD']
        for i, species in enumerate(self.analyzer.species_history.keys()):
            line, = self.ax_species_chart.plot([], [], color=colors[i], label=species, linewidth=2)
            self.species_lines[species] = line
        
        self.ax_species_chart.legend(loc='upper right', facecolor='#1a1a1a', edgecolor='gray')
        for text in self.ax_species_chart.get_legend().get_texts():
            text.set_color('white')
        
        # Ecosystem line
        self.ecosystem_line, = self.ax_ecosystem_chart.plot([], [], color='#00FF00', linewidth=2)
    
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
        self.speeds = [20, 50, 100, 200]  # milliseconds
        self.current_speed_idx = 1
        
        # Reset button
        self.btn_reset = Button(self.ax_reset_btn, 'Reset', color='#FF5722')
        self.btn_reset.on_clicked(self.reset_view)
        
        # Setup animation timer
        self.animation_timer = self.fig.canvas.new_timer(interval=self.speeds[self.current_speed_idx])
        self.animation_timer.add_callback(self.play_step)
    
    def _setup_stats_panel(self):
        """Setup the statistics and information panels."""
        # Stats panel
        self.ax_stats.set_facecolor('#0a0a0a')
        self.ax_stats.set_xticks([])
        self.ax_stats.set_yticks([])
        self.ax_stats.set_title('Ecosystem Statistics', color='white', fontweight='bold', fontsize=14)
        
        self.stats_text = self.ax_stats.text(
            0.05, 0.95, '', 
            color='white', 
            fontsize=10, 
            fontfamily='monospace',
            verticalalignment='top',
            transform=self.ax_stats.transAxes
        )
        
        # Species legend panel
        self.ax_legend.set_facecolor('#0a0a0a')
        self.ax_legend.set_xticks([])
        self.ax_legend.set_yticks([])
        self.ax_legend.set_title('Species Types', color='white', fontweight='bold', fontsize=12)
        
        # Events panel
        self.ax_events.set_facecolor('#0a0a0a')
        self.ax_events.set_xticks([])
        self.ax_events.set_yticks([])
        self.ax_events.set_title('Recent Events', color='white', fontweight='bold', fontsize=12)
        
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
        speed_labels = ['4x', '2x', '1x', '0.5x']
        
        self.btn_speed.label.set_text(speed_labels[self.current_speed_idx])
        self.animation_timer.interval = speed
        
        if self.is_playing:
            self.animation_timer.start()
    
    def reset_view(self, event):
        """Reset view to beginning."""
        self.slider.set_val(1)
        if self.is_playing:
            self.toggle_play(event)
    
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
            # Reset analyzer
            self.analyzer = EvolutionAnalyzer()
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
            
            # Record data every 10 ticks for performance
            if tick % 10 == 0:
                self.analyzer.record_tick(tick, self.entity_states, self.energy_grid)
        
        self.current_tick = target_tick
        self._update_visualization()
        self._update_stats_panel()
        self._update_events_panel(recent_events[-10:])
        self._update_evolution_charts()
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
                initial_energy = genome[4] * cfg.INITIAL_ENERGY_MULTIPLIER + cfg.ENERGY_PER_QUANTUM
            
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
        
        # Update energy display with better scaling
        if np.max(self.energy_grid) > 0:
            self.energy_display.set_data(self.energy_grid.T)
            self.energy_display.set_clim(0, max(2, np.max(self.energy_grid) * 0.8))
    
    def _update_stats_panel(self):
        """Update the statistics panel."""
        if not self.entity_states:
            self.stats_text.set_text(f"TICK: {self.current_tick}/{self.max_tick}\n\n⚠️ EXTINCTION EVENT")
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
        
        # Calculate predator percentage
        predator_count = species_counts.get("Aggressive Hunter", 0) + species_counts.get("Calm Predator", 0)
        predator_percentage = (predator_count / pop_count) * 100 if pop_count > 0 else 0
        
        # Build stats string with better formatting
        stats_lines = [
            f"TICK: {self.current_tick}/{self.max_tick}",
            "",
            "🌍 [ECOSYSTEM]",
            f"Population:         {pop_count:,}",
            f"Environmental Energy: {int(np.sum(self.energy_grid)):,}",
            f"Predator %:         {predator_percentage:.1f}%",
            "",
            "🧬 [SPECIES COUNTS]"
        ]
        
        species_icons = {
            "Aggressive Hunter": "🔴",
            "Calm Predator": "🟠", 
            "Anxious Forager": "🟡",
            "Peaceful Grazer": "🔵"
        }
        
        for species, count in sorted(species_counts.items()):
            icon = species_icons.get(species, "⚪")
            percentage = (count / pop_count) * 100
            stats_lines.append(f"{icon} {species}: {count} ({percentage:.1f}%)")
        
        stats_lines.extend([
            "",
            "📊 [AVERAGES]",
            f"Age:                {np.mean(ages):.1f}",
            f"Energy:             {np.mean(energies):.1f}",
            f"Body Strength:      {np.mean(genomes[:, 13]):.2f}",
            f"Body Speed:         {np.mean(genomes[:, 14]):.2f}",
            f"Sense Range:        {np.mean(genomes[:, 12]):.2f}",
            "",
            "🏆 [EXTREMES]",
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
    
    def _update_evolution_charts(self):
        """Update the evolutionary tracking charts."""
        if len(self.analyzer.tick_history) < 2:
            return
        
        ticks = np.array(self.analyzer.tick_history)
        
        # Update species evolution chart
        for species, line in self.species_lines.items():
            counts = np.array(self.analyzer.species_history[species])
            line.set_data(ticks, counts)
        
        # Update chart limits
        if len(ticks) > 0:
            self.ax_species_chart.set_xlim(0, max(ticks))
            max_pop = max([max(counts) if counts else 0 for counts in self.analyzer.species_history.values()])
            self.ax_species_chart.set_ylim(0, max(max_pop * 1.1, 10))
        
        # Update ecosystem health chart
        energies = np.array(self.analyzer.energy_history)
        self.ecosystem_line.set_data(ticks, energies)
        
        if len(energies) > 0:
            self.ax_ecosystem_chart.set_xlim(0, max(ticks))
            self.ax_ecosystem_chart.set_ylim(0, max(energies) * 1.1)
    
    def run(self):
        """Start the enhanced visualization."""
        # Create species legend with better formatting
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF2222', markersize=12, label='🔴 Aggressive Hunter'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6600', markersize=12, label='🟠 Calm Predator'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFDD00', markersize=12, label='🟡 Anxious Forager'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#00DDDD', markersize=12, label='🔵 Peaceful Grazer')
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

def generate_evolution_summary(run_number):
    """Generate a summary chart showing the complete evolution."""
    chronicle = load_chronicle(run_number)
    if not chronicle:
        return
    
    events = process_events(chronicle)
    analyzer = EvolutionAnalyzer()
    
    print("Analyzing complete evolution...")
    
    # Process all events to build complete history
    entity_states = {}
    energy_grid = np.zeros((cfg.WORLD_WIDTH, cfg.WORLD_HEIGHT))
    
    for tick in sorted(events.keys()):
        # Simple metabolism simulation
        for eid in list(entity_states.keys()):
            state = entity_states[eid]
            state['age'] += 1
            state['energy'] -= 5  # Simple metabolism
            if state['energy'] <= 0 or state['age'] >= 500:
                del entity_states[eid]
        
        # Process events
        for event in events[tick]:
            _, eid, action, x1, y1, x2, y2, extra = event[:8]
            eid, action = int(eid), int(action)
            
            if action == cfg.ACTION_SOLAR_INFUSION:
                energy_grid[int(x1), int(y1)] += 1
            elif action == cfg.ACTION_ENERGY_CONSUMED:
                if eid in entity_states:
                    entity_states[eid]['energy'] += cfg.ENERGY_PER_QUANTUM
                if energy_grid[int(x1), int(y1)] > 0:
                    energy_grid[int(x1), int(y1)] -= 1
            elif action in [cfg.ACTION_SPAWN, cfg.ACTION_REPRODUCE]:
                child_id = int(extra) if action == cfg.ACTION_REPRODUCE else eid
                genome = np.array(event[8:])
                species_name, color, size_mult = get_species_info(genome)
                entity_states[child_id] = {
                    'genome': genome,
                    'age': 0,
                    'energy': 100,
                    'species': species_name
                }
        
        # Record every 50 ticks
        if tick % 50 == 0:
            analyzer.record_tick(tick, entity_states, energy_grid)
    
    # Create summary chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor('#0a0a0a')
    
    ticks = np.array(analyzer.tick_history)
    colors = ['#FF2222', '#FF6600', '#FFDD00', '#00DDDD']
    
    # Species evolution
    ax1.set_facecolor('#0a0a0a')
    ax1.set_title(f'Evolution Summary - Run #{run_number}', color='white', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Tick', color='white')
    ax1.set_ylabel('Population', color='white')
    ax1.tick_params(colors='white')
    
    for i, (species, counts) in enumerate(analyzer.species_history.items()):
        if any(c > 0 for c in counts):  # Only plot species that existed
            ax1.plot(ticks, counts, color=colors[i], label=species, linewidth=2, marker='o', markersize=3)
    
    ax1.legend(facecolor='#1a1a1a', edgecolor='gray')
    for text in ax1.get_legend().get_texts():
        text.set_color('white')
    ax1.grid(True, alpha=0.3)
    
    # Total population and energy
    ax2.set_facecolor('#0a0a0a')
    ax2.set_xlabel('Tick', color='white')
    ax2.set_ylabel('Total Population', color='white')
    ax2.tick_params(colors='white')
    
    ax2.plot(ticks, analyzer.population_history, color='#00FF00', label='Total Population', linewidth=2)
    
    # Secondary y-axis for energy
    ax2_energy = ax2.twinx()
    ax2_energy.plot(ticks, analyzer.energy_history, color='#FFAA00', label='Environmental Energy', linewidth=2, linestyle='--')
    ax2_energy.set_ylabel('Environmental Energy', color='white')
    ax2_energy.tick_params(colors='white')
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_energy.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, facecolor='#1a1a1a', edgecolor='gray')
    for text in ax2.get_legend().get_texts():
        text.set_color('white')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print final statistics
    if analyzer.population_history:
        final_pop = analyzer.population_history[-1]
        final_species = {species: counts[-1] if counts else 0 for species, counts in analyzer.species_history.items()}
        
        print(f"\n=== EVOLUTION SUMMARY - RUN #{run_number} ===")
        print(f"Final Population: {final_pop}")
        print("Final Species Distribution:")
        for species, count in final_species.items():
            if count > 0:
                percentage = (count / final_pop) * 100 if final_pop > 0 else 0
                print(f"  {species}: {count} ({percentage:.1f}%)")
        
        # Check for predator extinction
        predator_count = final_species.get("Aggressive Hunter", 0) + final_species.get("Calm Predator", 0)
        if predator_count == 0:
            print("\n⚠️  PREDATOR EXTINCTION DETECTED!")
            print("   Possible causes:")
            print("   - High movement costs in terrain")
            print("   - Insufficient energy yield from predation")
            print("   - Reproduction costs too high for predators")

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
    
    # Ask user what they want to do
    print("\nVisualization Options:")
    print("1. Interactive playback (default)")
    print("2. Evolution summary charts")
    print("3. Both")
    
    try:
        choice = input("Enter choice (1-3, or press Enter for 1): ").strip()
        if not choice:
            choice = "1"
    except:
        choice = "1"
    
    if choice in ["1", "3"]:
        print("Starting interactive Orrery visualization...")
        orrery = EnhancedOrrery(run_number, events, terrain)
        orrery.run()
    
    if choice in ["2", "3"]:
        print("Generating evolution summary...")
        generate_evolution_summary(run_number)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Project Primordium Visualization")
    parser.add_argument("run_number", type=int, help="The run number to visualize")
    args = parser.parse_args()
    main(args.run_number)