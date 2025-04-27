import matplotlib.pyplot as plt
import numpy as np
import json
import time
import os
from collections import defaultdict
from rich.console import Console
import matplotlib.colors as mcolors
import math
import networkx as nx
import traceback
from matplotlib.widgets import Button
import matplotlib.patheffects as pe
import sys
import signal
from mpl_toolkits.mplot3d.art3d import Line3DCollection # Import this for efficient edge drawing

# Assuming config.py exists
try:
    import config
except ImportError:
    print("Error: config.py not found. Please ensure it exists and defines necessary variables.")
    # Define fallback defaults if config is missing
    class ConfigFallback:
        VIZ_POSITIONS_FILE = "positions.json"
        VIZ_PATHS_FILE = "paths.jsonl"
        NUM_CUBES = 24 # Example value matching your screenshot
        EXIT_TOKEN_INDEX = -1 # Example value
    config = ConfigFallback()
    print(f"Using fallback config: NUM_CUBES={config.NUM_CUBES}, EXIT_TOKEN_INDEX={config.EXIT_TOKEN_INDEX}")


console = Console()

# --- Configuration for Visualization ---
UPDATE_INTERVAL_SEC = 1 # Check file more often
LAYOUT_UPDATE_INTERVAL_BATCHES = 50 # Recompute layout more often initially
PLOT_UPDATE_INTERVAL_BATCHES = 1 # Redraw plot immediately after each batch
POSITION_FILE = config.VIZ_POSITIONS_FILE
PATHS_FILE = config.VIZ_PATHS_FILE
NUM_CUBES = config.NUM_CUBES
EXIT_TOKEN_INDEX = config.EXIT_TOKEN_INDEX
# MIN_EDGE_ALPHA removed: now alpha can be 0 for rarely used edges
# --- Tuned values for better visual ---
TEXT_OFFSET = 0.0005 # Further reduced offset based on the screenshot scale
TEXT_FONTSIZE = 8 # Slightly smaller font
TEXT_OUTLINE_LINEWIDTH = 1.0
LAYOUT_ITERATIONS = 250 # More iterations for potentially better layout
LAYOUT_K_FACTOR = 0.3 # Adjustment factor for k (tweak if needed)
EDGE_LINEWIDTH_SCALE = 0.7 # Factor to scale linewidth based on log count
HIGHLIGHT_LINEWIDTH_SCALE = 1.5 # Factor for highlighted edges
# --- End Config ---

# Global flag to control main loop
keep_running = True

# Global storage for initial plot limits and view
initial_xlim = None
initial_ylim = None
initial_zlim = None
initial_elev = 30 # Default view angle
initial_azim = -60 # Default view angle


def on_close(event):
    """Handler for the plot window close event."""
    global keep_running
    console.print("\n[yellow]Plot window closed. Stopping visualization script.[/yellow]")
    keep_running = False

def handle_sigint(sig, frame):
    """Handler for Ctrl+C interrupt"""
    global keep_running
    console.print("\n[bold yellow]Visualization script interrupted by user (Ctrl+C). Closing plot.[/]")
    keep_running = False

signal.signal(signal.SIGINT, handle_sigint)

def load_cube_positions(filepath):
    """Loads original cube positions from a JSON file or returns random if invalid."""
    if not os.path.exists(filepath):
        console.print(f"[yellow]Position file {filepath} not found. Using random initial positions.[/yellow]")
        # Use positions roughly centered around 0, with a small spread, similar to the screenshot scale
        scale = 0.01 # Example scale
        return (np.random.rand(NUM_CUBES, 3) - 0.5) * scale
    try:
        with open(filepath, 'r') as f:
            positions_list = json.load(f)
            # Ensure it's a list of lists/tuples with 3 coordinates
            if not isinstance(positions_list, list) or not all(isinstance(p, (list, tuple)) and len(p) == 3 for p in positions_list):
                 console.print(f"[red]Position file {filepath} has invalid format. Expected list of [x,y,z]. Using random positions.[/red]")
                 scale = 0.01
                 return (np.random.rand(NUM_CUBES, 3) - 0.5) * scale

            if len(positions_list) != NUM_CUBES:
                 console.print(f"[red]Position file {filepath} has {len(positions_list)} entries, expected {NUM_CUBES}. Using random positions.[/red]")
                 scale = 0.01
                 return (np.random.rand(NUM_CUBES, 3) - 0.5) * scale

            console.print(f"[green]Successfully loaded {len(positions_list)} positions from {filepath}[/green]")
            return np.array(positions_list, dtype=float)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error decoding JSON from position file {filepath}: {e}. Using random positions.[/red]")
        scale = 0.01
        return (np.random.rand(NUM_CUBES, 3) - 0.5) * scale
    except Exception as e:
        console.print(f"[red]Error loading position file {filepath}: {e}. Using random positions.[/red]")
        scale = 0.01
        return (np.random.rand(NUM_CUBES, 3) - 0.5) * scale


def read_path_data_incremental(filepath, last_pos):
    """Reads new lines from the paths file since the last read position."""
    new_data_entries = []
    last_transitions_in_new_block = []
    current_pos = last_pos

    if not os.path.exists(filepath):
        return [], last_pos, []

    try:
        with open(filepath, 'r') as f:
            f.seek(last_pos)
            new_lines = f.readlines()
            current_pos = f.tell()

            if new_lines:
                 valid_lines_count = 0
                 last_valid_data = None
                 for line in new_lines:
                      line = line.strip()
                      if line:
                          try:
                              data = json.loads(line)
                              if isinstance(data, dict) and "transitions" in data and isinstance(data["transitions"], list):
                                  new_data_entries.append(data)
                                  valid_lines_count += 1
                                  last_valid_data = data # Keep track of the last successfully parsed entry
                              else:
                                  console.print(f"[yellow]Skipping line with unexpected structure: {line[:100]}...[/yellow]")
                          except json.JSONDecodeError as e:
                              console.print(f"[yellow]Skipping malformed JSON line: {line[:100]}... - {e}[/yellow]")

                 if last_valid_data:
                     # Filter out exits for edge highlighting from the last valid entry
                     last_transitions_in_new_block = [tuple(t) for t in last_valid_data.get("transitions", []) if len(t)==2 and t[0] != EXIT_TOKEN_INDEX and t[1] != EXIT_TOKEN_INDEX]

    except FileNotFoundError:
        return [], last_pos, []
    except Exception as e:
        console.print(f"[red]Error reading paths file {filepath}: {e}[/red]")
        print(traceback.format_exc())
        return [], last_pos, []

    return new_data_entries, current_pos, last_transitions_in_new_block


def compute_graph_layout(transition_counts, num_cubes, existing_pos=None):
    """Computes a 3D force-directed layout based on transition counts."""
    G = nx.DiGraph()
    nodes = list(range(num_cubes))
    G.add_nodes_from(nodes)

    if not transition_counts:
         # console.print("[yellow]No transitions yet, cannot compute edge-based layout. Keeping existing positions.[/yellow]")
         return existing_pos # Keep current positions if no edges

    edge_list_with_weights = []
    for (from_idx, to_idx), count in transition_counts.items():
        if 0 <= from_idx < num_cubes and 0 <= to_idx < num_cubes:
             # Using log1p encourages separation even for low counts
             weight = math.log1p(count)
             edge_list_with_weights.append((from_idx, to_idx, {'weight': weight}))


    if not edge_list_with_weights:
        # console.print("[yellow]No valid transitions between existing nodes found. Keeping existing positions.[/yellow]")
        return existing_pos

    G.add_edges_from(edge_list_with_weights)

    pos_dict = None
    if existing_pos is not None and isinstance(existing_pos, np.ndarray) and existing_pos.shape == (num_cubes, 3):
         pos_dict = {i: tuple(existing_pos[i]) for i in range(num_cubes)}

    k_val = LAYOUT_K_FACTOR / math.sqrt(G.number_of_nodes()) if G.number_of_nodes() > 0 else None
    k_val = max(k_val, 0.001) # Ensure k is not too small

    console.log(f"Computing 3D layout: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges. k={k_val:.4f}, iters={LAYOUT_ITERATIONS}.")
    try:
        pos = nx.spring_layout(
            G,
            dim=3,
            k=k_val,
            pos=pos_dict,
            iterations=LAYOUT_ITERATIONS,
            weight='weight',
            # seed=42 # Optional: uncomment for deterministic layout if starting from scratch
        )
        pos_array = np.array([pos[i] for i in range(num_cubes)])
        # Blend new layout with existing positions to preserve overall spread
        if existing_pos is not None and isinstance(existing_pos, np.ndarray):
            blend = 0.3  # fraction of new layout influence
            pos_array = existing_pos * (1 - blend) + pos_array * blend

        # Check for NaN/Inf in computed positions
        if np.any(~np.isfinite(pos_array)):
             console.print("[red]Layout computation resulted in non-finite values. Keeping previous positions.[/red]")
             return existing_pos

        console.log("Layout computed.")
        return pos_array
    except Exception as e:
        console.print(f"[red]Error computing graph layout: {e}. Using previous positions.[/red]")
        print(traceback.format_exc())
        return existing_pos


def set_initial_limits_and_view(ax, node_positions):
    """Sets initial axis limits and view based on node positions and stores them globally."""
    global initial_xlim, initial_ylim, initial_zlim, initial_elev, initial_azim

    if node_positions is None or len(node_positions) == 0:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
    else:
        min_coords = node_positions.min(axis=0)
        max_coords = node_positions.max(axis=0)
        center = (min_coords + max_coords) / 2
        span = (max_coords - min_coords)
        # Ensure span is not zero, add a minimum size relative to typical scale
        if np.all(span < 1e-9): # If all nodes are at virtually the same point
             span = np.array([0.1, 0.1, 0.1]) # Set a fixed range

        margin = span * 0.20 # Use a 20% margin

        # Calculate new limits based on center, span, and margin
        new_xlim = (center[0] - span[0]/2 - margin[0], center[0] + span[0]/2 + margin[0])
        new_ylim = (center[1] - span[1]/2 - margin[1], center[1] + span[1]/2 + margin[1])
        new_zlim = (center[2] - span[2]/2 - margin[2], center[2] + span[2]/2 + margin[2])

        # Apply limits
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        ax.set_zlim(new_zlim)

    # Store these limits
    initial_xlim = ax.get_xlim3d()
    initial_ylim = ax.get_ylim3d()
    initial_zlim = ax.get_zlim3d()
    console.log(f"Set initial plot limits: X={initial_xlim}, Y={initial_ylim}, Z={initial_zlim}")

    # Set initial view
    ax.view_init(elev=initial_elev, azim=initial_azim)


def update_plot_with_collection(fig, ax, node_positions, transition_counts, max_count, last_path_transitions, sm, cbar):
    """Updates the 3D plot using Line3DCollection for edges."""
    global initial_xlim, initial_ylim, initial_zlim

    # Store current view before clearing
    current_elev = ax.elev
    current_azim = ax.azim
    # Store current limits before clearing (if they exist)
    current_xlim = ax.get_xlim3d() if initial_xlim is not None else None
    current_ylim = ax.get_ylim3d() if initial_ylim is not None else None
    current_zlim = ax.get_zlim3d() if initial_zlim is not None else None


    ax.cla() # Clear axes - necessary if layout changes or number of nodes changes

    if node_positions is None or len(node_positions) != NUM_CUBES:
        ax.set_title('Waiting for valid node positions...')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        # Restore limits/view if possible
        if initial_xlim is not None:
            ax.set_xlim3d(initial_xlim); ax.set_ylim3d(initial_ylim); ax.set_zlim3d(initial_zlim)
            ax.view_init(elev=current_elev, azim=current_azim)
        fig.canvas.draw_idle()
        return # Nothing more to draw

    # --- Draw Static Elements (Nodes and Labels) ---
    node_colors = ['red' if i == 0 else 'blue' for i in range(NUM_CUBES)]
    ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
               c=node_colors, s=50, alpha=0.7, depthshade=True)

    for i in range(NUM_CUBES):
         p = node_positions[i]
         ax.text(p[0] + TEXT_OFFSET, p[1] + TEXT_OFFSET, p[2] + TEXT_OFFSET, str(i),
                 color='white',
                 fontsize=TEXT_FONTSIZE,
                 ha='left', va='bottom', zorder=5,
                 path_effects=[pe.Stroke(linewidth=TEXT_OUTLINE_LINEWIDTH, foreground='black'), pe.Normal()])

    # --- Prepare and Draw Dynamic Elements (Edges) using Line3DCollection ---
    segments = []
    colors = []
    linewidths = []
    cmap = plt.get_cmap('hot')

    # Update colorbar normalization based on log scale
    # Use max_count for raw value display, log1p(max_count) for color mapping
    max_log_count = math.log1p(max_count) if max_count > 0 else 1.0 # Avoid division by zero or log(0)
    new_norm = mcolors.Normalize(vmin=0, vmax=max_log_count)
    sm.set_norm(new_norm)
    cbar.mappable.set_norm(new_norm)
    cbar.set_label(f'log(1 + Transition Frequency) [Max Raw: {max_count}]')


    for (from_idx, to_idx), count in transition_counts.items():
        if 0 <= from_idx < NUM_CUBES and 0 <= to_idx < NUM_CUBES:
            p_from = node_positions[from_idx]
            p_to = node_positions[to_idx]
            log_count = math.log1p(count)

            segments.append([p_from, p_to])

            is_in_last_path = (from_idx, to_idx) in last_path_transitions

            if is_in_last_path:
                 rgba = mcolors.to_rgba('lime', alpha=1.0)
                 colors.append(rgba)
                 linewidth = min(0.5 + log_count * HIGHLIGHT_LINEWIDTH_SCALE, 6.0) # Thicker, capped
            else:
                 norm_log_count = new_norm(log_count)
                 base_rgba = cmap(norm_log_count)
                 rgba = (base_rgba[0], base_rgba[1], base_rgba[2], norm_log_count)
                 colors.append(rgba)
                 linewidth = min(0.5 + log_count * EDGE_LINEWIDTH_SCALE, 4.0) # Normal width, capped

            linewidths.append(linewidth)

    if segments: # Only add collection if there are edges
        # Line3DCollection expects numpy array segments of shape (N, 2, 3)
        edge_collection = Line3DCollection(segments, colors=colors, linewidths=linewidths, picker=5)
        ax.add_collection3d(edge_collection)


    # --- Set Axes Properties ---
    ax.set_title(f'Spatial Cube Transitions (Unique Edges: {len(transition_counts)})')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    # Restore previous limits and view angle
    if current_xlim is not None:
         ax.set_xlim3d(current_xlim)
         ax.set_ylim3d(current_ylim)
         ax.set_zlim3d(current_zlim)
         ax.view_init(elev=current_elev, azim=current_azim)
    elif initial_xlim is not None: # If no previous limits, restore initial (shouldn't happen after first draw)
         ax.set_xlim3d(initial_xlim)
         ax.set_ylim3d(initial_ylim)
         ax.set_zlim3d(initial_zlim)
         ax.view_init(elev=initial_elev, azim=initial_azim)
    # else: let matplotlib autoscale for the very first draw if set_initial_limits failed


    fig.canvas.draw_idle()


def main():
    global node_positions, initial_xlim, initial_ylim, initial_zlim, keep_running

    console.rule("[bold blue]Starting Spatial Graph Visualization Sidecar[/]")

    # --- Initialization ---
    node_positions = load_cube_positions(POSITION_FILE)
    if node_positions is None:
         console.print("[red]Failed to initialize node positions. Exiting.[/red]")
         return

    last_pos = 0
    transition_counts = defaultdict(int)
    exit_counts_from_cube = defaultdict(int)
    last_path_transitions = set()

    plt.ion()
    fig = plt.figure(figsize=(12, 10))
    # Removed facecolor settings - use matplotlib defaults
    # fig.set_facecolor('darkgrey')
    ax = fig.add_subplot(111, projection='3d')
    # Removed facecolor setting - use matplotlib defaults (white grid usually)
    # ax.set_facecolor('black')


    fig.canvas.mpl_connect('close_event', on_close)

    # Colorbar setup (will be updated)
    cmap = plt.get_cmap('hot')
    norm = mcolors.Normalize(vmin=0, vmax=1) # Initial norm
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('log(1 + Transition Frequency)')

    # Reset View Button
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset View')

    def reset_view(event):
        console.log("Resetting view to initial state.")
        if initial_xlim is not None:
             ax.set_xlim3d(initial_xlim)
             ax.set_ylim3d(initial_ylim)
             ax.set_zlim3d(initial_zlim)
        ax.view_init(elev=initial_elev, azim=initial_azim)
        fig.canvas.draw_idle()

    button.on_clicked(reset_view)

    # --- Initial Setup ---
    # Set and store initial limits and view based on the loaded/random positions
    set_initial_limits_and_view(ax, node_positions)

    # Perform an initial draw with nodes and labels but no edges yet
    # Call the update function which handles cla() and drawing
    update_plot_with_collection(fig, ax, node_positions, transition_counts, 0, last_path_transitions, sm, cbar)
    plt.pause(0.1) # Allow plot window to appear

    # --- Main Loop Variables ---
    batches_read_since_layout = 0
    batches_read_since_plot_update = 0
    last_read_time = time.time()
    total_batches_read = 0


    console.log(f"Visualizing {NUM_CUBES} cubes. Check file every: {UPDATE_INTERVAL_SEC}s.")
    console.log(f"Recompute layout every: {LAYOUT_UPDATE_INTERVAL_BATCHES} batches.")
    console.log(f"Redraw plot every: {PLOT_UPDATE_INTERVAL_BATCHES} batches.")
    console.log(f"Waiting for paths file: {PATHS_FILE}")

    while keep_running:
        current_time = time.time()
        data_read_this_cycle = False # Flag to track if new data was processed

        # --- Check for New Data ---
        if current_time - last_read_time >= UPDATE_INTERVAL_SEC or batches_read_since_layout >= LAYOUT_UPDATE_INTERVAL_BATCHES or batches_read_since_plot_update >= PLOT_UPDATE_INTERVAL_BATCHES:
             # Ensure we check if any interval is met, not just read interval
            new_data_entries, new_last_pos, transitions_from_last_batch = read_path_data_incremental(PATHS_FILE, last_pos)
            last_read_time = current_time

            if new_last_pos != last_pos:
                last_pos = new_last_pos
                if new_data_entries:
                    data_read_this_cycle = True
                    batch_count_this_read = len(new_data_entries)
                    total_batches_read += batch_count_this_read
                    batches_read_since_layout += batch_count_this_read
                    batches_read_since_plot_update += batch_count_this_read

                    new_transitions_found = False
                    for entry in new_data_entries:
                        transitions_list = entry.get("transitions", [])
                        for t in transitions_list:
                            if isinstance(t, list) and len(t) == 2:
                                from_idx, to_idx = tuple(t)
                                if 0 <= from_idx < NUM_CUBES and 0 <= to_idx < NUM_CUBES :
                                     transition_counts[(from_idx, to_idx)] += 1
                                     new_transitions_found = True
                                elif from_idx != EXIT_TOKEN_INDEX and to_idx == EXIT_TOKEN_INDEX and 0 <= from_idx < NUM_CUBES:
                                     exit_counts_from_cube[from_idx] += 1

                    last_path_transitions = set(transitions_from_last_batch)

                    if new_transitions_found:
                        console.log(f"Read {batch_count_this_read} new batches (Total: {total_batches_read}). Unique edges: {len(transition_counts)}")
                        if exit_counts_from_cube:
                            exit_log_str = ", ".join([f"{c}->EXIT:{count}" for c, count in sorted(exit_counts_from_cube.items()) if count > 0])
                            if exit_log_str: console.log(f"Exits: {exit_log_str}")
                # If file existed but was empty or only malformed lines, new_data_entries will be empty

            # Check if file exists if it didn't before (only print if we tried to read)
            elif not os.path.exists(PATHS_FILE):
                 console.print(f"[yellow]Waiting for paths file: {PATHS_FILE}...[/yellow]", end="\r")


        # --- Determine if Layout or Plot Update is Needed ---
        needs_layout_compute = batches_read_since_layout >= LAYOUT_UPDATE_INTERVAL_BATCHES and len(transition_counts) > 0
        needs_redraw = batches_read_since_plot_update >= PLOT_UPDATE_INTERVAL_BATCHES or needs_layout_compute or (total_batches_read == 0 and os.path.exists(PATHS_FILE) and not data_read_this_cycle) # Also redraw initially once file is found/checked


        # --- Compute Layout (if needed) ---
        if needs_layout_compute:
            console.log("Triggering layout recomputation...")
            start_time = time.time()
            new_layout_positions = compute_graph_layout(transition_counts, NUM_CUBES, existing_pos=node_positions)
            end_time = time.time()
            console.log(f"Layout recomputation took {end_time - start_time:.2f} seconds.")

            layout_changed = False
            if new_layout_positions is not None:
                 # Check if layout actually changed significantly
                 if not np.allclose(node_positions, new_layout_positions, atol=1e-6): # Lower tolerance for check
                     node_positions = new_layout_positions
                     layout_changed = True
                     console.log("Node positions updated by layout.")
                 else:
                     console.log("Layout computation resulted in negligible changes.")

            batches_read_since_layout = 0 # Reset counter

            # If layout changed OR this was the first layout compute (initial_xlim is None)
            # Re-set limits and view to ensure the plot fits the new layout initially
            # The update_plot function will then restore based on the current view if possible
            if layout_changed or initial_xlim is None:
                 console.log("Layout changed or first layout, re-setting initial view limits.")
                 set_initial_limits_and_view(ax, node_positions)


        # --- Redraw Plot (if needed) ---
        # Redraw if needed based on plot interval, or if layout was just computed/changed
        if needs_redraw and node_positions is not None:
             console.log("Updating plot...")
             start_time = time.time()
             max_count = max(transition_counts.values()) if transition_counts else 0

             # Call the consolidated update function
             update_plot_with_collection(fig, ax, node_positions, transition_counts, max_count, last_path_transitions, sm, cbar)

             end_time = time.time()
             console.log(f"Plot redraw took {end_time - start_time:.3f} seconds.")
             batches_read_since_plot_update = 0 # Reset counter


        # --- Pause and Event Handling ---
        try:
            # Pause duration can be shorter if updates are frequent, longer if waiting
            # Use a small fixed pause to keep responsive and handle events
            plt.pause(0.05) # Shorter pause
        except Exception as e:
            if keep_running:
                console.print(f"[yellow]Error during plt.pause(): {e}. Might be closing.[/yellow]")
            keep_running = False


    # --- Cleanup ---
    console.print("[bold blue]Visualization loop finished. Closing plot window.[/]")
    plt.close('all')


if __name__ == "__main__":
    # Wait for position file before starting main logic
    if not os.path.exists(POSITION_FILE):
         console.print(f"[yellow]Visualization sidecar waiting for position file: {POSITION_FILE}...[/yellow]")
         while not os.path.exists(POSITION_FILE) and keep_running:
             try:
                 plt.pause(UPDATE_INTERVAL_SEC)
             except KeyboardInterrupt:
                 handle_sigint(None, None)
             except Exception:
                  time.sleep(UPDATE_INTERVAL_SEC)

         if not keep_running:
             console.print("[yellow]Exiting before position file appeared due to user interrupt.[/yellow]")
             sys.exit(0)
         if not os.path.exists(POSITION_FILE):
              console.print(f"[red]Position file {POSITION_FILE} did not appear. Exiting.[/red]")
              sys.exit(1)
         else:
              console.print(f"[green]Position file {POSITION_FILE} found. Starting visualization.[/green]")


    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Visualization script finalizing shutdown due to user interrupt.[/]")
        plt.close('all')
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred in visualization script: {e}[/]")
        console.print(traceback.format_exc())
        plt.close('all')
        sys.exit(1)