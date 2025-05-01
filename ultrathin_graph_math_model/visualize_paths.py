import redis
import json
import time
from collections import defaultdict, Counter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import math
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Dict, Tuple, List, Any
from rich.console import Console
from rich.progress import SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich_progress_console import console, make_progress

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PATH_LIST_KEY = 'spatial_graph:training_paths'
UPDATE_EVERY_N_BATCHES = 5
FIG_SIZE = (18, 10)
LAYOUT_VERTICAL_SPACING = 5
LAYOUT_HORIZONTAL_SPACING = 5
FIXED_NODE_SIZE = 250
EDGE_BASE_WIDTH = 0.3
EDGE_WIDTH_MULTIPLIER = 0.3
STATIC_NODE_COLOR = '#fff9c4'
END_NODE_EDGE_COLOR = 'red'
END_NODE_LINEWIDTH = 1.5

class PathVisualizer:
    def __init__(self):
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.text_handle: Optional[plt.Text] = None
        self.leaderboard_table_handle: Optional[plt.Table] = None

        self.G_viz: nx.DiGraph = nx.DiGraph()
        self.pos: Dict[Any, Tuple[float, float]] = {}
        self.point_coordinates: Dict[Any, Tuple[float, float]] = {}
        self.num_cubes: int = 0
        self.exit_token_index: Any = -1
        self.max_path_vis_depth: int = 30

        self.current_epoch: int = -1
        self.transition_counts: Dict[Tuple[Any, Any], int] = defaultdict(int)
        self.visit_counts: Dict[Any, int] = defaultdict(int)
        self.end_nodes_epoch: set = set()
        self.total_paths_processed_epoch: int = 0
        self.batches_since_last_update: int = 0
        self.last_batch_idx_processed: int = -1

        self.redis_client: Optional[redis.Redis] = None

        self.last_batch_time = time.time()
        self.paths_per_batch = 16
        self.batch_interval = 10
        self.display_time_per_path = self.batch_interval / self.paths_per_batch if self.paths_per_batch else 0.5
        self.current_batch_paths = []
        self.highlight_index = 0
        self.current_highlighted_path = None
        self.last_highlight_time = time.time()

        self.progress = make_progress()
        self.highlight_task_id: Optional[int] = None
        self._highlighting: bool = False

        self._load_config_and_coordinates()
        self._initialize_edge_colormap()

    def safe_log(self, *args, **kwargs):
        # Если прогрессбар активен, логируем через его консоль (stderr), иначе — в stdout
        if hasattr(self.progress, 'live') and self.progress.live and self.progress.live.is_started:
            self.progress.console.log(*args, **kwargs)
        else:
            console.log(*args, **kwargs)

    def safe_log_exception(self, show_locals=False):
        # Аналог safe_log для исключений с трассировкой
        if hasattr(self.progress, 'live') and self.progress.live and self.progress.live.is_started:
            self.progress.console.print_exception(show_locals=show_locals)
        else:
            console.print_exception(show_locals=show_locals)

    def _load_config_and_coordinates(self):
        json_path = 'points.json'
        try:
            import config
            self.num_cubes = getattr(config, 'NUM_CUBES', 0)
            self.exit_token_index = getattr(config, 'EXIT_TOKEN_INDEX', self.exit_token_index)
            self.max_path_vis_depth = getattr(config, 'ROUTING_SAFETY_LIMIT', self.max_path_vis_depth)
            json_path = getattr(config, 'POINTS_JSON_PATH', json_path)
            self.safe_log(f"Loaded constants from config.py: NUM_CUBES={self.num_cubes}, EXIT_TOKEN_INDEX={self.exit_token_index}, MAX_PATH_VIS_DEPTH={self.max_path_vis_depth}")
        except (ImportError, AttributeError, NameError) as e:
            self.safe_log(f"[yellow]Warning: Could not load config.py or specific constants ({e}). Using default values.[/yellow]")

        try:
            self.safe_log(f"Loading point coordinates from: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, dict) and 'points' in raw:
                points_data = raw['points']
            else:
                points_data = raw
            if not isinstance(points_data, list):
                raise ValueError("JSON data must be a list")

            loaded_coords = {}
            expected_keys = {'uid', 'x', 'y'}
            for point in points_data:
                if not expected_keys.issubset(point.keys()):
                    self.safe_log(f"[yellow]Warning: Point object missing 'uid', 'x', or 'y'. Skipping: {point}[/yellow]")
                    continue
                uid = point['uid']
                try:
                    x = float(point['x'])
                    y = float(point['y'])
                    loaded_coords[uid] = (x, y)
                except (ValueError, TypeError):
                    self.safe_log(f"[yellow]Warning: Invalid coordinates for uid {uid}. Skipping: {point}[/yellow]")

            self.point_coordinates = loaded_coords
            self.safe_log(f"Loaded coordinates for {len(self.point_coordinates)} points.")

            num_real_points = len([uid for uid in self.point_coordinates if uid != self.exit_token_index])
            if num_real_points > 0:
                self.num_cubes = num_real_points
                self.safe_log(f"Set NUM_CUBES based on JSON data (excluding EXIT_TOKEN_INDEX): {self.num_cubes}")
            elif self.num_cubes == 0:
                self.safe_log("[yellow]Warning: NUM_CUBES is 0. No points loaded.[/yellow]")

        except FileNotFoundError:
            self.safe_log(f"[bold red]Error: Coordinates file '{json_path}' not found. Visualization will be limited.[/bold red]")
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.safe_log(f"[bold red]Error parsing coordinates from '{json_path}': {e}. Visualization will be limited.[/bold red]")

    def _initialize_edge_colormap(self):
        self.edge_cmap = LinearSegmentedColormap.from_list(
            'custom_edge', ['#B0B0B0', '#E0D080', '#D06060', '#A03030', '#601010'], N=256
        )

    def _connect_redis(self) -> bool:
        if self.redis_client and self.redis_client.ping():
            return True
        try:
            self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
            self.redis_client.ping()
            self.safe_log(f"[Visualizer] Successfully connected to Redis {REDIS_HOST}:{REDIS_PORT}")
            return True
        except redis.exceptions.ConnectionError as e:
            self.safe_log(f"[bold red][Visualizer] Redis Connection Error: {e}. Retrying...[/bold red]")
            self.redis_client = None
            return False

    def _get_node_label(self, node_id: Any) -> str:
        return "EXIT" if node_id == self.exit_token_index else str(node_id)

    def _assign_fixed_layout(self):
        self.pos = {}
        if not self.point_coordinates:
            self.safe_log("[bold red]Error: No coordinates loaded. Cannot create fixed layout.[/bold red]")
            return

        max_x, min_x = -float('inf'), float('inf')
        max_y, min_y = -float('inf'), float('inf')
        has_points = False

        for uid, (x, y) in self.point_coordinates.items():
            self.pos[uid] = (x, y)
            if uid != self.exit_token_index:
                max_x, min_x = max(max_x, x), min(min_x, x)
                max_y, min_y = max(max_y, y), min(min_y, y)
                has_points = True

        if self.exit_token_index is not None:
            if self.exit_token_index not in self.pos:
                if has_points:
                    x_range = max_x - min_x if max_x > -float('inf') else 0
                    exit_x = max_x + max(LAYOUT_HORIZONTAL_SPACING, x_range * 0.15)
                    exit_y = (max_y + min_y) / 2 if max_y > -float('inf') else 0
                    self.pos[self.exit_token_index] = (exit_x, exit_y)
                    if not self._highlighting:
                        self.safe_log(f"Placed EXIT node ({self.exit_token_index}) at calculated position: ({exit_x:.2f}, {exit_y:.2f})")
                else:
                    self.pos[self.exit_token_index] = (0, 0)
                    self.safe_log(f"Placed EXIT node ({self.exit_token_index}) at fallback origin (0,0).")

        self.G_viz.add_nodes_from(self.pos.keys())
        if not self._highlighting:
            self.safe_log(f"Assigned fixed layout for {len(self.pos)} nodes based on coordinates.")

        if self.ax and has_points:
            x_range = max_x - min_x
            y_range = max_y - min_y
            padding_x = max(LAYOUT_HORIZONTAL_SPACING, x_range * 0.1)
            padding_y = max(LAYOUT_VERTICAL_SPACING, y_range * 0.1)
            final_max_x = self.pos.get(self.exit_token_index, (max_x, 0))[0] if self.exit_token_index is not None else max_x
            self.ax.set_xlim(min_x - padding_x, final_max_x + padding_x)
            self.ax.set_ylim(min_y - padding_y, max_y + padding_y)
            self.ax.figure.canvas.draw_idle()

    def _initialize_plot(self):
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            return
        self.safe_log("Initializing plot window...")
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=FIG_SIZE, dpi=100)
        self.ax.set_title("Waiting for epoch data...")
        self.ax.axis('off')
        self._assign_fixed_layout()
        self.text_handle = self.ax.text(0.01, 0.98, "Epoch: - | Batch: - | Paths: 0",
                                       transform=self.ax.transAxes, fontsize=10,
                                       verticalalignment='top')
        self.leaderboard_table_handle = None
        plt.show(block=False)
        try:
            self.fig.canvas.flush_events()
        except NotImplementedError:
            pass
        self.safe_log("Plot initialized and layout assigned.")

    def _prepare_node_styles(self) -> Tuple[list, list, list, list]:
        node_list = list(self.G_viz.nodes())
        node_colors = []
        node_edgecolors = []
        node_linewidths = []
        for node_id in node_list:
            node_colors.append(STATIC_NODE_COLOR)
            if node_id in self.end_nodes_epoch:
                node_edgecolors.append(END_NODE_EDGE_COLOR)
                node_linewidths.append(END_NODE_LINEWIDTH)
            else:
                node_edgecolors.append('black')
                node_linewidths.append(0.5)
        return node_list, node_colors, node_edgecolors, node_linewidths

    def _prepare_edge_styles(self) -> Tuple[list, list, list]:
        edge_list = list(self.G_viz.edges())
        edge_widths = []
        edge_colors = []
        max_transition_count = max(self.transition_counts.values()) if self.transition_counts else 1
        for u_id, v_id in edge_list:
            count = self.transition_counts.get((u_id, v_id), 0)
            if max_transition_count > 0:
                norm_count = math.log1p(count) / math.log1p(max_transition_count)
            else:
                norm_count = 0
            width = EDGE_BASE_WIDTH + EDGE_WIDTH_MULTIPLIER * norm_count * 3.0
            edge_widths.append(width)
            edge_colors.append(self.edge_cmap(norm_count))
        return edge_list, edge_widths, edge_colors

    def _draw_graph_elements(self, node_styles, edge_styles):
        node_list, node_colors, node_edgecolors, node_linewidths = node_styles
        edge_list, edge_widths, edge_colors = edge_styles
        if not self.pos or not self.G_viz:
            return
        drawable_nodes = [n for n in node_list if n in self.pos]
        pos_for_drawing = {n: self.pos[n] for n in drawable_nodes}
        if not pos_for_drawing:
            return
        nx.draw_networkx_edges(
            self.G_viz, self.pos,
            edgelist=edge_list,
            width=edge_widths,
            edge_color=edge_colors,
            arrows=True,
            arrowstyle='->',
            arrowsize=10,
            node_size=FIXED_NODE_SIZE,
            connectionstyle='arc3,rad=0.1',
            ax=self.ax
        )
        nx.draw_networkx_nodes(
            self.G_viz, pos_for_drawing,
            nodelist=drawable_nodes,
            node_size=FIXED_NODE_SIZE,
            node_color=[node_colors[i] for i, n in enumerate(node_list) if n in pos_for_drawing],
            edgecolors=[node_edgecolors[i] for i, n in enumerate(node_list) if n in pos_for_drawing],
            linewidths=[node_linewidths[i] for i, n in enumerate(node_list) if n in pos_for_drawing],
            ax=self.ax
        )
        labels = {node_id: self._get_node_label(node_id) for node_id in drawable_nodes}
        nx.draw_networkx_labels(self.G_viz, pos_for_drawing, labels=labels, font_size=8, ax=self.ax)

    def _safe_update_text(self, text: str):
        if not self.ax or not self.text_handle:
            return
        try:
            if self.text_handle.get_text() != text:
                self.text_handle.set_text(text)
        except Exception as e:
            self.safe_log(f"[yellow]Could not update text handle: {e}. Recreating.[/yellow]")
            try:
                self.text_handle.remove()
            except Exception:
                pass
            self.text_handle = self.ax.text(0.01, 0.98, text, transform=self.ax.transAxes, fontsize=10, verticalalignment='top')

    def _update_plot_info(self):
        title = f"Path Tree Visualization - Epoch {self.current_epoch} (Cumulative)"
        info_text = f"Epoch: {self.current_epoch} | Last Batch: {self.last_batch_idx_processed} | Paths: {self.total_paths_processed_epoch}"
        if self.ax:
            self.ax.set_title(title)
        self._safe_update_text(info_text)

    def reset_epoch_data(self, new_epoch: int):
        console.rule(f"Resetting counts for Epoch {new_epoch}")
        self.transition_counts = defaultdict(int)
        self.visit_counts = defaultdict(int)
        self.end_nodes_epoch = set()
        self.total_paths_processed_epoch = 0
        self.batches_since_last_update = 0
        self.current_epoch = new_epoch
        self.last_batch_idx_processed = -1
        self.G_viz.remove_edges_from(list(self.G_viz.edges()))
        if self.ax:
            self.update_plot(draw_table=False)

    def print_transition_leaderboard(self, top_n=10):
        leaderboard = self._get_transition_leaderboard(top_n)
        self.safe_log("\n[bold cyan]Top Transitions Leaderboard:[/bold cyan]")
        if not leaderboard or leaderboard == ["No transition data"] or leaderboard == ["No transitions counted"]:
            self.safe_log("  (No transition data available)")
        else:
            for i, row in enumerate(leaderboard):
                self.safe_log(f"{i+1}. {row}")
        self.safe_log("")

    def _get_transition_leaderboard(self, top_n=8) -> List[str]:
        if not self.transition_counts:
            return ["No transition data"]
        sorted_transitions = sorted(self.transition_counts.items(), key=lambda item: item[1], reverse=True)
        total_transitions = sum(self.transition_counts.values())
        if total_transitions == 0:
            return ["No transitions counted"]
        result = []
        for (u_id, v_id), count in sorted_transitions[:top_n]:
            percent = (count / total_transitions) * 100 if total_transitions else 0
            u_label = self._get_node_label(u_id)
            v_label = self._get_node_label(v_id)
            result.append(f"{u_label} → {v_label} : {percent:.1f}% ({count})")
        return result if result else ["No significant transitions"]

    def _draw_leaderboard_table(self):
        if not self.ax:
            return
        leaderboard = self._get_transition_leaderboard(top_n=8)
        table_data = [[row] for row in leaderboard]
        col_labels = ["Top Transitions"]
        table_bbox = [0.01, 0.75, 0.25, 0.22]
        try:
            if self.leaderboard_table_handle:
                self.leaderboard_table_handle.remove()
                self.leaderboard_table_handle = None
            if table_data:
                new_table = self.ax.table(
                    cellText=table_data, colLabels=col_labels, loc='upper left',
                    bbox=table_bbox, cellLoc='left', colLoc='center'
                )
                new_table.auto_set_font_size(False)
                new_table.set_fontsize(8)
                for key, cell in new_table.get_celld().items():
                    cell.set_linewidth(0.5)
                    if key[0] == 0:
                        cell.set_facecolor('#d1e5f0')
                        cell.set_fontsize(9)
                        cell.set_text_props(weight='bold')
                    else:
                        cell.set_facecolor('#f7fbff')
                self.leaderboard_table_handle = new_table
            else:
                self.leaderboard_table_handle = None
        except Exception as e:
            self.safe_log(f"[red]Error drawing leaderboard table: {e}[/red]")
            self.leaderboard_table_handle = None

    def update_plot(self, draw_table=False):
        if not self.fig or not self.ax or not plt.fignum_exists(self.fig.number):
            return
        self.ax.clear()
        self.ax.axis('off')
        self._assign_fixed_layout()
        self.leaderboard_table_handle = None
        if not self.G_viz or not self.pos:
            self.ax.set_title(f"Epoch {self.current_epoch} - Waiting for data or coordinates...")
            self._update_plot_info()
        else:
            node_styles = self._prepare_node_styles()
            edge_styles = self._prepare_edge_styles()
            self._draw_graph_elements(node_styles, edge_styles)
            self._update_plot_info()
            if draw_table:
                self._draw_leaderboard_table()
        self.fig.canvas.draw_idle()

    def _add_path_to_graph(self, path: list):
        if not path:
            return
        limited_path = path[:self.max_path_vis_depth]
        if not limited_path:
            return
        self.total_paths_processed_epoch += 1
        for node_id in limited_path:
            if node_id not in self.pos:
                self.safe_log(f"[yellow]Warning: Node {node_id} from path not in points.json. Assigning fallback position.[/yellow]")
                neighbors_pos = [self.pos[p] for p in limited_path if p in self.pos and p != node_id]
                if neighbors_pos:
                    avg_x = sum(p[0] for p in neighbors_pos) / len(neighbors_pos)
                    avg_y = sum(p[1] for p in neighbors_pos) / len(neighbors_pos)
                    self.pos[node_id] = (avg_x + np.random.randn()*LAYOUT_HORIZONTAL_SPACING*0.05,
                                        avg_y + np.random.randn()*LAYOUT_VERTICAL_SPACING*0.05)
                else:
                    self.pos[node_id] = (np.random.randn()*LAYOUT_HORIZONTAL_SPACING*0.1,
                                        np.random.randn()*LAYOUT_VERTICAL_SPACING*0.1)
                self.safe_log(f"Assigned fallback position {self.pos[node_id]} to node {node_id}")
            self.G_viz.add_node(node_id)
        if limited_path:
            self.visit_counts[limited_path[0]] += 1
        for i in range(len(limited_path) - 1):
            u_node_id = limited_path[i]
            v_node_id = limited_path[i+1]
            self.G_viz.add_edge(u_node_id, v_node_id)
            self.transition_counts[(u_node_id, v_node_id)] += 1
            self.visit_counts[v_node_id] += 1
        if limited_path:
            self.end_nodes_epoch.add(limited_path[-1])

    def _handle_epoch_logic(self, epoch: int, is_end_marker: bool, batch_idx: int) -> Tuple[bool, bool]:
        epoch_changed = False
        current_active_epoch = self.current_epoch != -1
        if epoch != -1 and epoch != self.current_epoch and not is_end_marker:
            if current_active_epoch and self.total_paths_processed_epoch > 0:
                self.safe_log(f"Final update for Epoch {self.current_epoch} before switching to {epoch}.")
                self.update_plot(draw_table=True)
                self.print_transition_leaderboard(top_n=10)
                if self.fig and plt.fignum_exists(self.fig.number):
                    try:
                        self.fig.canvas.flush_events()
                    except Exception:
                        pass
            self.reset_epoch_data(epoch)
            epoch_changed = True
            self.last_batch_idx_processed = batch_idx
            self.batches_since_last_update = 0
            return True, epoch_changed
        elif is_end_marker and epoch == self.current_epoch:
            self.safe_log(f"End of Epoch {self.current_epoch} marker received (Batch: {batch_idx}).")
            if self.total_paths_processed_epoch > 0:
                self.update_plot(draw_table=True)
                self.print_transition_leaderboard(top_n=10)
                if self.fig and plt.fignum_exists(self.fig.number):
                    try:
                        self.fig.canvas.flush_events()
                    except Exception:
                        pass
            else:
                self.safe_log("Epoch ended with no paths processed.")
            self.current_epoch = -1
            epoch_changed = True
            return False, epoch_changed
        elif self.current_epoch == -1 or epoch != self.current_epoch:
            return False, False
        if batch_idx > self.last_batch_idx_processed:
            self.last_batch_idx_processed = batch_idx
        self.batches_since_last_update += 1
        return True, False

    def _process_message_batch(self, messages_bytes: List[bytes]) -> bool:
        needs_update = False
        paths_added_in_batch = 0
        epoch_state_changed = False
        current_time = time.time()
        all_batch_paths: List[Tuple[Any, ...]] = []
        for message_bytes in messages_bytes:
            try:
                message_data = json.loads(message_bytes.decode('utf-8'))
                epoch = message_data.get("epoch", -1)
                is_end_marker = message_data.get("status") == "epoch_end"
                batch_idx = message_data.get("batch_index", -1)
                should_process_paths, epoch_changed = self._handle_epoch_logic(epoch, is_end_marker, batch_idx)
                if epoch_changed:
                    epoch_state_changed = True
                if not should_process_paths:
                    continue
                paths = message_data.get("paths", [])
                if isinstance(paths, list):
                    for path in paths:
                        if isinstance(path, list) and path:
                            self._add_path_to_graph(path)
                            paths_added_in_batch += 1
                            all_batch_paths.append(tuple(path))
                    if paths:
                        self.current_batch_paths = paths
                        self.highlight_index = len(self.current_batch_paths) - 1
                        self.batch_interval = current_time - self.last_batch_time
                        self.last_batch_time = current_time
                        self.paths_per_batch = len(paths)
                        self.display_time_per_path = self.batch_interval / self.paths_per_batch if self.paths_per_batch else 0.5
                        # Initialize or reset progress task for highlighting
                        if self.highlight_task_id is None:
                            self.highlight_task_id = self.progress.add_task("Highlighting paths", total=self.paths_per_batch)
                        else:
                            # Reset progress to new batch
                            self.progress.update(self.highlight_task_id, total=self.paths_per_batch, completed=0)
            except json.JSONDecodeError as e:
                self.safe_log(f"[red]Failed to decode JSON message: {e}[/red]")
            except Exception as e:
                self.safe_log(f"[red]Error processing message: {e}[/red]")
                self.safe_log_exception(show_locals=False)
        if paths_added_in_batch > 0:
            total_paths = len(all_batch_paths)
            if total_paths > 0:
                most_common_path, count = Counter(all_batch_paths).most_common(1)[0]
                if count > total_paths / 2:
                    self.safe_log(f"[yellow]Warning: More than 50% of paths in this batch are identical ({list(most_common_path)} occurred {count} out of {total_paths} times).[/yellow]")
            self.safe_log(f"[bold green]Added {paths_added_in_batch} paths for Epoch {self.current_epoch}.[/bold green]")
            needs_update = True
        if self.current_epoch != -1:
            if needs_update or (epoch_state_changed and paths_added_in_batch == 0) or self.batches_since_last_update >= UPDATE_EVERY_N_BATCHES:
                self.safe_log(f"Requesting plot update (Epoch: {self.current_epoch}, Batch: {self.last_batch_idx_processed}, Paths: {self.total_paths_processed_epoch}). Trigger: {'paths' if needs_update else ('epoch' if epoch_state_changed else 'batch count')}")
                self.update_plot(draw_table=False)
                self.batches_since_last_update = 0
                return True
        elif epoch_state_changed:
            return True
        return False

    def run(self):
        while not self._connect_redis():
            time.sleep(5)
        self._initialize_plot()
        self.progress.start()
        self.safe_log("Starting path processing loop...")
        try:
            while True:
                # Exit if plot closed
                if self.fig and not plt.fignum_exists(self.fig.number):
                    self.safe_log("Plot window closed by user. Exiting.")
                    break
                now = time.time()
                # Highlight cycle
                if now - self.last_highlight_time >= self.display_time_per_path:
                    self.last_highlight_time = now
                    # Fetch only latest message
                    try:
                        msg = self.redis_client.rpop(REDIS_PATH_LIST_KEY)
                        if msg:
                            data = json.loads(msg.decode('utf-8'))
                            paths = data.get('paths', []) or []
                            # Add to graph and set new batch
                            for path in paths:
                                if isinstance(path, list) and path:
                                    self._add_path_to_graph(path)
                            self.current_batch_paths = paths
                            self.highlight_index = len(self.current_batch_paths) - 1
                            # Adjust display speed based on arrival interval
                            fetch_interval = now - getattr(self, 'last_fetch_time', now)
                            self.last_fetch_time = now
                            if paths:
                                self.paths_per_batch = len(paths)
                                self.display_time_per_path = fetch_interval / len(paths)
                            # Reset or init progress bar
                            if self.highlight_task_id is None:
                                self.highlight_task_id = self.progress.add_task("Highlighting paths", total=self.paths_per_batch)
                            else:
                                # Reset progress to new batch
                                self.progress.update(self.highlight_task_id, total=self.paths_per_batch, completed=0)
                    except redis.exceptions.ConnectionError as e:
                        self.safe_log(f"[bold red]Redis error: {e}. Reconnecting...[/bold red]")
                        self.redis_client = None
                        time.sleep(5)
                        while not self._connect_redis(): time.sleep(5)
                    except Exception as e:
                        self.safe_log(f"[bold red]Fetch error: {e}[/bold red]")
                        self.safe_log_exception(show_locals=False)
                    # Show next path
                    if self.current_batch_paths and self.highlight_index >= 0:
                        self.current_highlighted_path = self.current_batch_paths[self.highlight_index]
                        self.highlight_index = (self.highlight_index - 1) % len(self.current_batch_paths)
                        self.update_plot(draw_table=False)
                        if self.highlight_task_id is not None:
                            self.progress.update(self.highlight_task_id, advance=1)
                # Refresh GUI
                if self.fig and plt.fignum_exists(self.fig.number):
                    try: self.fig.canvas.flush_events()
                    except Exception as e: self.safe_log(f"[yellow]GUI flush error: {e}[/yellow]")
                time.sleep(0.01)
        except KeyboardInterrupt:
            self.safe_log("\nProcessing stopped by user (Ctrl+C).")

if __name__ == "__main__":
    visualizer = PathVisualizer()
    visualizer.run()