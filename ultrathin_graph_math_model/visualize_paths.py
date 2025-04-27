# --- START OF FILE visualize_paths.py ---
import redis
import json
import time
from collections import defaultdict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from rich.console import Console
import math
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Dict, Tuple, List, Any

console = Console()

# --- Конфигурация ---
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PATH_LIST_KEY = 'spatial_graph:training_paths'
UPDATE_EVERY_N_BATCHES = 5
FIG_SIZE = (18, 10) # Увеличим высоту для лучшего обзора
LAYOUT_VERTICAL_SPACING = 5 # Увеличим для лучшего разделения
LAYOUT_HORIZONTAL_SPACING = 5 # Увеличим для лучшего разделения
FIXED_NODE_SIZE = 250 # Немного увеличим размер узла
EDGE_BASE_WIDTH = 0.3
EDGE_WIDTH_MULTIPLIER = 0.3 # Увеличим множитель для большей заметности
STATIC_NODE_COLOR = '#fff9c4' # светло-жёлтый
END_NODE_EDGE_COLOR = 'red'
END_NODE_LINEWIDTH = 1.5

class PathVisualizer:
    def __init__(self):
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.text_handle: Optional[plt.Text] = None
        self.leaderboard_table_handle: Optional[plt.Table] = None

        self.G_viz: nx.DiGraph = nx.DiGraph() # Используем простой ориентированный граф
        self.pos: Dict[Any, Tuple[float, float]] = {} # Карта: uid -> (x, y)
        self.point_coordinates: Dict[Any, Tuple[float, float]] = {}
        self.num_cubes: int = 0 # Будет установлено из JSON или config
        self.exit_token_index: Any = -1
        self.max_path_vis_depth: int = 30

        self.current_epoch: int = -1
        self.transition_counts: Dict[Tuple[Any, Any], int] = defaultdict(int)
        self.visit_counts: Dict[Any, int] = defaultdict(int)
        self.end_nodes_epoch: set = set() # Храним uid конечных узлов
        self.total_paths_processed_epoch: int = 0
        self.batches_since_last_update: int = 0
        self.last_batch_idx_processed: int = -1

        self.redis_client: Optional[redis.Redis] = None

        self._load_config_and_coordinates()
        self._initialize_edge_colormap()
        # Расположение узлов выполняется при инициализации плота

    def _load_config_and_coordinates(self):
        """Загружает конфигурацию и координаты точек."""
        json_path = 'points.json' # Default
        try:
            import config
            self.num_cubes = getattr(config, 'NUM_CUBES', 0) # Default 0, set later
            self.exit_token_index = getattr(config, 'EXIT_TOKEN_INDEX', self.exit_token_index)
            self.max_path_vis_depth = getattr(config, 'ROUTING_SAFETY_LIMIT', self.max_path_vis_depth)
            json_path = getattr(config, 'POINTS_JSON_PATH', json_path)
            console.log(f"Loaded constants from config.py: NUM_CUBES={self.num_cubes}, EXIT_TOKEN_INDEX={self.exit_token_index}, MAX_PATH_VIS_DEPTH={self.max_path_vis_depth}")
        except (ImportError, AttributeError, NameError) as e:
            console.print(f"[yellow]Warning: Could not load config.py or specific constants ({e}). Using default values.[/yellow]")

        try:
            console.log(f"Loading point coordinates from: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                points_data = json.load(f)
            if not isinstance(points_data, list):
                raise ValueError("JSON data must be a list")

            loaded_coords = {}
            expected_keys = {'uid', 'x', 'y'}
            for point in points_data:
                if not expected_keys.issubset(point.keys()):
                    console.print(f"[yellow]Warning: Point object missing 'uid', 'x', or 'y'. Skipping: {point}[/yellow]")
                    continue
                uid = point['uid']
                try:
                    # NetworkX использует (x, y), matplotlib тоже.
                    x = float(point['x'])
                    y = float(point['y'])
                    loaded_coords[uid] = (x, y)
                except (ValueError, TypeError):
                     console.print(f"[yellow]Warning: Invalid coordinates for uid {uid}. Skipping: {point}[/yellow]")

            self.point_coordinates = loaded_coords
            console.log(f"Loaded coordinates for {len(self.point_coordinates)} points.")

            # Обновляем num_cubes на основе реальных точек (исключая EXIT)
            num_real_points = len([uid for uid in self.point_coordinates if uid != self.exit_token_index])
            if num_real_points > 0:
                 self.num_cubes = num_real_points
                 console.log(f"Set NUM_CUBES based on JSON data (excluding EXIT_TOKEN_INDEX): {self.num_cubes}")
            elif self.num_cubes == 0: # Если config не загрузился и JSON пуст
                 console.print("[yellow]Warning: NUM_CUBES is 0. No points loaded.[/yellow]")


        except FileNotFoundError:
            console.print(f"[bold red]Error: Coordinates file '{json_path}' not found. Visualization will be limited.[/bold red]")
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            console.print(f"[bold red]Error parsing coordinates from '{json_path}': {e}. Visualization will be limited.[/bold red]")

    def _initialize_edge_colormap(self):
        """Инициализирует цветовую карту для ребер."""
        # От серого к красно-коричневому
        self.edge_cmap = LinearSegmentedColormap.from_list(
            'custom_edge', ['#B0B0B0', '#E0D080', '#D06060', '#A03030', '#601010'], N=256
        )

    def _connect_redis(self) -> bool:
        """Устанавливает соединение с Redis."""
        if self.redis_client and self.redis_client.ping():
            return True
        try:
            self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
            self.redis_client.ping()
            console.log(f"[Visualizer] Successfully connected to Redis {REDIS_HOST}:{REDIS_PORT}")
            return True
        except redis.exceptions.ConnectionError as e:
            console.print(f"[bold red][Visualizer] Redis Connection Error: {e}. Retrying...[/bold red]")
            self.redis_client = None
            return False

    def _get_node_label(self, node_id: Any) -> str:
        """Возвращает метку для узла."""
        return "EXIT" if node_id == self.exit_token_index else str(node_id)

    def _assign_fixed_layout(self):
        """
        Назначает фиксированные позиции узлам на основе self.point_coordinates.
        Вызывается один раз при инициализации графика.
        Добавляет все узлы из координат в self.G_viz.
        """
        self.pos = {}
        if not self.point_coordinates:
            console.print("[bold red]Error: No coordinates loaded. Cannot create fixed layout.[/bold red]")
            return

        max_x, min_x = -float('inf'), float('inf')
        max_y, min_y = -float('inf'), float('inf')
        has_points = False

        # 1. Установить позиции из JSON
        for uid, (x, y) in self.point_coordinates.items():
            self.pos[uid] = (x, y)
            if uid != self.exit_token_index:
                max_x, min_x = max(max_x, x), min(min_x, x)
                max_y, min_y = max(max_y, y), min(min_y, y)
                has_points = True

        # 2. Разместить EXIT узел (если он есть и не был в JSON)
        if self.exit_token_index is not None:
            if self.exit_token_index not in self.pos:
                if has_points:
                    # Размещаем справа, на среднем Y, с отступом
                    x_range = max_x - min_x if max_x > -float('inf') else 0
                    exit_x = max_x + max(LAYOUT_HORIZONTAL_SPACING, x_range * 0.15)
                    exit_y = (max_y + min_y) / 2 if max_y > -float('inf') else 0
                    self.pos[self.exit_token_index] = (exit_x, exit_y)
                    console.log(f"Placed EXIT node ({self.exit_token_index}) at calculated position: ({exit_x:.2f}, {exit_y:.2f})")
                else:
                    self.pos[self.exit_token_index] = (0, 0) # Fallback
                    console.log(f"Placed EXIT node ({self.exit_token_index}) at fallback origin (0,0).")

        # 3. Добавить все узлы с позициями в граф G_viz, чтобы они были видны
        self.G_viz.add_nodes_from(self.pos.keys())

        console.log(f"Assigned fixed layout for {len(self.pos)} nodes based on coordinates.")

        # 4. Установить границы осей для стабильности
        if self.ax and has_points:
            x_range = max_x - min_x
            y_range = max_y - min_y
            padding_x = max(LAYOUT_HORIZONTAL_SPACING, x_range * 0.1)
            padding_y = max(LAYOUT_VERTICAL_SPACING, y_range * 0.1)

            # Учитываем позицию EXIT при расчете правой границы
            final_max_x = self.pos.get(self.exit_token_index, (max_x, 0))[0] if self.exit_token_index is not None else max_x

            self.ax.set_xlim(min_x - padding_x, final_max_x + padding_x)
            self.ax.set_ylim(min_y - padding_y, max_y + padding_y)
            self.ax.figure.canvas.draw_idle() # Обновить оси


    def _initialize_plot(self):
        """Инициализирует окно Matplotlib и назначает расположение."""
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            return # Уже инициализировано

        console.log("Initializing plot window...")
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=FIG_SIZE, dpi=100)
        self.ax.set_title("Waiting for epoch data...")
        self.ax.axis('off') # Отключим стандартные оси
        # Важно: Сначала создаем оси, потом назначаем layout, который может установить xlim/ylim
        self._assign_fixed_layout()

        self.text_handle = self.ax.text(0.01, 0.98, "Epoch: - | Batch: - | Paths: 0",
                                        transform=self.ax.transAxes, fontsize=10,
                                        verticalalignment='top')
        self.leaderboard_table_handle = None # Инициализируем как None

        plt.show(block=False)
        try:
            self.fig.canvas.flush_events()
        except NotImplementedError:
            pass
        console.log("Plot initialized and layout assigned.")


    def _prepare_node_styles(self) -> Tuple[list, list, list, list]:
        """Готовит списки стилей для узлов (UIDs)."""
        # Рисуем только те узлы, которые есть в графе G_viz
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
        """Готовит списки стилей для ребер (u_id, v_id)."""
        # Берем ребра из графа G_viz
        edge_list = list(self.G_viz.edges())
        edge_widths = []
        edge_colors = []
        max_transition_count = max(self.transition_counts.values()) if self.transition_counts else 1

        for u_id, v_id in edge_list:
            count = self.transition_counts.get((u_id, v_id), 0)
            if max_transition_count > 0:
                # Логарифмическое масштабирование для лучшего восприятия разницы
                norm_count = math.log1p(count) / math.log1p(max_transition_count)
            else:
                norm_count = 0
            # Увеличим базовую ширину и множитель для лучшей видимости
            width = EDGE_BASE_WIDTH + EDGE_WIDTH_MULTIPLIER * norm_count * 3.0
            edge_widths.append(width)
            edge_colors.append(self.edge_cmap(norm_count))

        return edge_list, edge_widths, edge_colors

    def _draw_graph_elements(self, node_styles, edge_styles):
        """Отрисовывает узлы, ребра и метки на основе статического self.pos."""
        node_list, node_colors, node_edgecolors, node_linewidths = node_styles
        edge_list, edge_widths, edge_colors = edge_styles

        if not self.pos or not self.G_viz: return # Нечего рисовать

        # Убедимся, что все узлы в node_list имеют позицию
        drawable_nodes = [n for n in node_list if n in self.pos]
        pos_for_drawing = {n: self.pos[n] for n in drawable_nodes}

        if not pos_for_drawing: return # Нет узлов для отрисовки

        # Рисуем ребра СНАЧАЛА
        nx.draw_networkx_edges(
            self.G_viz, self.pos, # Используем полное pos для ребер
            edgelist=edge_list, # Ребра из G_viz
            width=edge_widths,
            edge_color=edge_colors,
            arrows=True,
            arrowstyle='->',
            arrowsize=10, # Немного больше стрелки
            node_size=FIXED_NODE_SIZE,
            connectionstyle='arc3,rad=0.1', # Слегка изогнутые ребра
            ax=self.ax
        )

        # Затем рисуем узлы из списка drawable_nodes
        nx.draw_networkx_nodes(
            self.G_viz, pos_for_drawing, # Используем отфильтрованные позиции
            nodelist=drawable_nodes, # Рисуем только существующие узлы
            node_size=FIXED_NODE_SIZE,
            node_color=[node_colors[i] for i, n in enumerate(node_list) if n in pos_for_drawing],
            edgecolors=[node_edgecolors[i] for i, n in enumerate(node_list) if n in pos_for_drawing],
            linewidths=[node_linewidths[i] for i, n in enumerate(node_list) if n in pos_for_drawing],
            ax=self.ax
        )

        # Метки рисуем последними для drawable_nodes
        labels = {node_id: self._get_node_label(node_id) for node_id in drawable_nodes}
        nx.draw_networkx_labels(self.G_viz, pos_for_drawing, labels=labels, font_size=8, ax=self.ax)


    def _safe_update_text(self, text: str):
        """Безопасно обновляет текстовое поле на графике."""
        if not self.ax or not self.text_handle: return
        try:
            if self.text_handle.get_text() != text:
                self.text_handle.set_text(text)
        except Exception as e:
            console.print(f"[yellow]Could not update text handle: {e}. Recreating.[/yellow]")
            try: self.text_handle.remove()
            except Exception: pass
            self.text_handle = self.ax.text(0.01, 0.98, text, transform=self.ax.transAxes, fontsize=10, verticalalignment='top')

    def _update_plot_info(self):
        """Обновляет заголовок и информационный текст."""
        title = f"Path Tree Visualization - Epoch {self.current_epoch} (Cumulative)"
        info_text = f"Epoch: {self.current_epoch} | Last Batch: {self.last_batch_idx_processed} | Paths: {self.total_paths_processed_epoch}"
        if self.ax: self.ax.set_title(title)
        self._safe_update_text(info_text)

    def _get_transition_leaderboard(self, top_n=8) -> List[str]:
        """Формирует список строк для таблицы лидеров переходов."""
        if not self.transition_counts: return ["No transition data"]
        # Сортируем по убыванию количества
        sorted_transitions = sorted(self.transition_counts.items(), key=lambda item: item[1], reverse=True)

        # Считаем общий вес (сумма логарифмов) для процентов, если хотим % от "важности"
        # total_log_weight = sum(math.log1p(c) for c in self.transition_counts.values())
        # Или просто по количеству
        total_transitions = sum(self.transition_counts.values())
        if total_transitions == 0: return ["No transitions counted"]

        result = []
        for (u_id, v_id), count in sorted_transitions[:top_n]:
            percent = (count / total_transitions) * 100 if total_transitions else 0
            u_label = self._get_node_label(u_id)
            v_label = self._get_node_label(v_id)
            result.append(f"{u_label} → {v_label} : {percent:.1f}% ({count})")
        return result if result else ["No significant transitions"]

    def _draw_leaderboard_table(self):
         """Отрисовывает таблицу лидеров на графике."""
         if not self.ax: return
         leaderboard = self._get_transition_leaderboard(top_n=8)
         table_data = [[row] for row in leaderboard]
         col_labels = ["Top Transitions"]
          # Разместим таблицу немного ниже, чтобы не перекрывать инфо-текст
         table_bbox = [0.01, 0.75, 0.25, 0.22] # x, y, width, height

         try:
             # Удаляем старую таблицу, если она есть
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
                     if key[0] == 0: # Header
                         cell.set_facecolor('#d1e5f0'); cell.set_fontsize(9); cell.set_text_props(weight='bold')
                     else: # Data
                         cell.set_facecolor('#f7fbff')
                 self.leaderboard_table_handle = new_table # Сохраняем ссылку
             else:
                 self.leaderboard_table_handle = None

         except Exception as e:
             console.print(f"[red]Error drawing leaderboard table: {e}[/red]")
             self.leaderboard_table_handle = None # Сброс при ошибке


    def update_plot(self, draw_table=False):
        """Обновляет график, используя статическое расположение."""
        if not self.fig or not self.ax or not plt.fignum_exists(self.fig.number):
            return

        start_time = time.time()

        self.ax.clear() # Очищаем оси перед перерисовкой
        self.ax.axis('off') # Снова отключаем оси

        # Важно: Восстанавливаем границы осей после clear()
        self._assign_fixed_layout() # Переназначаем границы, т.к. clear() их сбросил

        # Сбросим ссылку на старую таблицу (она удалена вместе с clear())
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

        self.fig.canvas.draw_idle() # Запрос на перерисовку
        end_time = time.time()
        # console.log(f"Plot update requested. Took {end_time - start_time:.4f} seconds.")


    def reset_epoch_data(self, new_epoch: int):
        """Сбрасывает данные для новой эпохи, сохраняя узлы и расположение."""
        console.rule(f"Resetting counts for Epoch {new_epoch}")

        # Сбрасываем только изменяемые данные эпохи
        self.transition_counts = defaultdict(int)
        self.visit_counts = defaultdict(int)
        self.end_nodes_epoch = set()
        self.total_paths_processed_epoch = 0
        self.batches_since_last_update = 0
        self.max_depth_processed = 0 # Хотя глубина больше не используется для графа
        self.current_epoch = new_epoch
        self.last_batch_idx_processed = -1

        # Очищаем только рёбра графа, узлы остаются
        self.G_viz.remove_edges_from(list(self.G_viz.edges()))

        # Очищаем график (вызовет update_plot для перерисовки без ребер)
        if self.ax:
             self.update_plot(draw_table=False) # Перерисовать с пустыми ребрами
             # self.ax.set_title(f"Epoch {self.current_epoch} - Waiting for data...")
             # self._update_plot_info() # Обновит текст

        # Запрос перерисовки пустого графика (без ребер)
        # if self.fig and plt.fignum_exists(self.fig.number):
        #     self.fig.canvas.draw_idle()


    def print_transition_leaderboard(self, top_n=10):
        """Выводит ТОП переходов в консоль."""
        leaderboard = self._get_transition_leaderboard(top_n)
        console.print("\n[bold cyan]Top Transitions Leaderboard:[/bold cyan]")
        if not leaderboard or leaderboard == ["No transition data"] or leaderboard == ["No transitions counted"]:
             console.print("  (No transition data available)")
        else:
            for i, row in enumerate(leaderboard):
                console.print(f"{i+1}. {row}")
        console.print("")

    def _add_path_to_graph(self, path: list):
        """Добавляет один путь в граф (узлы=uid) и обновляет статистики."""
        if not path: return
        limited_path = path[:self.max_path_vis_depth]
        if not limited_path: return

        self.total_paths_processed_epoch += 1

        # Добавляем узлы и проверяем наличие координат
        for node_id in limited_path:
            if node_id not in self.pos:
                # Этот узел не был в points.json, нужно ему назначить позицию
                console.print(f"[yellow]Warning: Node {node_id} from path not in points.json. Assigning fallback position.[/yellow]")
                # Пытаемся разместить рядом с соседями по пути или случайно
                neighbors_pos = [self.pos[p] for p in limited_path if p in self.pos and p != node_id]
                if neighbors_pos:
                    avg_x = sum(p[0] for p in neighbors_pos) / len(neighbors_pos)
                    avg_y = sum(p[1] for p in neighbors_pos) / len(neighbors_pos)
                    # Добавляем небольшой случайный сдвиг
                    self.pos[node_id] = (avg_x + np.random.randn()*LAYOUT_HORIZONTAL_SPACING*0.05,
                                         avg_y + np.random.randn()*LAYOUT_VERTICAL_SPACING*0.05)
                else:
                    # Если нет соседей с позицией, ставим случайно около (0,0)
                    self.pos[node_id] = (np.random.randn()*LAYOUT_HORIZONTAL_SPACING*0.1,
                                         np.random.randn()*LAYOUT_VERTICAL_SPACING*0.1)
                console.log(f"Assigned fallback position {self.pos[node_id]} to node {node_id}")

            # Добавляем узел в граф, если его еще нет (NetworkX игнорирует, если есть)
            self.G_viz.add_node(node_id)

        # Обновляем счетчики визитов и переходов
        if limited_path:
            self.visit_counts[limited_path[0]] += 1 # Первый узел

        for i in range(len(limited_path) - 1):
            u_node_id = limited_path[i]
            v_node_id = limited_path[i+1]

            # Добавляем ребро (NetworkX DiGraph не создает дубликаты)
            self.G_viz.add_edge(u_node_id, v_node_id)

            # Обновляем счетчики
            self.transition_counts[(u_node_id, v_node_id)] += 1
            self.visit_counts[v_node_id] += 1 # Последующие узлы

        # Отмечаем конечный узел пути
        if limited_path:
            self.end_nodes_epoch.add(limited_path[-1])


    def _handle_epoch_logic(self, epoch: int, is_end_marker: bool, batch_idx: int) -> Tuple[bool, bool]:
        """
        Обрабатывает логику смены и завершения эпохи.
        Возвращает: (продолжить_обработку_сообщения, эпоха_изменилась)
        """
        epoch_changed = False
        current_active_epoch = self.current_epoch != -1

        if epoch != -1 and epoch != self.current_epoch and not is_end_marker:
            # Начало новой эпохи
            if current_active_epoch and self.total_paths_processed_epoch > 0:
                console.log(f"Final update for Epoch {self.current_epoch} before switching to {epoch}.")
                self.update_plot(draw_table=True)
                self.print_transition_leaderboard(top_n=10)
                if self.fig and plt.fignum_exists(self.fig.number):
                    try: self.fig.canvas.flush_events()
                    except Exception: pass
                    # time.sleep(0.5) # Дадим время посмотреть перед сбросом

            self.reset_epoch_data(epoch)
            epoch_changed = True
            # Важно: сообщение может содержать пути УЖЕ для новой эпохи
            self.last_batch_idx_processed = batch_idx # Обновляем батч для новой эпохи
            self.batches_since_last_update = 0 # Сбрасываем счетчик батчей для обновления
            return True, epoch_changed

        elif is_end_marker and epoch == self.current_epoch:
            # Конец текущей активной эпохи
            console.log(f"End of Epoch {self.current_epoch} marker received (Batch: {batch_idx}).")
            if self.total_paths_processed_epoch > 0:
                self.update_plot(draw_table=True)
                self.print_transition_leaderboard(top_n=10)
                if self.fig and plt.fignum_exists(self.fig.number):
                    try: self.fig.canvas.flush_events()
                    except Exception: pass
                    # time.sleep(0.5) # Дадим время посмотреть
            else:
                console.log("Epoch ended with no paths processed.")
            # Сбрасываем в состояние ожидания (-1), но не вызываем reset_epoch_data здесь,
            # чтобы не потерять финальный вид эпохи до прихода новой.
            # Сброс произойдет при получении первого сообщения новой эпохи.
            self.current_epoch = -1 # Ставим статус "ожидание"
            epoch_changed = True # Сигнализируем об изменении состояния
            # Это сообщение-маркер не содержит путей, не обрабатываем его дальше
            return False, epoch_changed

        elif self.current_epoch == -1 or epoch != self.current_epoch:
            # Либо мы в ожидании (-1), либо пришло сообщение для старой/будущей эпохи
            # Игнорируем это сообщение
            return False, False

        # Сообщение относится к текущей активной эпохе
        if batch_idx > self.last_batch_idx_processed:
             self.last_batch_idx_processed = batch_idx
        self.batches_since_last_update += 1
        return True, False # Обрабатываем сообщение, эпоха не менялась


    def _process_message_batch(self, messages_bytes: List[bytes]) -> bool:
        """Обрабатывает пачку сообщений из Redis. Возвращает True, если график нуждается в обновлении."""
        needs_update = False
        paths_added_in_batch = 0
        epoch_state_changed = False # Флаг, что эпоха сменилась или завершилась в этой пачке

        for message_bytes in messages_bytes:
            try:
                message_data = json.loads(message_bytes.decode('utf-8'))
                epoch = message_data.get("epoch", -1)
                is_end_marker = message_data.get("status") == "epoch_end"
                batch_idx = message_data.get("batch_index", -1)

                should_process_paths, epoch_changed = self._handle_epoch_logic(epoch, is_end_marker, batch_idx)

                if epoch_changed:
                    epoch_state_changed = True # Запоминаем, что было изменение

                if not should_process_paths:
                    continue # Пропускаем пути из этого сообщения

                # Обработка путей, если необходимо
                paths = message_data.get("paths", [])
                if isinstance(paths, list):
                    for path in paths:
                        if isinstance(path, list) and path:
                            self._add_path_to_graph(path)
                            paths_added_in_batch += 1

            except json.JSONDecodeError as e:
                console.print(f"[red]Failed to decode JSON message: {e}[/red]")
            except Exception as e:
                console.print(f"[red]Error processing message: {e}[/red]")
                console.print_exception(show_locals=False)

        if paths_added_in_batch > 0:
            console.log(f"[bold green]Added {paths_added_in_batch} paths for Epoch {self.current_epoch}.[/bold green]")
            needs_update = True # Нужно обновить из-за новых путей

        # Решение об обновлении графика
        if self.current_epoch != -1: # Обновляем только если есть активная эпоха
            if needs_update or (epoch_state_changed and paths_added_in_batch == 0) or self.batches_since_last_update >= UPDATE_EVERY_N_BATCHES:
                 # Обновляем, если: есть новые пути ИЛИ эпоха сменилась/завершилась (даже без путей) ИЛИ накоплено N батчей
                console.log(f"Requesting plot update (Epoch: {self.current_epoch}, Batch: {self.last_batch_idx_processed}, Paths: {self.total_paths_processed_epoch}). Trigger: {'paths' if needs_update else ('epoch' if epoch_state_changed else 'batch count')}")
                self.update_plot(draw_table=False) # Запрос на обновление без таблицы
                self.batches_since_last_update = 0
                return True # График был (или будет) обновлен
        elif epoch_state_changed: # Если эпоха только что завершилась (стали -1)
             # Мы уже обновили график в _handle_epoch_logic с финальным состоянием
             # Ничего больше делать не нужно
             return True # Обновление уже было сделано/запрошено

        return False # График не обновлялся


    def run(self):
        """Основной цикл обработки."""
        while not self._connect_redis():
            time.sleep(5)

        # Инициализируем плот здесь, чтобы self.ax был доступен для _assign_fixed_layout
        self._initialize_plot()
        console.log("Starting path processing loop...")

        try:
            while True:
                # 1. Проверка активности окна
                if self.fig and not plt.fignum_exists(self.fig.number):
                    console.log("Plot window closed by user. Exiting.")
                    break

                # 2. Чтение и обработка данных из Redis
                plot_updated_or_requested = False
                try:
                    # Используем BLPOP для ожидания с таймаутом, чтобы не грузить CPU
                    # Возвращает кортеж (key, value) или None по таймауту
                    # Мы читаем все сразу, так что можно оставить lrange + delete,
                    # но добавить небольшую паузу если пусто.
                    messages_bytes = self.redis_client.lrange(REDIS_PATH_LIST_KEY, 0, -1)
                    if messages_bytes:
                         # Очищаем список ПОСЛЕ успешного чтения
                         # TODO: Сделать атомарно через LMOVE или Lua скрипт в проде
                         # Для визуализации пока сойдет
                        pipe = self.redis_client.pipeline()
                        pipe.ltrim(REDIS_PATH_LIST_KEY, len(messages_bytes), -1) # Оставить все ПОСЛЕ прочитанных
                        pipe.execute()

                        # console.log(f"Read {len(messages_bytes)} messages from Redis.")
                        plot_updated_or_requested = self._process_message_batch(messages_bytes)
                    else:
                         # Нет сообщений, ждем немного
                         pass # Пауза будет ниже

                except redis.exceptions.ConnectionError as e:
                     console.print(f"[bold red]Redis connection error: {e}. Attempting to reconnect...[/bold red]")
                     self.redis_client = None
                     time.sleep(5)
                     while not self._connect_redis():
                          time.sleep(5)
                     continue # Пропустить остаток цикла и попробовать снова прочитать
                except Exception as e:
                     console.print(f"[bold red]Error during Redis operation or message processing: {e}[/bold red]")
                     console.print_exception(show_locals=False)
                     # Пауза чтобы избежать зацикливания при постоянной ошибке
                     time.sleep(1)


                # 3. Обработка событий GUI и пауза
                if self.fig and plt.fignum_exists(self.fig.number):
                    try:
                        self.fig.canvas.flush_events()
                    except NotImplementedError:
                        pass
                    except Exception as e:
                        console.print(f"[yellow]Warning: Error flushing GUI events: {e}[/yellow]")

                # Пауза: короче если было обновление, длиннее если нет
                sleep_time = 0.05 if plot_updated_or_requested else 0.2
                time.sleep(sleep_time)


        except KeyboardInterrupt:
            console.log("\nProcessing stopped by user (Ctrl+C).")
        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred in the main loop: {e}[/bold red]")
            console.print_exception(show_locals=False)
        finally:
            console.log("Path processing finished.")
            if self.fig and plt.fignum_exists(self.fig.number):
                # Финальное обновление с таблицей, если была активная эпоха
                if self.current_epoch != -1 and self.total_paths_processed_epoch > 0:
                    console.log(f"Displaying final state for Epoch {self.current_epoch}...")
                    try:
                         self.update_plot(draw_table=True)
                         self.fig.canvas.flush_events()
                    except Exception as final_draw_e:
                         console.print(f"[yellow]Warning: Error during final plot update: {final_draw_e}[/yellow]")
                elif not self.ax: # Если график так и не создался
                     console.print("[yellow]Plot window was likely not initialized.[/yellow]")

                if self.ax: # Только если оси существуют
                     console.log("Final plot displayed. Close the plot window to exit completely.")
                     plt.ioff()
                     plt.show() # Блокирующий вызов для просмотра

            if self.redis_client:
                self.redis_client.close()


if __name__ == "__main__":
    visualizer = PathVisualizer()
    visualizer.run()
# --- END OF FILE visualize_paths.py ---