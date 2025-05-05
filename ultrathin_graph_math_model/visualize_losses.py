import redis
import json
import time
import matplotlib.pyplot as plt
import numpy as np # Добавлено для downsampling
import os
from rich.console import Console
from collections import defaultdict
from typing import List, Optional, Tuple, Dict, Any

console = Console()

# --- Конфигурация ---
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PATH_LIST_KEY = 'spatial_graph:training_paths'
UPDATE_EVERY_N_BATCHES = 15 # Увеличено для уменьшения зависаний
FIG_SIZE = (15, 6)

ACC_TARGET_SCALE = 100
AUX_VAR_LOSS_SCALE = 100

SLOPE_THRESHOLD_GOOD = -0.001
SLOPE_THRESHOLD_BAD = 0.001

# Ключи Redis для сохранения истории
LOSS_HISTORY_KEY = 'spatial_graph:overall_losses'
ACC_INDICES_HISTORY_KEY = 'spatial_graph:overall_batch_indices'
ACC_TARGETS_HISTORY_KEY = 'spatial_graph:overall_acc_targets'

# Настройки для оптимизации правого графика
OVERALL_PLOT_MAX_POINTS = 2500 # Макс. число точек для отображения на правом графике
OVERALL_PLOT_RECENT_POINTS = 500 # Число последних точек, которые всегда отображаются полностью

class LossVisualizer:
    def __init__(self):
        plt.ion()
        self.fig, (self.ax_batch, self.ax_overall_acc) = plt.subplots(1, 2, figsize=FIG_SIZE)

        # Настройка левого графика
        self.ax_batch.grid(True)
        self.ax_batch.set_title("Training Metrics per Batch (Waiting for Epoch)")
        self.ax_batch.set_xlabel("Batch Index (Current Epoch)")
        self.ax_batch.set_ylabel("Metric Value")

        # Настройка правого графика
        self.ax_overall_acc.grid(True)
        self.ax_overall_acc.set_title("Overall Accuracy per Batch")
        self.ax_overall_acc.set_xlabel("Global Batch Index")
        self.ax_overall_acc.set_ylabel("Accuracy (acc_target)")

        self.redis_client: Optional[redis.Redis] = None

        # Данные текущей эпохи
        self.current_epoch: int = -1
        self.batch_indices: List[int] = []
        self.avg_losses: List[float] = []
        self.acc_targets: List[float] = []
        self.aux_var_losses: List[float] = []
        self.aux_lb_losses: List[float] = []
        self.batches_since_last_update: int = 0
        self.current_epoch_sum_loss: float = 0.0
        self.current_epoch_batch_count: int = 0

        # Данные по завершенным эпохам (не рисуются, но сохраняются)
        self.overall_epochs: List[int] = []
        self.overall_losses: List[float] = []

        # Данные по всем батчам
        self.overall_batch_counter: int = 0
        self.overall_batch_indices: List[int] = []
        self.overall_acc_targets: List[float] = []

        # Флаг для проверки первого сообщения (для эвристики очистки истории)
        self.first_message_processed: bool = False

        self._connect_redis()
        self._load_overall_history()
        self._initialize_plot()

    def _connect_redis(self) -> bool:
        if self.redis_client:
            try:
                self.redis_client.ping()
                return True
            except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
                console.print(f"[yellow]Redis connection lost: {e}. Reconnecting...[/yellow]")
                self.redis_client = None
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                decode_responses=False, socket_timeout=10, socket_connect_timeout=5
            )
            self.redis_client.ping()
            console.log(f"[Visualizer] Successfully connected to Redis {REDIS_HOST}:{REDIS_PORT}")
            return True
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            console.print(f"[bold red][Visualizer] Redis Connection Error: {e}. Retrying...[/bold red]")
            self.redis_client = None
            return False
        except Exception as e:
            console.print(f"[bold red][Visualizer] Unexpected error connecting to Redis: {e}. Retrying...[/bold red]")
            self.redis_client = None
            return False

    def _load_overall_history(self):
        if not self.redis_client:
            console.print("[yellow]Cannot load history, Redis not connected.[/yellow]")
            return
        try:
            # Загрузка потерь по эпохам
            history_bytes_loss = self.redis_client.get(LOSS_HISTORY_KEY)
            if history_bytes_loss:
                history_data = json.loads(history_bytes_loss.decode('utf-8'))
                self.overall_epochs = [int(e) for e in history_data.get('epochs', [])]
                self.overall_losses = [float(l) for l in history_data.get('losses', [])]
                console.log(f"Loaded overall loss history: {len(self.overall_epochs)} epochs (not plotted).")
            else:
                console.log("No overall loss history found in Redis.")
                self.overall_epochs, self.overall_losses = [], []

            # Загрузка точности по батчам
            history_bytes_acc_indices = self.redis_client.get(ACC_INDICES_HISTORY_KEY)
            history_bytes_acc_targets = self.redis_client.get(ACC_TARGETS_HISTORY_KEY)

            if history_bytes_acc_indices and history_bytes_acc_targets:
                loaded_indices = json.loads(history_bytes_acc_indices.decode('utf-8'))
                loaded_targets = json.loads(history_bytes_acc_targets.decode('utf-8'))

                if loaded_indices and loaded_targets and len(loaded_indices) == len(loaded_targets):
                    self.overall_batch_indices = [int(i) for i in loaded_indices]
                    self.overall_acc_targets = [float(t) for t in loaded_targets]
                    self.overall_batch_counter = self.overall_batch_indices[-1] + 1 if self.overall_batch_indices else 0
                    console.log(f"Loaded overall accuracy history: {len(self.overall_batch_indices)} batches (Counter: {self.overall_batch_counter}).")
                else:
                    console.print("[yellow]Warning: Mismatched or empty overall accuracy history found. Resetting.[/yellow]")
                    self.overall_batch_indices, self.overall_acc_targets, self.overall_batch_counter = [], [], 0
            else:
                console.log("No overall accuracy history found in Redis.")
                self.overall_batch_indices, self.overall_acc_targets, self.overall_batch_counter = [], [], 0

        except Exception as e:
            console.print(f"[red]Error loading overall history: {e}[/red]")
            self.overall_epochs, self.overall_losses = [], []
            self.overall_batch_indices, self.overall_acc_targets, self.overall_batch_counter = [], [], 0

    def _save_overall_history(self):
        if not self.redis_client:
            console.print("[yellow]Cannot save history, Redis not connected.[/yellow]")
            return
        try:
            pipe = self.redis_client.pipeline()
            history_data_loss = {'epochs': self.overall_epochs, 'losses': self.overall_losses}
            pipe.set(LOSS_HISTORY_KEY, json.dumps(history_data_loss))
            pipe.set(ACC_INDICES_HISTORY_KEY, json.dumps(self.overall_batch_indices))
            pipe.set(ACC_TARGETS_HISTORY_KEY, json.dumps(self.overall_acc_targets))
            pipe.execute()
            console.log(f"Saved overall history: {len(self.overall_epochs)} epochs (loss), {len(self.overall_batch_indices)} batches (accuracy).")
        except Exception as e:
            console.print(f"[red]Error saving overall history: {e}[/red]")

    def _initialize_plot(self):
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            console.print("[bold red]Plot figure not available. Cannot initialize.[/bold red]")
            return
        try:
            self.ax_batch.clear()
            self.ax_overall_acc.clear()

            self.ax_batch.set_title("Training Metrics per Batch (Waiting for Epoch)")
            self.ax_batch.set_xlabel("Batch Index (Current Epoch)")
            self.ax_batch.set_ylabel("Metric Value")
            self.ax_batch.grid(True)
            self.ax_batch.legend()

            self.ax_overall_acc.set_title("Overall Accuracy per Batch")
            self.ax_overall_acc.set_xlabel("Global Batch Index")
            self.ax_overall_acc.set_ylabel("Accuracy (acc_target)")
            self.ax_overall_acc.grid(True)
            self._update_overall_accuracy_plot() # Нарисовать загруженные данные точности

            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            try:
                self.fig.canvas.flush_events()
            except NotImplementedError: pass
            plt.pause(0.01)
        except Exception as e:
            console.print(f"[bold red]Error initializing/clearing plot axes: {e}[/bold red]")

    def _clear_plot(self):
        """Очищает левый график (данные текущей эпохи)."""
        if not self.ax_batch or not self.fig or not plt.fignum_exists(self.fig.number): return
        try:
            self.ax_batch.clear()
            self.ax_batch.set_title(f"Training Metrics per Batch (Epoch {self.current_epoch})")
            self.ax_batch.set_xlabel("Batch Index (Current Epoch)")
            self.ax_batch.set_ylabel("Metric Value")
            self.ax_batch.grid(True)
            self.batch_indices, self.avg_losses, self.acc_targets = [], [], []
            self.aux_var_losses, self.aux_lb_losses = [], []
            self.batches_since_last_update = 0
            self.current_epoch_sum_loss, self.current_epoch_batch_count = 0.0, 0
            console.log(f"Cleared batch plot for Epoch {self.current_epoch}.")
        except Exception as e:
            console.print(f"[red]Error clearing batch plot: {e}[/red]")

    def _calculate_trend(self, x: List[int], y: List[float]) -> Optional[Tuple[List[float], str]]:
        if len(x) < 2: return None
        n = len(x)
        try:
            x_float = [float(xi) for xi in x]
            y_float = [float(yi) for yi in y]
            mean_x, mean_y = sum(x_float) / n, sum(y_float) / n
            numerator = sum((x_float[i] - mean_x) * (y_float[i] - mean_y) for i in range(n))
            denominator = sum((x_float[i] - mean_x)**2 for i in range(n))
            epsilon = 1e-9
            if abs(denominator) > epsilon:
                slope = numerator / denominator
                intercept = mean_y - slope * mean_x
                trend_y = [slope * xi + intercept for xi in x_float]
                trend_color = 'green' if slope < SLOPE_THRESHOLD_GOOD else ('red' if slope > SLOPE_THRESHOLD_BAD else 'orange')
            else:
                trend_y, trend_color = [mean_y] * n, 'grey'
            return trend_y, trend_color
        except (OverflowError, ValueError) as e:
             console.print(f"[yellow]Warning: Numerical issue calculating trend: {e}. Skipping trend line.[/yellow]")
             return None
        except Exception as e:
            console.print(f"[red]Error calculating trend: {e}[/red]")
            return None

    def _update_plot(self):
        """Обновляет левый график (батчи текущей эпохи)."""
        if not self.fig or not self.ax_batch or not plt.fignum_exists(self.fig.number): return
        try:
            title = self.ax_batch.get_title()
            xlabel = self.ax_batch.get_xlabel()
            ylabel = self.ax_batch.get_ylabel()
            self.ax_batch.clear()
            self.ax_batch.set_title(title); self.ax_batch.set_xlabel(xlabel); self.ax_batch.set_ylabel(ylabel)
            self.ax_batch.grid(True)

            if self.batch_indices:
                scaled_acc = [v * ACC_TARGET_SCALE for v in self.acc_targets]
                scaled_aux_var = [v * AUX_VAR_LOSS_SCALE for v in self.aux_var_losses]
                self.ax_batch.plot(self.batch_indices, self.avg_losses, label='Avg Loss', color='blue', alpha=0.8)
                self.ax_batch.plot(self.batch_indices, scaled_acc, label=f'Acc@Target (x{ACC_TARGET_SCALE})', color='green', alpha=0.8)
                self.ax_batch.plot(self.batch_indices, scaled_aux_var, label=f'Aux Var Loss (x{AUX_VAR_LOSS_SCALE})', color='red', alpha=0.8)
                self.ax_batch.plot(self.batch_indices, self.aux_lb_losses, label='Aux LB Loss', color='purple', alpha=0.8)
                trend_result = self._calculate_trend(self.batch_indices, self.avg_losses)
                if trend_result:
                    trend_y, trend_color = trend_result
                    self.ax_batch.plot(self.batch_indices, trend_y, label='Trend (Avg Loss)', color=trend_color, linestyle='--', linewidth=2)
                self.ax_batch.legend()
            self.fig.canvas.draw_idle()
        except Exception as e:
            console.print(f"[red]Error updating batch plot: {e}[/red]")

    def _get_downsampled_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает прореженные данные для общего графика точности."""
        indices = np.array(self.overall_batch_indices)
        targets = np.array(self.overall_acc_targets)
        n_points = len(indices)

        if n_points <= OVERALL_PLOT_MAX_POINTS:
            return indices, targets # Возвращаем все точки

        # Выбираем последние точки
        recent_indices = indices[-OVERALL_PLOT_RECENT_POINTS:]
        recent_targets = targets[-OVERALL_PLOT_RECENT_POINTS:]

        # Выбираем из старых точек
        older_indices = indices[:-OVERALL_PLOT_RECENT_POINTS]
        older_targets = targets[:-OVERALL_PLOT_RECENT_POINTS]
        n_older = len(older_indices)
        n_sample_older = OVERALL_PLOT_MAX_POINTS - OVERALL_PLOT_RECENT_POINTS

        if n_sample_older <= 0: # Если нужно показать только недавние
             return recent_indices, recent_targets

        # Прореживаем старые точки
        step = max(1, n_older // n_sample_older)
        sampled_older_indices = older_indices[::step]
        sampled_older_targets = older_targets[::step]

        # Объединяем старые прореженные и все недавние
        final_indices = np.concatenate((sampled_older_indices, recent_indices))
        final_targets = np.concatenate((sampled_older_targets, recent_targets))

        return final_indices, final_targets

    def _update_overall_accuracy_plot(self):
        """Обновляет правый график (точность по всем батчам) с downsampling."""
        if not self.fig or not self.ax_overall_acc or not plt.fignum_exists(self.fig.number): return
        try:
            self.ax_overall_acc.clear()
            self.ax_overall_acc.set_title("Overall Accuracy per Batch")
            self.ax_overall_acc.set_xlabel("Global Batch Index")
            self.ax_overall_acc.set_ylabel("Accuracy (acc_target)")
            self.ax_overall_acc.grid(True)

            if self.overall_batch_indices and self.overall_acc_targets:
                # Получаем прореженные данные
                plot_indices, plot_targets = self._get_downsampled_data()

                if len(plot_indices) > 0:
                    # Рисуем основные данные (прореженные)
                    self.ax_overall_acc.plot(plot_indices, plot_targets, label='Batch Accuracy', color='darkorange', linestyle='-', linewidth=1, alpha=0.7)

                    # Рассчитываем и рисуем сглаженное среднее на прореженных данных
                    window_size = min(50, len(plot_indices) // 2) # Адаптивное окно, не больше половины точек
                    if window_size >= 10: # Имеет смысл сглаживать, если окно достаточно большое
                        # Используем numpy для скользящего среднего (более эффективно)
                        smoothed_acc = np.convolve(plot_targets, np.ones(window_size)/window_size, mode='valid')
                        # Индексы для сглаженных данных (соответствуют концу окна)
                        smoothed_indices = plot_indices[window_size-1:]

                        if len(smoothed_indices) > 0:
                            self.ax_overall_acc.plot(smoothed_indices, smoothed_acc, label=f'Smoothed Acc (w={window_size})', color='red', linestyle='-', linewidth=1.5)

                    self.ax_overall_acc.legend()
                else:
                     # Если после прореживания ничего не осталось (маловероятно)
                     self.ax_overall_acc.legend()


            self.fig.canvas.draw_idle()
        except Exception as e:
            console.print(f"[red]Error updating overall accuracy plot: {e}[/red]")

    def _process_message_batch(self, messages_bytes: List[bytes]) -> Tuple[bool, bool]:
        """Обрабатывает пачку сообщений. Возвращает (needs_plot_update, epoch_just_finished)"""
        needs_plot_update = False
        epoch_finished_in_batch = False
        data_added = False

        for message_bytes in messages_bytes:
            try:
                message_str = message_bytes.decode('utf-8', errors='ignore')
                if not message_str: continue

                message_data: Dict[str, Any] = json.loads(message_str)
                epoch = message_data.get("epoch")
                is_end_marker = message_data.get("status") == "epoch_end"
                batch_idx = message_data.get("batch_index")

                # --- Эвристика очистки истории при первом сообщении ---
                if not self.first_message_processed:
                    self.first_message_processed = True
                    # Проверяем, если есть загруженная история и новая эпоха "слишком маленькая"
                    is_potential_restart = False
                    if epoch is not None and isinstance(epoch, int):
                         if self.overall_epochs and epoch < self.overall_epochs[-1]:
                              is_potential_restart = True
                         # Также считаем рестартом, если эпоха 0 и уже есть данные
                         elif epoch == 0 and (self.overall_epochs or self.overall_batch_indices):
                              is_potential_restart = True

                    if is_potential_restart:
                         console.print("[bold yellow]Detected potential training restart. Clearing loaded overall history.[/bold yellow]")
                         self.overall_epochs = []
                         self.overall_losses = []
                         self.overall_batch_indices = []
                         self.overall_acc_targets = []
                         self.overall_batch_counter = 0
                         # Сигнализируем о необходимости обновить (пустой) правый график
                         needs_plot_update = True
                # --- Конец эвристики ---

                if epoch is not None and not isinstance(epoch, int): continue
                if batch_idx is not None and not isinstance(batch_idx, int): continue

                # Обработка конца эпохи
                if is_end_marker:
                    if epoch is not None and epoch == self.current_epoch:
                        console.log(f"End of Epoch {self.current_epoch} marker received.")
                        if self.avg_losses:
                            last_avg_loss = self.avg_losses[-1]
                            console.log(f"Epoch {self.current_epoch} finished with last recorded avg_loss: {last_avg_loss:.4f} (data saved).")
                            if self.current_epoch not in self.overall_epochs:
                                self.overall_epochs.append(self.current_epoch)
                                self.overall_losses.append(last_avg_loss)
                            else: # Обновление
                                try: self.overall_losses[self.overall_epochs.index(self.current_epoch)] = last_avg_loss
                                except ValueError: self.overall_epochs.append(self.current_epoch); self.overall_losses.append(last_avg_loss)
                            epoch_finished_in_batch = True
                        else:
                            console.log(f"Epoch {self.current_epoch} finished marker received, but no batch losses were recorded.")
                    continue

                # Обработка начала новой эпохи / продолжение текущей
                if epoch is not None and epoch != self.current_epoch:
                    if self.current_epoch != -1 and self.avg_losses and not epoch_finished_in_batch:
                        console.log(f"[yellow]New Epoch {epoch} started before end marker for Epoch {self.current_epoch}. Finalizing Epoch {self.current_epoch} based on last batch.[/yellow]")
                        last_avg_loss = self.avg_losses[-1]
                        if self.current_epoch not in self.overall_epochs:
                            self.overall_epochs.append(self.current_epoch); self.overall_losses.append(last_avg_loss)
                        else:
                             try: self.overall_losses[self.overall_epochs.index(self.current_epoch)] = last_avg_loss
                             except ValueError: self.overall_epochs.append(self.current_epoch); self.overall_losses.append(last_avg_loss)
                        epoch_finished_in_batch = True

                    console.log(f"New Epoch {epoch} detected. Clearing batch plot.")
                    self.current_epoch = epoch
                    self._clear_plot()
                    needs_plot_update = True
                    epoch_finished_in_batch = False

                # Обработка данных батча
                if epoch == self.current_epoch and batch_idx is not None:
                    avg_loss = message_data.get("avg_loss")
                    acc_target = message_data.get("acc_target")
                    aux_var_loss = message_data.get("aux_var_loss")
                    aux_lb_loss = message_data.get("aux_lb_loss")

                    metrics_ok = (
                        avg_loss is not None and isinstance(avg_loss, (int, float)) and
                        aux_var_loss is not None and isinstance(aux_var_loss, (int, float)) and
                        aux_lb_loss is not None and isinstance(aux_lb_loss, (int, float))
                    )

                    if metrics_ok:
                        self.batch_indices.append(batch_idx)
                        self.avg_losses.append(float(avg_loss))
                        self.aux_var_losses.append(float(aux_var_loss))
                        self.aux_lb_losses.append(float(aux_lb_loss))

                        current_acc = None
                        if acc_target is not None and isinstance(acc_target, (int, float)):
                            current_acc = float(acc_target)
                            self.acc_targets.append(current_acc)
                            self.overall_batch_indices.append(self.overall_batch_counter)
                            self.overall_acc_targets.append(current_acc)
                        else:
                            self.acc_targets.append(0.0) # Placeholder

                        self.overall_batch_counter += 1
                        data_added = True
                        self.current_epoch_sum_loss += float(avg_loss)
                        self.current_epoch_batch_count += 1
                        self.batches_since_last_update += 1

                        if self.batches_since_last_update >= UPDATE_EVERY_N_BATCHES:
                            needs_plot_update = True
                            self.batches_since_last_update = 0
                    else:
                         if any(message_data.get(k) is not None for k in ["avg_loss", "acc_target", "aux_var_loss", "aux_lb_loss"]):
                             console.print(f"[yellow]Warning: Missing/invalid metric data Epoch {epoch}, Batch {batch_idx}. Skipping. Data: {message_data}[/yellow]")

            except json.JSONDecodeError as e:
                console.print(f"[red]Failed to decode JSON: {e} - Content: {message_bytes[:100]}...[/red]")
            except Exception as e:
                console.print(f"[red]Error processing message: {e}[/red]")

        if data_added and not needs_plot_update and self.batches_since_last_update > 0:
             needs_plot_update = True

        return needs_plot_update, epoch_finished_in_batch

    def run(self):
        while not self._connect_redis():
            console.log("Retrying Redis connection in 5 seconds...")
            time.sleep(5)

        if not self.fig or not self.ax_batch:
            console.print("[bold red]Failed to initialize plot. Exiting.[/bold red]")
            return

        console.log("Starting loss visualization loop...")
        last_save_time = time.time()
        SAVE_INTERVAL = 60

        try:
            while True:
                if not self.fig or not plt.fignum_exists(self.fig.number):
                    console.log("Plot window closed by user. Exiting.")
                    break

                needs_plot_update = False
                epoch_just_finished = False
                messages_bytes: List[bytes] = []

                try:
                    if not self.redis_client:
                        if not self._connect_redis(): time.sleep(5); continue
                        else: console.log("Redis reconnected during loop.")

                    pipe = self.redis_client.pipeline()
                    pipe.lrange(REDIS_PATH_LIST_KEY, 0, -1)
                    pipe.ltrim(REDIS_PATH_LIST_KEY, 0, 0) # Prepare atomic clear
                    results = pipe.execute()
                    messages_bytes = results[0]

                    if messages_bytes:
                        self.redis_client.ltrim(REDIS_PATH_LIST_KEY, len(messages_bytes), -1) # Actual clear
                        needs_plot_update, epoch_just_finished = self._process_message_batch(messages_bytes)

                except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
                    console.print(f"[bold red]Redis connection error: {e}. Retrying...[/bold red]")
                    self.redis_client = None; time.sleep(5); continue
                except Exception as e:
                    console.print(f"[bold red]Error during Redis/processing: {e}[/bold red]")
                    time.sleep(1)

                # Обновление графиков
                if needs_plot_update:
                    self._update_plot()
                    self._update_overall_accuracy_plot()

                # Сохранение истории
                if epoch_just_finished:
                    self._save_overall_history()
                    last_save_time = time.time()
                elif time.time() - last_save_time > SAVE_INTERVAL:
                     if self.overall_batch_indices or self.overall_epochs:
                         self._save_overall_history()
                         last_save_time = time.time()

                if not self.fig or not plt.fignum_exists(self.fig.number): break

                # Обновление GUI
                try:
                    if needs_plot_update or epoch_just_finished:
                        self.fig.canvas.flush_events()
                except Exception as e:
                    if not isinstance(e, NotImplementedError):
                        console.print(f"[yellow]Warning: Error flushing GUI events: {type(e).__name__} - {e}[/yellow]")

                sleep_time = 0.15 if (needs_plot_update or epoch_just_finished) else 0.5 # Немного увеличили паузу
                time.sleep(sleep_time)

        except (KeyboardInterrupt, SystemExit):
            console.log("\nProcessing stopped by user.")
        except Exception as e:
            console.print(f"[bold red]An unexpected error occurred in the main loop: {e}[/bold red]")
        finally:
            console.log("Loss visualization finishing...")
            if self.redis_client and (self.overall_epochs or self.overall_batch_indices):
                console.log("Performing final save of history...")
                self._save_overall_history()

            if self.fig and plt.fignum_exists(self.fig.number):
                console.log("Final plot displayed. Close the plot window to exit completely.")
                try: plt.ioff(); plt.show()
                except Exception as e: console.print(f"[yellow]Warning: Error showing final plot: {e}[/yellow]")
            if self.redis_client:
                try: self.redis_client.close(); console.log("Redis connection closed.")
                except Exception as e: console.print(f"[yellow]Warning: Error closing Redis connection: {e}[/yellow]")

if __name__ == "__main__":
    visualizer = LossVisualizer()
    visualizer.run()