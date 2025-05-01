# --- START OF FILE visualize_losses.py ---
import redis
import json
import time
import matplotlib.pyplot as plt
import os
from rich.console import Console
from collections import defaultdict

console = Console()

# --- Конфигурация ---
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
REDIS_PATH_LIST_KEY = 'spatial_graph:training_paths' # Ключ списка в Redis (тот же, что использует train.py)
UPDATE_EVERY_N_BATCHES = 1 # Обновлять график каждую N батчей
FIG_SIZE = (12, 8)

# Масштабирующие множители для визуализации на одном графике (приблизительные)
# ВНИМАНИЕ: Это приблизительное масштабирование, основанное на одном примере.
# Для точного масштабирования нужны известные диапазоны значений.
ACC_TARGET_SCALE = 100
AUX_VAR_LOSS_SCALE = 100

class LossVisualizer:
    def __init__(self):
        self.fig: plt.Figure = None
        self.ax: plt.Axes = None
        self.redis_client: redis.Redis = None

        self.current_epoch: int = -1
        self.batch_indices: list = []
        self.avg_losses: list = []
        self.acc_targets: list = []
        self.aux_var_losses: list = []
        self.aux_lb_losses: list = []
        self.batches_since_last_update: int = 0

        self._connect_redis()
        self._initialize_plot()

    def _connect_redis(self) -> bool:
        """Устанавливает соединение с Redis."""
        if self.redis_client:
            try:
                self.redis_client.ping()
                return True
            except redis.exceptions.ConnectionError:
                self.redis_client = None # Сброс при ошибке
        try:
            self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
            self.redis_client.ping()
            console.log(f"[Visualizer] Successfully connected to Redis {REDIS_HOST}:{REDIS_PORT}")
            return True
        except redis.exceptions.ConnectionError as e:
            console.print(f"[bold red][Visualizer] Redis Connection Error: {e}. Retrying...[/bold red]")
            self.redis_client = None
            return False

    def _initialize_plot(self):
        """Инициализирует окно Matplotlib."""
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            return # Уже инициализировано

        console.log("Initializing plot window...")
        plt.ion() # Включаем интерактивный режим
        self.fig, self.ax = plt.subplots(figsize=FIG_SIZE)
        self.ax.set_title("Training Metrics per Batch (Epoch -)")
        self.ax.set_xlabel("Batch Index")
        self.ax.set_ylabel("Metric Value")
        self.ax.grid(True)
        self.ax.legend() # Инициализируем пустую легенду

        plt.show(block=False)
        try:
            self.fig.canvas.flush_events()
        except NotImplementedError:
            pass
        console.log("Plot initialized.")

    def _clear_plot(self):
        """Очищает график для новой эпохи."""
        if self.ax:
            self.ax.clear()
            self.ax.set_title(f"Training Metrics per Batch (Epoch {self.current_epoch})")
            self.ax.set_xlabel("Batch Index")
            self.ax.set_ylabel("Metric Value")
            self.ax.grid(True)
            # Легенда будет добавлена при отрисовке данных
            self.batch_indices = []
            self.avg_losses = []
            self.acc_targets = []
            self.aux_var_losses = []
            self.aux_lb_losses = []
            self.batches_since_last_update = 0
            console.log(f"Plot cleared for Epoch {self.current_epoch}.")


    def _update_plot(self):
        """Обновляет график с текущими данными."""
        if not self.fig or not self.ax or not plt.fignum_exists(self.fig.number):
            return

        # Очищаем оси перед перерисовкой, но сохраняем заголовок и метки
        title = self.ax.get_title()
        xlabel = self.ax.get_xlabel()
        ylabel = self.ax.get_ylabel()
        self.ax.clear()
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True)


        if self.batch_indices:
            # Применяем примерное масштабирование для визуализации на одном графике
            # ВНИМАНИЕ: Это приблизительное масштабирование, основанное на одном примере.
            # Для точного масштабирования нужны известные диапазоны значений.
            scaled_acc_targets = [v * ACC_TARGET_SCALE for v in self.acc_targets]
            scaled_aux_var_losses = [v * AUX_VAR_LOSS_SCALE for v in self.aux_var_losses]

            # Отрисовываем каждую метрику разными цветами с прозрачным фоном
            line_avg_loss, = self.ax.plot(self.batch_indices, self.avg_losses, label='Avg Loss', color='blue')
            self.ax.fill_between(self.batch_indices, self.avg_losses, color='blue', alpha=0.1)

            line_acc_target, = self.ax.plot(self.batch_indices, scaled_acc_targets, label=f'Acc@Target (x{ACC_TARGET_SCALE})', color='green')
            self.ax.fill_between(self.batch_indices, scaled_acc_targets, color='green', alpha=0.1)

            line_aux_var_loss, = self.ax.plot(self.batch_indices, scaled_aux_var_losses, label=f'Aux Var Loss (x{AUX_VAR_LOSS_SCALE})', color='red')
            self.ax.fill_between(self.batch_indices, scaled_aux_var_losses, color='red', alpha=0.1)

            line_aux_lb_loss, = self.ax.plot(self.batch_indices, self.aux_lb_losses, label='Aux LB Loss', color='purple')
            self.ax.fill_between(self.batch_indices, self.aux_lb_losses, color='purple', alpha=0.1)


            # Добавляем легенду
            self.ax.legend()

            # --- Добавление линии тренда (линейная регрессия Avg Loss с цветом по тренду) ---
            if len(self.batch_indices) > 1:
                # Вычисляем линейную регрессию
                x = self.batch_indices
                y = self.avg_losses
                n = len(x)
                mean_x = sum(x) / n
                mean_y = sum(y) / n

                # Вычисляем числитель и знаменатель для наклона
                numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
                denominator = sum((x[i] - mean_x)**2 for i in range(n))

                trend_color = 'orange' # По умолчанию - стагнация
                if denominator != 0:
                    slope = numerator / denominator
                    intercept = mean_y - slope * mean_x

                    # Определяем цвет по наклону (тренда Avg Loss)
                    # Пороги могут потребовать настройки
                    SLOPE_THRESHOLD_GOOD = -0.001
                    SLOPE_THRESHOLD_BAD = 0.001

                    if slope < SLOPE_THRESHOLD_GOOD:
                        trend_color = 'green' # Хороший тренд (потери снижаются)
                    elif slope > SLOPE_THRESHOLD_BAD:
                        trend_color = 'red' # Плохой тренд (потери растут)
                    else:
                        trend_color = 'orange' # Стагнация

                    # Генерируем значения для линии тренда
                    trend_y = [slope * xi + intercept for xi in x]

                    # Отрисовываем линию тренда
                    self.ax.plot(x, trend_y, label='Trend (Avg Loss Linear)', color=trend_color, linestyle='-', linewidth=2)
                else:
                    # Если все x одинаковы (например, только одна точка), рисуем горизонтальную линию
                    self.ax.axhline(mean_y, color=trend_color, linestyle='-', linewidth=2, label='Trend (Avg Loss Linear)')

            # --- Конец добавления линии тренда ---

        self.fig.canvas.draw_idle() # Запрос на перерисовку

    def _process_message_batch(self, messages_bytes: list):
        """Обрабатывает пачку сообщений из Redis."""
        needs_plot_update = False

        for message_bytes in messages_bytes:
            try:
                message_data = json.loads(message_bytes.decode('utf-8'))
                epoch = message_data.get("epoch", -1)
                is_end_marker = message_data.get("status") == "epoch_end"
                batch_idx = message_data.get("batch_index", -1)

                if is_end_marker:
                    if epoch == self.current_epoch:
                        console.log(f"End of Epoch {self.current_epoch} marker received.")
                        # Финальное обновление графика для завершенной эпохи
                        self._update_plot()
                        needs_plot_update = True
                        # Сбрасываем данные для следующей эпохи
                        self.current_epoch = -1 # Ставим статус "ожидание"
                        self._clear_plot() # Очищаем график для следующей эпохи
                    continue # Маркер конца эпохи не содержит данных для отрисовки

                # Если пришло сообщение для новой эпохи
                if epoch != -1 and epoch != self.current_epoch:
                    if self.current_epoch != -1: # Если была активная эпоха, завершаем ее
                         console.log(f"New Epoch {epoch} detected. Finalizing Epoch {self.current_epoch}.")
                         self._update_plot() # Финальное обновление для старой эпохи
                         needs_plot_update = True
                         # time.sleep(0.5) # Дадим время посмотреть
                    self.current_epoch = epoch
                    self._clear_plot() # Очищаем график для новой эпохи
                    console.log(f"Starting visualization for Epoch {self.current_epoch}.")


                # Если сообщение относится к текущей активной эпохе
                if epoch == self.current_epoch and batch_idx != -1:
                    avg_loss = message_data.get("avg_loss")
                    acc_target = message_data.get("acc_target")
                    aux_var_loss = message_data.get("aux_var_loss")
                    aux_lb_loss = message_data.get("aux_lb_loss")

                    if avg_loss is not None and acc_target is not None and aux_var_loss is not None and aux_lb_loss is not None:
                        self.batch_indices.append(batch_idx)
                        self.avg_losses.append(avg_loss)
                        self.acc_targets.append(acc_target)
                        self.aux_var_losses.append(aux_var_loss)
                        self.aux_lb_losses.append(aux_lb_loss)
                        self.batches_since_last_update += 1
                        # console.log(f"Epoch {epoch}, Batch {batch_idx}: Loss={avg_loss:.4f}, Acc={acc_target:.4f}")

                        if self.batches_since_last_update >= UPDATE_EVERY_N_BATCHES:
                            self._update_plot()
                            needs_plot_update = True
                            self.batches_since_last_update = 0

            except json.JSONDecodeError as e:
                console.print(f"[red]Failed to decode JSON message: {e}[/red]")
            except Exception as e:
                console.print(f"[red]Error processing message: {e}[/red]")
                console.print_exception(show_locals=False)

        return needs_plot_update


    def run(self):
        """Основной цикл обработки."""
        while not self._connect_redis():
            time.sleep(5)

        console.log("Starting loss visualization loop...")

        try:
            while True:
                # 1. Проверка активности окна
                if self.fig and not plt.fignum_exists(self.fig.number):
                    console.log("Plot window closed by user. Exiting.")
                    break

                # 2. Чтение и обработка данных из Redis
                plot_updated_or_requested = False
                try:
                    # Читаем все доступные сообщения
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
            console.log("Loss visualization finished.")
            if self.fig and plt.fignum_exists(self.fig.number):
                 console.log("Final plot displayed. Close the plot window to exit completely.")
                 plt.ioff()
                 plt.show() # Блокирующий вызов для просмотра

            if self.redis_client:
                self.redis_client.close()


if __name__ == "__main__":
    visualizer = LossVisualizer()
    visualizer.run()
# --- END OF FILE visualize_losses.py ---