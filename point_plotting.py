import pygame
import math
import asyncio
import platform
import json

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 1400, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Интерактивные точки")
font = pygame.font.SysFont('Segoe UI', 28, bold=True)
point_font = pygame.font.SysFont('Segoe UI', 16)
axis_label_font = pygame.font.SysFont('Segoe UI', 12)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Smart palette (now more saturated)
PASTEL_BG = (225, 230, 255)  # более яркий голубой фон
FRAME_COLOR = (60, 60, 130)  # насыщенная тёмно-синяя рамка
GRID_COLOR = (255, 60, 120, 100)  # яркая розово-красная сетка
GRID_MAIN_COLOR = (80, 80, 100)  # менее контрастная тёмно-серая сетка
AXIS_COLOR = (50, 50, 180)  # ярко-синий для осей
LABEL_COLOR = (30, 30, 120)  # насыщенный тёмно-синий для подписей
POINT_INPUT_COLOR = (255, 40, 40)  # ярко-красный для input
POINT_REGULAR_COLOR = (30, 120, 255)  # насыщенный синий для обычных
POINT_SELECTED_COLOR = (255, 180, 0)  # ярко-жёлтый для выделения
CONNECTION_DIRECTED_COLOR = (120, 0, 200)  # насыщенный фиолетовый
CONNECTION_UNDIRECTED_COLOR = (0, 180, 220)  # насыщенный голубой

def get_point_color(point, selected_point_id=None):
    if selected_point_id is not None and point['id'] == selected_point_id:
        return POINT_SELECTED_COLOR
    if point['type'] == 'input':
        return POINT_INPUT_COLOR
    return POINT_REGULAR_COLOR

def get_connection_color(conn_type):
    if conn_type == 'directed':
        return CONNECTION_DIRECTED_COLOR
    return CONNECTION_UNDIRECTED_COLOR

# Global variables
points = []
uid_counter = 0
current_mode = 'кубик'
max_distance = 50
scale = 1.0  # Initial scale (1.0 = 100%)
selected_point_id = None # For 'ветка-связи'
first_selected_directed_point_id = None # For 'направленная-связь'
input_active = None  # 'distance' or 'scale' or None
input_text = ''
offset_x, offset_y = 400, 300  # Center of the grid
velocity_x, velocity_y = 0, 0  # Panning velocities
acceleration = 0.5  # Pixels per frame acceleration
max_velocity = 10  # Max pixels per frame
zoom_speed = 0.1  # Scale change per mouse wheel tick

# For move and delete tools
move_dragging = False
move_point = None
move_offset = (0, 0)
last_click_time = 0  # For detecting double-click
last_click_id = None
double_click_threshold = 400  # ms for double-click

# Track manually deleted paths
deleted_connections = []  # list of dicts: {'type':..., 'from':..., 'to':...}

# Кнопки будут распределяться автоматически
button_labels = ['кубик-вход', 'кубик', 'ветка-связи', 'направленная-связь', 'перемещение', 'удаление', 'Сохранить', 'Загрузить из JSON']
mode_buttons = []  # Заполняется динамически в draw()

# Инпуты будут размещаться справа от кнопок, координаты обновляются в draw()
distance_input_rect = pygame.Rect(0, 0, 120, 40)
scale_input_rect = pygame.Rect(0, 0, 120, 40)

def world_to_screen(x, y):
    return (x * scale + offset_x, -y * scale + offset_y)

def screen_to_world(x, y):
    return ((x - offset_x) / scale, (offset_y - y) / scale)

def snap_to_grid(x, y):
    step = max_distance / 2
    grid_x = round(x / step) * step
    grid_y = round(y / step) * step
    return grid_x, grid_y

def add_point(x, y, type):
    global uid_counter
    point = {'id': uid_counter, 'x': x, 'y': y, 'type': type, 'auto_neighbors': set(), 'manual_neighbors': set(), 'directed_neighbors': set(), 'blocked_neighbors': set()}
    points.append(point)
    uid_counter += 1
    recalculate_auto_neighbors() # Recalculate for all points after adding

def recalculate_auto_neighbors():
    for point in points:
        point['auto_neighbors'].clear()
    for i, point1 in enumerate(points):
        for point2 in points[i+1:]:
            dist = math.hypot(point1['x'] - point2['x'], point1['y'] - point2['y'])
            if dist <= max_distance:
                # Проверяем, есть ли уже ручная, направленная или заблокированная автосвязь
                has_manual_or_directed = (
                    point2['id'] in point1.get('manual_neighbors', set()) or
                    point2['id'] in point1.get('directed_neighbors', set()) or
                    point1['id'] in point2.get('manual_neighbors', set()) or
                    point1['id'] in point2.get('directed_neighbors', set())
                )
                # skip blocked auto connections
                if point2['id'] in point1.get('blocked_neighbors', set()) or point1['id'] in point2.get('blocked_neighbors', set()):
                    continue
                if not has_manual_or_directed:
                    point1['auto_neighbors'].add(point2['id'])
                    point2['auto_neighbors'].add(point1['id'])

def recalculate_ids_and_neighbors():
    # Пересчитать id, auto_neighbors, manual_neighbors, directed_neighbors после удаления точки
    id_map = {old['id']: new_id for new_id, old in enumerate(points)}
    for idx, point in enumerate(points):
        point['id'] = idx
    for point in points:
        point['auto_neighbors'] = set(id_map[nid] for nid in point['auto_neighbors'] if nid in id_map)
        point['manual_neighbors'] = set(id_map[nid] for nid in point['manual_neighbors'] if nid in id_map)
        point['directed_neighbors'] = set(id_map[nid] for nid in point['directed_neighbors'] if nid in id_map)
        point['blocked_neighbors'] = set(id_map[nid] for nid in point['blocked_neighbors'] if nid in id_map)
    recalculate_auto_neighbors()

def get_point_at(x, y):
    for point in points:
        px, py = world_to_screen(point['x'], point['y'])
        if math.hypot(x - px, y - py) <= 10:
            return point
    return None

def get_hovered_connection(x, y, threshold=8):  # increased tolerance for auto lines
    # Returns (type, p1_id, p2_id) if mouse is near a connection segment
    for p1 in points:
        p1x, p1y = world_to_screen(p1['x'], p1['y'])
        # directed connections
        for nid in p1['directed_neighbors']:
            p2 = next((p for p in points if p['id']==nid), None)
            if p2:
                p2x, p2y = world_to_screen(p2['x'], p2['y'])
                # compute distance to segment
                dx, dy = p2x - p1x, p2y - p1y
                if dx==0 and dy==0: continue
                t = max(0, min(1, ((x-p1x)*dx + (y-p1y)*dy)/(dx*dx+dy*dy)))
                cx, cy = p1x + t*dx, p1y + t*dy
                if math.hypot(x-cx, y-cy) <= threshold:
                    return ('directed', p1['id'], p2['id'])
        # undirected manual connections to avoid duplicates
        for nid in p1['manual_neighbors']:
            if nid > p1['id']:
                p2 = next((p for p in points if p['id'] == nid), None)
                if p2:
                    p2x, p2y = world_to_screen(p2['x'], p2['y'])
                    dx, dy = p2x - p1x, p2y - p1y
                    if dx==0 and dy==0: continue
                    t = max(0, min(1, ((x-p1x)*dx + (y-p1y)*dy)/(dx*dx+dy*dy)))
                    cx, cy = p1x + t*dx, p1y + t*dy
                    if math.hypot(x-cx, y-cy) <= threshold:
                        return ('manual', p1['id'], p2['id'])
        # also check auto connections
        for nid in p1['auto_neighbors']:
            if nid > p1['id']:
                p2 = next((p for p in points if p['id']==nid), None)
                if p2:
                    p2x, p2y = world_to_screen(p2['x'], p2['y'])
                    dx, dy = p2x - p1x, p2y - p1y
                    if dx==0 and dy==0: continue
                    t = max(0, min(1, ((x-p1x)*dx + (y-p1y)*dy)/(dx*dx+dy*dy)))
                    cx, cy = p1x + t*dx, p1y + t*dy
                    if math.hypot(x-cx, y-cy) <= threshold:
                        return ('auto', p1['id'], p2['id'])
    return None

def save_points_to_json(filename="points.json"):
    # Проверка на пропущенные uid
    uids = sorted(p["id"] for p in points)
    if uids:
        missing = [i for i in range(uids[0], uids[-1] + 1) if i not in uids]
        if missing:
            print(f"Ошибка: отсутствуют точки с Uid: {missing}. Сохранение отменено.")
            return
    data = []
    for p in points:
        # Calculate all neighbors for path_neighbors
        auto_neighbors = p.get('auto_neighbors', set())
        manual_neighbors = p.get('manual_neighbors', set())
        directed_out_neighbors = p.get('directed_neighbors', set())
        # directed_in_neighbors = set(other_p['id'] for other_p in points if p['id'] in other_p.get('directed_neighbors', set())) # Removed incoming neighbors
        
        all_neighbors_set = auto_neighbors | manual_neighbors | directed_out_neighbors # Excluded directed_in_neighbors
        path_neighbors_list = list(sorted(all_neighbors_set))
        # Add exit neighbor (-1) for non-input points if exit not disabled
        if p['type'] != 'input' and not p.get('no_exit', False):
            path_neighbors_list.append(-1)
        # Remove duplicates and sort
        path_neighbors_list = sorted(set(path_neighbors_list))
        
        data.append({
            "uid": p["id"],
            "is_input": p["type"] == "input",
            "no_exit": p.get("no_exit", False),
            "manual_neighbors": list(sorted(manual_neighbors)), # Keep for loading compatibility
            "directed_neighbors": list(sorted(directed_out_neighbors)), # Keep for loading compatibility
            "path_neighbors": path_neighbors_list, # Add the combined list
            "x": round(p["x"], 1),
            "y": round(p["y"], 1)
        })
    output = {"dist": max_distance, "points": data}
    # Include record of deleted connections
    if deleted_connections:
        output['deleted_connections'] = deleted_connections.copy()
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

def load_points_from_json(filename="points.json"):
    global points, uid_counter, selected_point_id, first_selected_directed_point_id, max_distance, deleted_connections
    try:
        with open(filename, "r", encoding="utf-8") as f:
            raw = json.load(f)
        raw_deleted = []
        if isinstance(raw, dict) and "points" in raw:
            max_distance = raw.get("dist", max_distance)
            data = raw.get("points", [])
            raw_deleted = raw.get("deleted_connections", [])
        else:
            data = raw
            raw_deleted = []
        points = []
        uid_counter = 0
        for item in data:
            p = {
                'id': item['uid'],
                'type': 'input' if item.get('is_input', False) else 'normal',
                'x': item['x'],
                'y': item['y'],
                'auto_neighbors': set(), # Will be recalculated
                'manual_neighbors': set(item.get('manual_neighbors', [])),
                'directed_neighbors': set(item.get('directed_neighbors', [])),
                'blocked_neighbors': set(),
                'no_exit': item.get('no_exit', False)
            }
            points.append(p)
            uid_counter = max(uid_counter, item['uid']+1)
        # Initialize auto_neighbors then apply blocked from saved deletions
        recalculate_auto_neighbors()
        deleted_connections = raw_deleted.copy()
        # Apply blocks to prevent reconnecting
        for entry in deleted_connections:
            typ, i1, i2 = entry['type'], entry['from'], entry['to']
            p1 = next((p for p in points if p['id']==i1), None)
            p2 = next((p for p in points if p['id']==i2), None)
            if p1 and p2:
                p1['blocked_neighbors'].add(i2)
                p2['blocked_neighbors'].add(i1)
        recalculate_auto_neighbors()
        selected_point_id = None
        first_selected_directed_point_id = None
    except Exception as e:
        print(f"Ошибка загрузки: {e}")

def draw_arrow(surface, color, start, end, arrow_size=20, amplitude=5, frequency=2): # Увеличен размер стрелки
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return # Avoid division by zero if start and end are the same
    angle = math.atan2(dy, dx)

    # Calculate points for the wavy line
    points_list = []
    num_segments = 20 # Number of segments for the curve
    for i in range(num_segments + 1):
        t = i / num_segments # Parameter from 0 to 1
        # Position along the straight line
        x_line = start[0] + t * dx
        y_line = start[1] + t * dy
        # Perpendicular offset using sine wave
        offset = amplitude * math.sin(t * frequency * 2 * math.pi)
        # Apply offset perpendicular to the line direction
        # Perpendicular vector is (-dy/length, dx/length)
        x_wave = x_line - offset * (dy / length)
        y_wave = y_line + offset * (dx / length)
        points_list.append((x_wave, y_wave))

    # Draw the wavy line using line segments
    if len(points_list) > 1:
        pygame.draw.lines(surface, color, False, points_list, 2)

    # Draw the arrowhead at the end point (using the original end point for placement)
    # Arrowhead direction should be based on the last segment of the wavy line if possible,
    # but using the overall angle is simpler and usually sufficient.
    arrow_angle = math.atan2(start[1] - end[1], start[0] - end[0]) # Use original angle for arrowhead orientation
    point1 = (end[0] + arrow_size * math.cos(arrow_angle - math.pi/6),
              end[1] + arrow_size * math.sin(arrow_angle - math.pi/6))
    point2 = (end[0] + arrow_size * math.cos(arrow_angle + math.pi/6),
              end[1] + arrow_size * math.sin(arrow_angle + math.pi/6))
    pygame.draw.polygon(surface, color, (end, point1, point2))

def draw():
    if not pygame.display.get_init() or not screen:
        return
    screen.fill(PASTEL_BG)
    # Draw border (рамка)
    pygame.draw.rect(screen, FRAME_COLOR, (0, 0, WIDTH, HEIGHT), 8, border_radius=0)

    # Draw red transparent grid if Shift is held
    mods = pygame.key.get_mods()
    if mods & pygame.KMOD_SHIFT:
        grid_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        step = max_distance / 2 * scale
        x0_world = screen_to_world(0, 0)[0]
        x1_world = screen_to_world(WIDTH, 0)[0]
        y0_world = screen_to_world(0, HEIGHT)[1]
        y1_world = screen_to_world(0, 0)[1]  # Top of screen world y

        # Draw vertical lines
        start_x = math.floor(x0_world / (max_distance/2)) * (max_distance/2)
        for i in range(int((x1_world - start_x) / (max_distance/2)) + 2):
             x = start_x + i * (max_distance/2)
             sx, _ = world_to_screen(x, 0)
             pygame.draw.line(grid_surface, GRID_COLOR, (sx, 0), (sx, HEIGHT))

        # Draw horizontal lines
        step_world = max_distance / 2
        start_y = math.floor(y0_world / step_world) * step_world
        for i in range(int((y1_world - start_y) / step_world) + 2):
            y = start_y + i * step_world
            _, sy = world_to_screen(0, y)
            pygame.draw.line(grid_surface, GRID_COLOR, (0, sy), (WIDTH, sy))

        screen.blit(grid_surface, (0, 0))


    # Draw axes
    pygame.draw.line(screen, AXIS_COLOR, (0, offset_y), (WIDTH, offset_y), 2)  # X-axis
    pygame.draw.line(screen, AXIS_COLOR, (offset_x, 0), (offset_x, HEIGHT), 2)  # Y-axis
    # Draw axis labels and grid lines
    grid_step_display = 50 * scale
    if grid_step_display > 20: # Only draw labels and major grid lines if scale is sufficient
        x_start_world = screen_to_world(0, 0)[0]
        x_end_world = screen_to_world(WIDTH, 0)[0]
        y_start_world = screen_to_world(0, HEIGHT)[1]
        y_end_world = screen_to_world(0, 0)[1]

        # Draw vertical major grid lines and X labels
        major_grid_world_step = 50
        start_x_major = math.ceil(x_start_world / major_grid_world_step) * major_grid_world_step
        for i in range(int((x_end_world - start_x_major) / major_grid_world_step) + 2):
            x_world = start_x_major + i * major_grid_world_step
            sx, _ = world_to_screen(x_world, 0)
            pygame.draw.line(screen, GRID_MAIN_COLOR, (sx, 0), (sx, HEIGHT))
            if x_world != 0:
                 screen.blit(axis_label_font.render(str(int(x_world/50)), True, LABEL_COLOR), (sx - 10, offset_y + 10))

        # Draw horizontal major grid lines and Y labels
        start_y_major = math.ceil(y_start_world / major_grid_world_step) * major_grid_world_step
        for i in range(int((y_end_world - start_y_major) / major_grid_world_step) + 2):
            y_world = start_y_major + i * major_grid_world_step
            _, sy = world_to_screen(0, y_world)
            pygame.draw.line(screen, GRID_MAIN_COLOR, (0, sy), (WIDTH, sy))
            if y_world != 0:
                 screen.blit(axis_label_font.render(str(int(y_world/50)), True, LABEL_COLOR), (offset_x + 10, sy - 10))


    # Draw connections
    drawn_connections = set() # To avoid drawing the same undirected line twice
    for point1 in points:
        p1x, p1y = world_to_screen(point1['x'], point1['y'])

        # Draw directed connections FROM point1
        for neighbor_id in point1['directed_neighbors']:
            point2 = next((p for p in points if p['id']==neighbor_id), None)
            if point2:
                p2x, p2y = world_to_screen(point2['x'], point2['y'])
                draw_arrow(screen, get_connection_color('directed'), (p1x, p1y), (p2x, p2y))
                # Mark this pair as having a directed connection for drawing purposes
                drawn_connections.add(tuple(sorted((point1['id'], point2['id']))))

    # Draw undirected connections (auto and manual)
    for point1 in points:
         p1x, p1y = world_to_screen(point1['x'], point1['y'])
         undirected_neighbors = point1['auto_neighbors'] | point1['manual_neighbors']
         for neighbor_id in undirected_neighbors:
             point2 = next((p for p in points if p['id'] == neighbor_id), None)
             if point2:
                 # Only draw if no directed connection exists between this pair and we haven't drawn it from the other side
                 if tuple(sorted((point1['id'], point2['id']))) not in drawn_connections:
                     p2x, p2y = world_to_screen(point2['x'], point2['y'])
                     pygame.draw.line(screen, get_connection_color('undirected'), (p1x, p1y), (p2x, p2y), 2) # Мягкая линия для undirected
                     drawn_connections.add(tuple(sorted((point1['id'], point2['id']))))


    # Draw points and labels
    for point in points:
        px, py = world_to_screen(point['x'], point['y'])
        color = get_point_color(point, selected_point_id)
        pygame.draw.circle(screen, color, (px, py), 7 if point['id'] == selected_point_id else 5)
        # Count all neighbors (auto, manual, and directed - either to or from)
        all_neighbors_count = len(point['auto_neighbors'] | point['manual_neighbors'] | point['directed_neighbors'] | set(p['id'] for p in points if point['id'] in p['directed_neighbors']))

        text = point_font.render(f"ID:{point['id']}, CN:{all_neighbors_count}", True, BLACK)
        # Смещаем подпись вправо и чуть выше точки
        screen.blit(text, (px + 10, py - 18))
        # Draw no-exit indicator (small red square)
        if point.get('no_exit', False) and point['type'] != 'input':
            size = 8
            rect = pygame.Rect(px - size, py - size, size*2, size*2)
            pygame.draw.rect(screen, RED, rect, 2)

    # Highlight hovered element in delete mode
    if current_mode == 'удаление':
        mx, my = pygame.mouse.get_pos()
        hp = get_point_at(mx, my)
        if hp:
            hx, hy = world_to_screen(hp['x'], hp['y'])
            size = 10
            hr = pygame.Rect(hx-size, hy-size, size*2, size*2)
            pygame.draw.rect(screen, FRAME_COLOR, hr, 2)
        else:
            hc = get_hovered_connection(mx, my)
            if hc:
                _, id1, id2 = hc
                p1 = next(p for p in points if p['id']==id1)
                p2 = next(p for p in points if p['id']==id2)
                x1, y1 = world_to_screen(p1['x'], p1['y'])
                x2, y2 = world_to_screen(p2['x'], p2['y'])
                left = min(x1, x2) - 5; top = min(y1, y2) - 5
                w = abs(x2 - x1) + 10; h = abs(y2 - y1) + 10
                rr = pygame.Rect(left, top, w, h)
                pygame.draw.rect(screen, FRAME_COLOR, rr, 2)
    # Highlight hovered element when no tool selected
    if current_mode is None:
        mx, my = pygame.mouse.get_pos()
        hp = get_point_at(mx, my)
        if hp:
            hx, hy = world_to_screen(hp['x'], hp['y'])
            size = 10
            hr = pygame.Rect(hx - size, hy - size, size*2, size*2)
            pygame.draw.rect(screen, FRAME_COLOR, hr, 2)
        else:
            hc = get_hovered_connection(mx, my)
            if hc:
                _, id1, id2 = hc
                p1 = next(p for p in points if p['id']==id1)
                p2 = next(p for p in points if p['id']==id2)
                x1, y1 = world_to_screen(p1['x'], p1['y'])
                x2, y2 = world_to_screen(p2['x'], p2['y'])
                left = min(x1, x2) - 5; top = min(y1, y2) - 5
                w = abs(x2 - x1) + 10; h = abs(y2 - y1) + 10
                rr = pygame.Rect(left, top, w, h)
                pygame.draw.rect(screen, FRAME_COLOR, rr, 2)

    # Draw selected point highlight for 'ветка-связи'
    if current_mode == 'ветка-связи' and selected_point_id is not None:
        selected_point = next((p for p in points if p['id'] == selected_point_id), None)
        if selected_point:
            px, py = world_to_screen(selected_point['x'], selected_point['y'])
            pygame.draw.circle(screen, GREEN, (px, py), 15, 2)

    # Draw selected point highlight for 'направленная-связь'
    if current_mode == 'направленная-связь' and first_selected_directed_point_id is not None:
        selected_point = next((p for p in points if p['id'] == first_selected_directed_point_id), None)
        if selected_point:
            px, py = world_to_screen(selected_point['x'], selected_point['y'])
            pygame.draw.circle(screen, (255, 165, 0), (px, py), 15, 2) # Orange highlight

    # Кнопки теперь вертикально в левой панели с переносом текста
    global mode_buttons, distance_input_rect, scale_input_rect
    mode_buttons = []
    button_count = len(button_labels)
    panel_x = 10
    panel_y = 10
    panel_width = 210  # Ширина красного прямоугольника минус отступы
    button_width = panel_width - 16  # 8px отступ с каждой стороны
    button_y = panel_y + 8
    mouse_pos = pygame.mouse.get_pos()
    mouse_pressed = pygame.mouse.get_pressed()[0]
    for label in button_labels:
        # --- Перенос текста по ширине ---
        def word_wrap(text, font, max_width):
            words = text.split(' ')
            lines = []
            current = ''
            for word in words:
                test = current + (' ' if current else '') + word
                if font.size(test)[0] <= max_width:
                    current = test
                else:
                    if current:
                        lines.append(current)
                    current = word
            if current:
                lines.append(current)
            return lines
        lines = word_wrap(label, font, button_width - 12)
        button_height = 14 + len(lines) * (font.get_height() + 2)
        rect = pygame.Rect(panel_x + 8, button_y, button_width, button_height)
        mode_buttons.append({'rect': rect, 'label': label})
        # Цвета и стиль
        is_active = (label == current_mode)
        hovered = rect.collidepoint(mouse_pos)
        pressed = hovered and mouse_pressed
        color_bg = (220, 220, 240)
        color_border = (100, 100, 120)
        color_text = (40, 40, 60)
        if is_active:
            color_bg = (120, 180, 255)
            color_border = (0, 90, 220)
            color_text = (0, 0, 0)
        if hovered:
            color_bg = (180, 210, 255) if is_active else (200, 230, 255)
            color_border = (40, 140, 255)
        if pressed:
            color_bg = (100, 150, 200) if is_active else (170, 200, 230)
            color_border = (0, 60, 120)
        pygame.draw.rect(screen, color_bg, rect, border_radius=8)
        pygame.draw.rect(screen, color_border, rect, 2, border_radius=8)
        # --- Рисуем текст с переносом ---
        for i, line in enumerate(lines):
            text_surf = font.render(line, True, color_text)
            text_rect = text_surf.get_rect(center=(rect.centerx, rect.y + 8 + i * (font.get_height() + 2) + font.get_height() // 2))
            screen.blit(text_surf, text_rect)
        button_y += button_height + 8  # Отступ между кнопками

    # Инпуты над кнопками справа
    input_margin = 20 # Adjusted margin
    input_height = button_height
    input_y = button_y - input_height - 15  # Adjusted vertical offset
    input_width = 120
    distance_input_rect = pygame.Rect(WIDTH - 2*input_margin - 2*input_width, input_y, input_width, input_height)
    scale_input_rect = pygame.Rect(WIDTH - input_margin - input_width, input_y, input_width, input_height)

    pygame.draw.rect(screen, (245,245,255), distance_input_rect, border_radius=10) # Smaller radius
    pygame.draw.rect(screen, BLACK, distance_input_rect, 2, border_radius=10)
    distance_surface = font.render(input_text if input_active == 'distance' else f"dist: {max_distance}", True, BLACK)
    screen.blit(distance_surface, (distance_input_rect.x + 10, distance_input_rect.y + 8))

    pygame.draw.rect(screen, (245,245,255), scale_input_rect, border_radius=10) # Smaller radius
    pygame.draw.rect(screen, BLACK, scale_input_rect, 2, border_radius=10)
    scale_surface = font.render(input_text if input_active == 'scale' else f"scale: {round(scale, 2)}", True, BLACK)
    screen.blit(scale_surface, (scale_input_rect.x + 10, scale_input_rect.y + 8))

    pygame.display.flip()

def update_loop():
    global current_mode, selected_point_id, first_selected_directed_point_id, input_active, input_text, max_distance, scale, offset_x, offset_y
    global velocity_x, velocity_y, move_dragging, move_point, move_offset, last_click_time, last_click_id, double_click_threshold
    # Panning logic
    keys = pygame.key.get_pressed()
    accelerating = False
    if not input_active and current_mode not in ['перемещение', 'удаление']:
        if keys[pygame.K_a]:  # Left (A)
            velocity_x = min(velocity_x + acceleration, max_velocity)
            accelerating = True
        if keys[pygame.K_d]:  # Right (D)
            velocity_x = max(velocity_x - acceleration, -max_velocity)
            accelerating = True
        if keys[pygame.K_w]:  # Up (W)
            velocity_y = min(velocity_y + acceleration, max_velocity)
            accelerating = True
        if keys[pygame.K_s]:  # Down (S)
            velocity_y = max(velocity_y - acceleration, -max_velocity)
            accelerating = True
    # Apply velocity to offsets
    offset_x += velocity_x
    offset_y += velocity_y
    # Sharp stop if no keys are pressed
    if not accelerating:
        velocity_x = 0
        velocity_y = 0
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                x, y = event.pos
                # Check button clicks
                button_clicked = False
                for button in mode_buttons:
                    if button['rect'].collidepoint(x, y):
                        button_clicked = True
                        if button['label'] == 'Сохранить':
                            save_points_to_json()
                            print('Точки сохранены в points.json')
                        elif button['label'] == 'Загрузить из JSON':
                            load_points_from_json()
                            print('Загружено из points.json')
                        elif button['label'] == current_mode:
                            # Deactivate current mode to allow only double-click logic
                            current_mode = None
                            input_active = None
                            move_dragging = False
                            move_point = None
                            selected_point_id = None
                            first_selected_directed_point_id = None
                        else:
                            current_mode = button['label']
                            input_active = None
                            move_dragging = False
                            move_point = None
                            selected_point_id = None
                            first_selected_directed_point_id = None
                        break

                if not button_clicked:
                    # Check input clicks
                    if distance_input_rect.collidepoint(x, y):
                        input_active = 'distance'
                        input_text = str(max_distance) # Pre-fill with current value
                    elif scale_input_rect.collidepoint(x, y):
                        input_active = 'scale'
                        input_text = str(round(scale, 2)) # Pre-fill with current value
                    else: # Click not on UI, graph area
                        wx, wy = screen_to_world(x, y)
                        mods = pygame.key.get_mods()
                        clicked_point = get_point_at(x, y)

                        if current_mode in ['кубик', 'кубик-вход']:
                            type = 'input' if current_mode == 'кубик-вход' else 'normal'
                            if mods & pygame.KMOD_SHIFT:
                                wx, wy = snap_to_grid(wx, wy)
                            add_point(wx, wy, type)

                        elif current_mode == 'ветка-связи':
                            if clicked_point:
                                if selected_point_id is None:
                                    selected_point_id = clicked_point['id']
                                elif selected_point_id != clicked_point['id']:
                                    point1 = next((p for p in points if p['id'] == selected_point_id), None)
                                    point2 = clicked_point
                                    if point1 and point2:
                                        # Toggle manual connection
                                        if point2['id'] in point1['manual_neighbors']:
                                            point1['manual_neighbors'].remove(point2['id'])
                                            point2['manual_neighbors'].remove(point1['id'])
                                        else:
                                            point1['manual_neighbors'].add(point2['id'])
                                            point2['manual_neighbors'].add(point1['id'])
                                    selected_point_id = None # Reset selection

                        elif current_mode == 'направленная-связь':
                            if clicked_point:
                                if first_selected_directed_point_id is None:
                                    first_selected_directed_point_id = clicked_point['id']
                                elif first_selected_directed_point_id != clicked_point['id']:
                                    point1 = next((p for p in points if p['id'] == first_selected_directed_point_id), None)
                                    point2 = clicked_point
                                    if point1 and point2:
                                        # Add directed connection A->B
                                        point1['directed_neighbors'].add(point2['id'])
                                        # Remove any manual undirected connection between A and B
                                        point1['manual_neighbors'].discard(point2['id'])
                                        point2['manual_neighbors'].discard(point1['id'])
                                        # Remove directed connection B->A if it existed
                                        point2['directed_neighbors'].discard(point1['id'])
                                    first_selected_directed_point_id = None # Reset selection
                                else: # Clicked the same point twice
                                    first_selected_directed_point_id = None # Deselect

                        elif current_mode == 'перемещение':
                            if clicked_point:
                                move_dragging = True
                                move_point = clicked_point
                                # Calculate offset from point center to mouse click
                                px, py = world_to_screen(clicked_point['x'], clicked_point['y'])
                                move_offset = (px - x, py - y)

                        elif current_mode == 'удаление':
                            # delete clicked point first; if none, delete hovered connection
                            if clicked_point:
                                # delete clicked point and all its connections
                                points.remove(clicked_point)
                                recalculate_ids_and_neighbors()
                                # clear selections if deleted point was selected
                                if selected_point_id == clicked_point['id']:
                                    selected_point_id = None
                                if first_selected_directed_point_id == clicked_point['id']:
                                    first_selected_directed_point_id = None
                            else:
                                conn = get_hovered_connection(x, y)
                                if conn:
                                    typ, i1, i2 = conn
                                    p1 = next((p for p in points if p['id']==i1), None)
                                    p2 = next((p for p in points if p['id']==i2), None)
                                    if p1 and p2:
                                        entry = {'type': typ, 'from': i1, 'to': i2}
                                        deleted_connections.append(entry)
                                        if typ == 'directed':
                                            p1['directed_neighbors'].discard(i2)
                                        elif typ == 'manual':
                                            # remove and block future auto-reconnection
                                            p1['manual_neighbors'].discard(i2)
                                            p2['manual_neighbors'].discard(i1)
                                            p1['blocked_neighbors'].add(i2)
                                            p2['blocked_neighbors'].add(i1)
                                            recalculate_auto_neighbors()
                                        elif typ == 'auto':
                                            p1['blocked_neighbors'].add(i2)
                                            p2['blocked_neighbors'].add(i1)
                                            recalculate_auto_neighbors()

                            # Double-click toggle no-exit for non-input points
                            current_time = pygame.time.get_ticks()
                            if clicked_point and clicked_point['type'] != 'input':
                                if last_click_id == clicked_point['id'] and current_time - last_click_time <= double_click_threshold:
                                    clicked_point['no_exit'] = not clicked_point.get('no_exit', False)
                                    last_click_time = 0
                                    last_click_id = None
                                else:
                                    last_click_time = current_time
                                    last_click_id = clicked_point['id']
                        elif current_mode is None:
                            # Double-click toggle no-exit for non-input points
                            current_time = pygame.time.get_ticks()
                            if clicked_point and clicked_point['type'] != 'input':
                                if last_click_id == clicked_point['id'] and current_time - last_click_time <= double_click_threshold:
                                    clicked_point['no_exit'] = not clicked_point.get('no_exit', False)
                                    last_click_time = 0
                                    last_click_id = None
                                else:
                                    last_click_time = current_time
                                    last_click_id = clicked_point['id']

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and current_mode == 'перемещение' and move_dragging:
                move_dragging = False
                move_point = None
                recalculate_auto_neighbors() # Recalculate auto neighbors after moving a point

        elif event.type == pygame.MOUSEMOTION:
            if move_dragging and current_mode == 'перемещение' and move_point:
                x, y = event.pos
                # Apply offset to mouse position to get the new screen position of the point center
                new_px, new_py = x + move_offset[0], y + move_offset[1]
                # Convert new screen position back to world coordinates
                wx, wy = screen_to_world(new_px, new_py)

                mods = pygame.key.get_mods()
                if mods & pygame.KMOD_SHIFT:
                    wx, wy = snap_to_grid(wx, wy)

                move_point['x'] = wx
                move_point['y'] = wy

        elif event.type == pygame.MOUSEWHEEL:
            if not input_active:
                x, y = pygame.mouse.get_pos()
                # Get world coordinates of cursor before zoom
                wx, wy = screen_to_world(x, y)
                # Apply scale change
                delta = event.y * zoom_speed
                new_scale = scale + delta
                if 0.1 <= new_scale <= 10.0: # Limit zoom
                    scale = new_scale
                    # Adjust offsets to keep cursor's world position fixed
                    new_screen_x = wx * scale + offset_x
                    new_screen_y = -wy * scale + offset_y
                    offset_x += x - new_screen_x
                    offset_y += y - new_screen_y
                    # Recalculate auto neighbors as distance threshold changes relative to screen
                    recalculate_auto_neighbors()


        elif event.type == pygame.KEYDOWN:
            if input_active:
                if event.key == pygame.K_RETURN:
                    if input_active == 'distance':
                        try:
                            new_distance = float(input_text)
                            if new_distance > 0: # Ensure distance is positive
                                max_distance = new_distance
                                recalculate_ids_and_neighbors() # Recalculate all neighbors based on new distance
                        except ValueError:
                            print("Неверный ввод для дистанции.")
                        input_active = None
                        input_text = ''
                    elif input_active == 'scale':
                        try:
                            new_scale = float(input_text)
                            if 0.1 <= new_scale <= 10.0: # Apply scale limits
                                old_scale = scale
                                scale = new_scale
                                # Adjust offsets to keep the center of the screen roughly in the same world location
                                offset_x = WIDTH / 2 - (WIDTH / 2 - offset_x) * (scale / old_scale)
                                offset_y = HEIGHT / 2 - (HEIGHT / 2 - offset_y) * (scale / old_scale)
                                recalculate_auto_neighbors() # Recalculate auto neighbors after scale change
                            else:
                                print("Масштаб должен быть от 0.1 до 10.0.")
                        except ValueError:
                            print("Неверный ввод для масштаба.")
                        input_active = None
                        input_text = ''
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                elif event.unicode.isdigit() or (event.unicode == '.' and '.' not in input_text):
                    input_text += event.unicode
            # Allow panning with arrow keys or WASD even when not in a point placement mode
            # This is handled by checking keys state in update_loop, but keydown can stop input.
            elif event.key == pygame.K_ESCAPE:
                 if selected_point_id is not None:
                     selected_point_id = None
                 if first_selected_directed_point_id is not None:
                     first_selected_directed_point_id = None


    return True

async def main():
    running = True
    while running:
        running = update_loop()
        draw()
        await asyncio.sleep(1.0 / 60)

if platform.system() == "Emscripten":
    # Running in a web browser
    asyncio.ensure_future(main())
else:
    # Running as a native application
    if __name__ == "__main__":
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("Завершено по Ctrl+C")
        finally:
            pygame.quit() # Ensure pygame quits cleanly on exit