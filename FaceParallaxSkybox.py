import cv2
import numpy as np
import pygame
import mediapipe as mp
from collections import deque
import os

# Initialize Pygame
pygame.init()

# Get display info and set window to nearly full screen size
display_info = pygame.display.Info()
WIDTH, HEIGHT = display_info.current_w - 100, display_info.current_h - 100

# Create resizable windowed mode
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("3D Skybox Window - Off-Axis Projection")
clock = pygame.time.Clock()

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# 3D Grid settings
GRID_SIZE = 10  # 10x10 grid
GRID_DEPTH = 1.5  # Depth in meters (scene depth)

# Skybox settings
SKYBOX_DISTANCE = 50.0  # How far away the skybox appears (meters)
SKYBOX_SCALE = 60.0  # Size of each skybox face (meters)

# Screen physical dimensions (approximate for a typical laptop/monitor)
SCREEN_WIDTH_M = 0.285  # meters (approx 11.2 inches for typical laptop)
SCREEN_HEIGHT_M = 0.160  # meters (approx 6.3 inches)

# Camera position on screen (typically top center)
CAMERA_OFFSET_X = 0.0  # meters from center
CAMERA_OFFSET_Y = -SCREEN_HEIGHT_M / 2  # top of screen

# Face tracking
face_position = [0.5, 0.5]
face_depth = 0.4  # Distance from screen in meters (estimated)
face_smoothing = deque(maxlen=10)
parallax_sensitivity = 1.0  # Runtime adjustable sensitivity

# Global skybox textures (loaded once at startup)
skybox_textures = {}


def load_skybox_textures(skybox_folder="clouds1"):
    """
    Load all 6 skybox cubemap faces from the specified folder.
    Returns a dictionary mapping face names to pygame surfaces.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    skybox_path = os.path.join(script_dir, skybox_folder)
    
    face_names = {
        'north': 'clouds1_north.bmp',  # Front (-Z)
        'south': 'clouds1_south.bmp',  # Back (+Z)
        'east': 'clouds1_east.bmp',    # Right (+X)
        'west': 'clouds1_west.bmp',    # Left (-X)
        'up': 'clouds1_up.bmp',        # Top (+Y)
        'down': 'clouds1_down.bmp'     # Bottom (-Y)
    }
    
    textures = {}
    
    for face_name, filename in face_names.items():
        filepath = os.path.join(skybox_path, filename)
        try:
            # Load image
            surface = pygame.image.load(filepath)
            # Scale down for performance
            surface = pygame.transform.smoothscale(surface, (512, 512))
            textures[face_name] = surface
            print(f"Loaded skybox face: {face_name}")
        except Exception as e:
            print(f"Warning: Could not load skybox face {face_name} from {filepath}: {e}")
            # Create a fallback colored surface
            fallback = pygame.Surface((512, 512))
            colors = {
                'north': (100, 150, 255),
                'south': (80, 120, 200),
                'east': (120, 160, 255),
                'west': (90, 140, 220),
                'up': (150, 200, 255),
                'down': (70, 100, 150)
            }
            fallback.fill(colors.get(face_name, (100, 100, 100)))
            textures[face_name] = fallback
    
    return textures


def create_off_axis_projection(eye_x, eye_y, eye_z, near, far):
    """
    Create an off-axis projection matrix based on eye position.
    """
    left = -SCREEN_WIDTH_M / 2 - eye_x
    right = SCREEN_WIDTH_M / 2 - eye_x
    bottom = -SCREEN_HEIGHT_M / 2 - eye_y
    top = SCREEN_HEIGHT_M / 2 - eye_y

    scale = near / eye_z
    left *= scale
    right *= scale
    bottom *= scale
    top *= scale

    return {
        'left': left,
        'right': right,
        'bottom': bottom,
        'top': top,
        'near': near,
        'far': far,
        'eye_z': eye_z
    }


def project_3d_point(x, y, z, frustum, eye_x_base, eye_y_base, parallax_sens):
    """
    Project a 3D point using the off-axis frustum with depth-based parallax.
    """
    near = frustum['near']
    eye_z = frustum['eye_z']

    depth_from_screen = abs(z)
    parallax_factor = depth_from_screen / (eye_z + depth_from_screen)
    
    x_adjusted = x - (eye_x_base * parallax_factor * parallax_sens)
    y_adjusted = y - (eye_y_base * parallax_factor * parallax_sens)

    z_dist = eye_z + abs(z)
    if z_dist <= 0.001:
        z_dist = 0.001

    scale = near / z_dist
    x_proj = x_adjusted * scale
    y_proj = y_adjusted * scale

    left = frustum['left']
    right = frustum['right']
    bottom = frustum['bottom']
    top = frustum['top']

    x_norm = (x_proj - left) / (right - left)
    y_norm = (y_proj - bottom) / (top - bottom)

    screen_x = x_norm * WIDTH
    screen_y = (1 - y_norm) * HEIGHT

    return screen_x, screen_y, scale


def get_view_ray_for_pixel(screen_x, screen_y, frustum):
    """
    Calculate the view ray direction for a given screen pixel.
    This ray starts from the eye and goes through the pixel on the near plane.
    """
    # Convert screen coordinates to normalized device coordinates [-1, 1]
    ndc_x = (screen_x / WIDTH) * 2 - 1
    ndc_y = 1 - (screen_y / HEIGHT) * 2  # Flip Y
    
    # Map to frustum space
    left = frustum['left']
    right = frustum['right']
    bottom = frustum['bottom']
    top = frustum['top']
    near = frustum['near']
    
    # Point on near plane in camera space
    near_x = left + (ndc_x + 1) * 0.5 * (right - left)
    near_y = bottom + (ndc_y + 1) * 0.5 * (top - bottom)
    near_z = -near  # Near plane is at -near in camera space
    
    # Ray direction (from origin through near plane point)
    length = np.sqrt(near_x**2 + near_y**2 + near_z**2)
    if length < 0.0001:
        length = 1.0
    
    ray_x = near_x / length
    ray_y = near_y / length
    ray_z = near_z / length
    
    return ray_x, ray_y, ray_z


def sample_cubemap(ray_x, ray_y, ray_z):
    """
    Sample the cubemap based on a view ray direction.
    Returns the skybox face name and UV coordinates (0-1 range).
    """
    abs_x = abs(ray_x)
    abs_y = abs(ray_y)
    abs_z = abs(ray_z)
    
    # Determine which face to use based on dominant axis
    if abs_x >= abs_y and abs_x >= abs_z:
        # X-dominant: east or west
        if ray_x > 0:
            # East face (+X)
            face = 'east'
            u = (-ray_z / abs_x + 1) * 0.5
            v = (-ray_y / abs_x + 1) * 0.5
        else:
            # West face (-X)
            face = 'west'
            u = (ray_z / abs_x + 1) * 0.5
            v = (-ray_y / abs_x + 1) * 0.5
    elif abs_y >= abs_x and abs_y >= abs_z:
        # Y-dominant: up or down
        if ray_y > 0:
            # Up face (+Y)
            face = 'up'
            u = (ray_x / abs_y + 1) * 0.5
            v = (ray_z / abs_y + 1) * 0.5
        else:
            # Down face (-Y)
            face = 'down'
            u = (ray_x / abs_y + 1) * 0.5
            v = (-ray_z / abs_y + 1) * 0.5
    else:
        # Z-dominant: north or south
        if ray_z < 0:
            # North face (-Z, front)
            face = 'north'
            u = (ray_x / abs_z + 1) * 0.5
            v = (-ray_y / abs_z + 1) * 0.5
        else:
            # South face (+Z, back)
            face = 'south'
            u = (-ray_x / abs_z + 1) * 0.5
            v = (-ray_y / abs_z + 1) * 0.5
    
    # Clamp UV to [0, 1]
    u = max(0, min(1, u))
    v = max(0, min(1, v))
    
    return face, u, v


def draw_skybox_background(surface, frustum, eye_x_base, eye_y_base, parallax_sens):
    """
    Draw the skybox as a background using proper view-centered cubemap sampling.
    The skybox surrounds the viewer and rotates with head movement.
    """
    if not skybox_textures:
        return
    
    # Sample the skybox at regular intervals across the screen
    # Using lower resolution for performance, then scale up
    sample_step = 4  # Sample every 4 pixels
    
    # Create a surface to draw skybox on
    skybox_surface = pygame.Surface((WIDTH, HEIGHT))
    
    for screen_y in range(0, HEIGHT, sample_step):
        for screen_x in range(0, WIDTH, sample_step):
            # Get view ray for this pixel
            ray_x, ray_y, ray_z = get_view_ray_for_pixel(screen_x, screen_y, frustum)
            
            # Apply head rotation offset (view direction)
            # This makes skybox respond to head movement
            rotation_x = eye_x_base / (frustum['eye_z'] * 0.1)  # Convert to angle
            rotation_y = eye_y_base / (frustum['eye_z'] * 0.1)
            
            # Rotate ray by head offset
            # Simple rotation around Y axis (horizontal head movement)
            cos_y = np.cos(rotation_x)
            sin_y = np.sin(rotation_x)
            rotated_x = ray_x * cos_y - ray_z * sin_y
            rotated_z = ray_x * sin_y + ray_z * cos_y
            
            # Rotate around X axis (vertical head movement)
            cos_x = np.cos(-rotation_y)
            sin_x = np.sin(-rotation_y)
            rotated_y = ray_y * cos_x - rotated_z * sin_x
            final_z = ray_y * sin_x + rotated_z * cos_x
            
            # Sample cubemap
            face, u, v = sample_cubemap(rotated_x, rotated_y, final_z)
            
            if face in skybox_textures:
                texture = skybox_textures[face]
                tex_width, tex_height = texture.get_size()
                
                # Sample texture at UV coordinates
                tex_x = int(u * (tex_width - 1))
                tex_y = int(v * (tex_height - 1))
                
                try:
                    color = texture.get_at((tex_x, tex_y))
                    # Draw filled rectangle for this sample region
                    pygame.draw.rect(skybox_surface, color,
                                   (screen_x, screen_y, sample_step, sample_step))
                except:
                    pass
    
    # Blit the skybox surface to the main surface
    surface.blit(skybox_surface, (0, 0))


def draw_3d_grid(surface, face_x, face_y, face_z, parallax_sens):
    """
    Draw a 3D grid tunnel with proper off-axis projection and depth-based parallax
    """
    eye_x = (face_x - 0.5) * SCREEN_WIDTH_M * parallax_sens
    eye_y = -(face_y - 0.5) * SCREEN_HEIGHT_M * parallax_sens
    eye_z = face_z

    near = 0.01
    far = eye_z + GRID_DEPTH + 1.0
    frustum = create_off_axis_projection(eye_x, eye_y, eye_z, near, far)

    grid_width = SCREEN_WIDTH_M * 3.0
    grid_height = SCREEN_HEIGHT_M * 3.0

    points = {}
    for k in range(GRID_SIZE + 1):
        z = -0.1 - k * (GRID_DEPTH / GRID_SIZE)
        for i in range(GRID_SIZE + 1):
            for j in range(GRID_SIZE + 1):
                x = -grid_width / 2 + i * (grid_width / GRID_SIZE)
                y = -grid_height / 2 + j * (grid_height / GRID_SIZE)
                screen_x, screen_y, scale = project_3d_point(x, y, z, frustum, eye_x, eye_y, parallax_sens)
                points[(i, j, k)] = (screen_x, screen_y, z, scale)

    lines = []
    for k in range(GRID_SIZE + 1):
        for i in range(GRID_SIZE):
            lines.append(((i, 0, k), (i + 1, 0, k), k))
            lines.append(((i, GRID_SIZE, k), (i + 1, GRID_SIZE, k), k))
        for j in range(GRID_SIZE):
            lines.append(((0, j, k), (0, j + 1, k), k))
            lines.append(((GRID_SIZE, j, k), (GRID_SIZE, j + 1, k), k))

    for i in range(GRID_SIZE + 1):
        for k in range(GRID_SIZE):
            lines.append(((i, 0, k), (i, 0, k + 1), k))
            lines.append(((i, GRID_SIZE, k), (i, GRID_SIZE, k + 1), k))
    for j in range(1, GRID_SIZE):
        for k in range(GRID_SIZE):
            lines.append(((0, j, k), (0, j, k + 1), k))
            lines.append(((GRID_SIZE, j, k), (GRID_SIZE, j, k + 1), k))

    lines_with_depth = []
    for start, end, depth_layer in lines:
        if start in points and end in points:
            avg_z = (points[start][2] + points[end][2]) / 2
            lines_with_depth.append((avg_z, start, end))

    lines_with_depth.sort()

    for avg_z, start, end in lines_with_depth:
        x1, y1, z1, scale1 = points[start]
        x2, y2, z2, scale2 = points[end]

        depth_ratio = abs(avg_z) / GRID_DEPTH
        brightness = int(80 + 120 * depth_ratio)
        brightness = max(50, min(255, brightness))

        color = (
            max(0, min(255, brightness)),
            max(0, min(255, brightness + 20)),
            max(0, min(255, brightness + 50))
        )
        avg_scale = (scale1 + scale2) / 2
        thickness = max(1, int(2 * avg_scale))

        if (-200 < x1 < WIDTH + 200 and -200 < y1 < HEIGHT + 200 and
                -200 < x2 < WIDTH + 200 and -200 < y2 < HEIGHT + 200):
            try:
                pygame.draw.line(surface, color,
                                 (int(x1), int(y1)),
                                 (int(x2), int(y2)),
                                 thickness)
            except:
                pass

    return frustum, eye_x, eye_y, parallax_sens


def draw_ui(surface, face_pos, face_z, parallax_sens):
    """Draw UI overlay"""
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    title = font.render("3D SKYBOX WINDOW - OFF-AXIS PROJECTION", True, (0, 255, 200))
    surface.blit(title, (10, 10))

    face_text = small_font.render(
        f"Face: ({face_pos[0]:.2f}, {face_pos[1]:.2f}) | Depth: {face_z * 100:.1f}cm",
        True, (200, 200, 200)
    )
    surface.blit(face_text, (10, 50))

    instructions = [
        "Move your head - screen acts as a window to the sky!",
        "Skybox at infinite distance with 3D grid in foreground",
        "UP/DOWN: adjust viewing distance",
        f"Parallax: {parallax_sens:.2f} (LEFT/RIGHT to adjust)",
        "F: fullscreen | Q/ESC: quit"
    ]

    y_offset = HEIGHT - 135
    for i, line in enumerate(instructions):
        text = small_font.render(line, True, (200, 200, 200))
        surface.blit(text, (10, y_offset + i * 25))

    pygame.draw.circle(surface, (0, 255, 200), (WIDTH - 100, 100), 50, 2)
    indicator_x = int(WIDTH - 150 + face_pos[0] * 100)
    indicator_y = int(50 + face_pos[1] * 100)
    pygame.draw.circle(surface, (255, 100, 100), (indicator_x, indicator_y), 8)

    depth_bar_x = WIDTH - 50
    depth_bar_y = HEIGHT // 2 - 100
    depth_bar_height = 200
    pygame.draw.rect(surface, (100, 100, 100),
                     (depth_bar_x - 10, depth_bar_y, 20, depth_bar_height), 2)

    depth_norm = (face_z - 0.2) / 0.6
    depth_norm = max(0, min(1, depth_norm))
    depth_y = depth_bar_y + depth_bar_height - int(depth_norm * depth_bar_height)
    pygame.draw.circle(surface, (255, 100, 100), (depth_bar_x, depth_y), 8)

    near_text = small_font.render("Near", True, (150, 150, 150))
    far_text = small_font.render("Far", True, (150, 150, 150))
    surface.blit(near_text, (depth_bar_x - 35, depth_bar_y + depth_bar_height + 5))
    surface.blit(far_text, (depth_bar_x - 25, depth_bar_y - 25))


def main():
    global face_position, face_depth, parallax_sensitivity, screen, WIDTH, HEIGHT, skybox_textures

    print("Loading skybox textures...")
    skybox_textures = load_skybox_textures()
    print(f"Loaded {len(skybox_textures)} skybox faces")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:

        running = True
        face_detected = False
        is_fullscreen = False
        windowed_width, windowed_height = WIDTH, HEIGHT

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    if not is_fullscreen:
                        WIDTH, HEIGHT = event.w, event.h
                        windowed_width, windowed_height = WIDTH, HEIGHT
                        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_ESCAPE:
                        if is_fullscreen:
                            is_fullscreen = False
                            WIDTH, HEIGHT = windowed_width, windowed_height
                            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                        else:
                            running = False
                    elif event.key == pygame.K_UP:
                        face_depth = min(0.8, face_depth + 0.05)
                    elif event.key == pygame.K_DOWN:
                        face_depth = max(0.2, face_depth - 0.05)
                    elif event.key == pygame.K_LEFT:
                        parallax_sensitivity = max(0.0, parallax_sensitivity - 0.1)
                    elif event.key == pygame.K_RIGHT:
                        parallax_sensitivity = min(3.0, parallax_sensitivity + 0.1)
                    elif event.key == pygame.K_f:
                        is_fullscreen = not is_fullscreen
                        if is_fullscreen:
                            windowed_width, windowed_height = WIDTH, HEIGHT
                            display_info = pygame.display.Info()
                            WIDTH, HEIGHT = display_info.current_w, display_info.current_h
                            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
                        else:
                            WIDTH, HEIGHT = windowed_width, windowed_height
                            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_results = face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                face_detected = True
                face_landmarks = face_results.multi_face_landmarks[0]
                nose = face_landmarks.landmark[1]
                face_smoothing.append((nose.x, nose.y))
                avg_x = sum(f[0] for f in face_smoothing) / len(face_smoothing)
                avg_y = sum(f[1] for f in face_smoothing) / len(face_smoothing)
                face_position = [avg_x, avg_y]
            else:
                face_detected = False

            screen.fill((5, 5, 15))

            if face_detected:
                eye_x = (face_position[0] - 0.5) * SCREEN_WIDTH_M * parallax_sensitivity
                eye_y = -(face_position[1] - 0.5) * SCREEN_HEIGHT_M * parallax_sensitivity
                eye_z = face_depth
                
                near = 0.01
                far = eye_z + SKYBOX_DISTANCE + 10.0
                frustum = create_off_axis_projection(eye_x, eye_y, eye_z, near, far)
                
                # Render: skybox -> grid -> UI
                draw_skybox_background(screen, frustum, eye_x, eye_y, parallax_sensitivity)
                draw_3d_grid(screen, face_position[0], face_position[1], face_depth, parallax_sensitivity)
                draw_ui(screen, face_position, face_depth, parallax_sensitivity)

            fps = int(clock.get_fps())
            fps_font = pygame.font.Font(None, 24)
            fps_text = fps_font.render(f"FPS: {fps}", True, (100, 255, 100))
            screen.blit(fps_text, (WIDTH - 100, 10))

            pygame.display.flip()
            clock.tick(144)

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
