import cv2
import numpy as np
import pygame
import mediapipe as mp
from collections import deque

# Initialize Pygame
pygame.init()

# Get display info and set window to nearly full screen size
display_info = pygame.display.Info()
WIDTH, HEIGHT = display_info.current_w - 100, display_info.current_h - 100

# Create resizable windowed mode
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("3D Grid - Proper Off-Axis Projection")
clock = pygame.time.Clock()

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# 3D Grid settings
GRID_SIZE = 10  # 10x10 grid
GRID_DEPTH = 1.5  # Depth in meters (scene depth) - increased for deeper tunnel

# Screen physical dimensions (approximate for a typical laptop/monitor)
# These should be adjusted based on your actual screen size
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


def create_off_axis_projection(eye_x, eye_y, eye_z, near, far):
    """
    Create an off-axis projection matrix based on eye position.
    This is the key technique from the article.

    eye_x, eye_y, eye_z: Eye position in meters relative to screen center
    near: Near clipping plane distance
    far: Far clipping plane distance
    """
    # Calculate frustum bounds based on eye position and screen edges
    # The eye looks perpendicular to the screen plane

    left = -SCREEN_WIDTH_M / 2 - eye_x
    right = SCREEN_WIDTH_M / 2 - eye_x
    bottom = -SCREEN_HEIGHT_M / 2 - eye_y
    top = SCREEN_HEIGHT_M / 2 - eye_y

    # Scale by near plane distance
    # This creates the asymmetric frustum
    scale = near / eye_z
    left *= scale
    right *= scale
    bottom *= scale
    top *= scale

    # Build projection matrix (OpenGL style, needs conversion for our use)
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

    x, y, z: 3D point coordinates in world space (meters)
    frustum: Dictionary containing frustum parameters
    eye_x_base, eye_y_base: Base eye offset for parallax scaling
    parallax_sens: Parallax sensitivity multiplier
    """
    near = frustum['near']
    far = frustum['far']
    eye_z = frustum['eye_z']

    # For proper head-tracking parallax, objects at screen plane (z=0) should not move
    # Objects behind screen should shift based on viewing angle
    # The parallax effect increases with distance from screen
    
    # Calculate depth from screen plane
    depth_from_screen = abs(z)
    
    # Parallax offset: farther objects appear to shift more when viewing angle changes
    # At z=0 (screen plane), no parallax. As z gets more negative, more parallax.
    parallax_factor = depth_from_screen / (eye_z + depth_from_screen)
    
    # Apply parallax offset scaled by sensitivity
    x_adjusted = x - (eye_x_base * parallax_factor * parallax_sens)
    y_adjusted = y - (eye_y_base * parallax_factor * parallax_sens)

    # Z distance from eye
    z_dist = eye_z + abs(z)

    if z_dist <= 0.001:
        z_dist = 0.001

    # Perspective divide
    scale = near / z_dist

    # Project to near plane
    x_proj = x_adjusted * scale
    y_proj = y_adjusted * scale

    # Map from frustum space to screen space
    left = frustum['left']
    right = frustum['right']
    bottom = frustum['bottom']
    top = frustum['top']

    # Normalize to [0, 1]
    x_norm = (x_proj - left) / (right - left)
    y_norm = (y_proj - bottom) / (top - bottom)

    # Convert to screen coordinates
    screen_x = x_norm * WIDTH
    screen_y = (1 - y_norm) * HEIGHT  # Flip Y for screen coords

    return screen_x, screen_y, scale


def draw_3d_grid(surface, face_x, face_y, face_z, parallax_sens):
    """
    Draw a 3D grid tunnel with proper off-axis projection and depth-based parallax
    """
    # Convert normalized face position to eye position in meters
    # face_x, face_y are in [0, 1], center screen is 0.5, 0.5
    # When face is at edge of screen (0 or 1), eye should be at physical edge
    eye_x = (face_x - 0.5) * SCREEN_WIDTH_M * parallax_sens
    # Invert Y so moving head up moves view up (natural tracking)
    eye_y = -(face_y - 0.5) * SCREEN_HEIGHT_M * parallax_sens
    eye_z = face_z  # Distance from screen

    # Create off-axis projection frustum
    near = 0.01  # 1cm near plane
    far = eye_z + GRID_DEPTH + 1.0  # Extend beyond grid

    frustum = create_off_axis_projection(eye_x, eye_y, eye_z, near, far)

    # Grid dimensions in world space (meters)
    # Grid should be large enough to fill view and allow "looking around corners"
    grid_width = SCREEN_WIDTH_M * 3.0  # Extended even more to draw beyond screen edges
    grid_height = SCREEN_HEIGHT_M * 3.0

    # Generate all grid points
    points = {}

    for k in range(GRID_SIZE + 1):  # Depth layers
        # Grid extends from near screen to deep behind screen (all negative Z)
        z = -0.1 - k * (GRID_DEPTH / GRID_SIZE)  # Start at -0.1m, go deeper

        for i in range(GRID_SIZE + 1):  # Horizontal
            for j in range(GRID_SIZE + 1):  # Vertical
                # 3D position in world space (meters, centered on screen)
                x = -grid_width / 2 + i * (grid_width / GRID_SIZE)
                y = -grid_height / 2 + j * (grid_height / GRID_SIZE)

                screen_x, screen_y, scale = project_3d_point(x, y, z, frustum, eye_x, eye_y, parallax_sens)
                points[(i, j, k)] = (screen_x, screen_y, z, scale)

    lines = []

    # ===== 1. HORIZONTAL LINES AT EACH DEPTH (top and bottom edges) =====
    for k in range(GRID_SIZE + 1):
        # Top edge
        for i in range(GRID_SIZE):
            lines.append(((i, 0, k), (i + 1, 0, k), k))
        # Bottom edge
        for i in range(GRID_SIZE):
            lines.append(((i, GRID_SIZE, k), (i + 1, GRID_SIZE, k), k))

    # ===== 2. VERTICAL LINES AT EACH DEPTH (left and right edges) =====
    for k in range(GRID_SIZE + 1):
        # Left edge
        for j in range(GRID_SIZE):
            lines.append(((0, j, k), (0, j + 1, k), k))
        # Right edge
        for j in range(GRID_SIZE):
            lines.append(((GRID_SIZE, j, k), (GRID_SIZE, j + 1, k), k))

    # ===== 3. DEPTH LINES - Connect from front to back =====
    # Top edge into depth
    for i in range(GRID_SIZE + 1):
        for k in range(GRID_SIZE):
            lines.append(((i, 0, k), (i, 0, k + 1), k))

    # Bottom edge into depth
    for i in range(GRID_SIZE + 1):
        for k in range(GRID_SIZE):
            lines.append(((i, GRID_SIZE, k), (i, GRID_SIZE, k + 1), k))

    # Left edge into depth (skip corners)
    for j in range(1, GRID_SIZE):
        for k in range(GRID_SIZE):
            lines.append(((0, j, k), (0, j, k + 1), k))

    # Right edge into depth (skip corners)
    for j in range(1, GRID_SIZE):
        for k in range(GRID_SIZE):
            lines.append(((GRID_SIZE, j, k), (GRID_SIZE, j, k + 1), k))

    # Sort by depth (draw far to near)
    lines_with_depth = []
    for start, end, depth_layer in lines:
        if start in points and end in points:
            avg_z = (points[start][2] + points[end][2]) / 2
            lines_with_depth.append((avg_z, start, end))

    lines_with_depth.sort()  # Sort ascending (most negative = farthest)

    # Draw all lines
    for avg_z, start, end in lines_with_depth:
        x1, y1, z1, scale1 = points[start]
        x2, y2, z2, scale2 = points[end]

        # Color based on depth
        depth_ratio = abs(avg_z) / GRID_DEPTH
        # Invert brightness so far lines are pale/light
        brightness = int(80 + 120 * depth_ratio)  # Far = lighter (200), Near = darker (80)
        brightness = max(50, min(255, brightness))

        # Front face gets brighter, special color
        if k == 0:  # Front layer of grid
            color = (
                max(0, min(255, brightness)),
                max(0, min(255, brightness + 50)),
                max(0, min(255, brightness + 100))
            )
            thickness = 3
        else:
            color = (
                max(0, min(255, brightness)),
                max(0, min(255, brightness + 20)),
                max(0, min(255, brightness + 50))
            )
            avg_scale = (scale1 + scale2) / 2
            thickness = max(1, int(2 * avg_scale))

        # Only draw if visible on screen (with margin)
        if (-200 < x1 < WIDTH + 200 and -200 < y1 < HEIGHT + 200 and
                -200 < x2 < WIDTH + 200 and -200 < y2 < HEIGHT + 200):
            try:
                pygame.draw.line(surface, color,
                                 (int(x1), int(y1)),
                                 (int(x2), int(y2)),
                                 thickness)
            except:
                pass  # Skip invalid coordinates

    # Return frustum and eye positions for cube drawing
    return frustum, eye_x, eye_y, parallax_sens


def draw_cube_3d(surface, cube, frustum, eye_x_base, eye_y_base, parallax_sens):
    """
    Draw a 3D cube with proper off-axis projection and depth-based parallax
    cube: dict with 'x', 'y', 'z', 'size', 'color'
    """
    cx, cy, cz = cube['x'], cube['y'], cube['z']
    size = cube['size']
    half_size = size / 2
    
    # Define 8 vertices of the cube
    vertices = [
        (cx - half_size, cy - half_size, cz - half_size),  # 0: back-bottom-left
        (cx + half_size, cy - half_size, cz - half_size),  # 1: back-bottom-right
        (cx + half_size, cy + half_size, cz - half_size),  # 2: back-top-right
        (cx - half_size, cy + half_size, cz - half_size),  # 3: back-top-left
        (cx - half_size, cy - half_size, cz + half_size),  # 4: front-bottom-left
        (cx + half_size, cy - half_size, cz + half_size),  # 5: front-bottom-right
        (cx + half_size, cy + half_size, cz + half_size),  # 6: front-top-right
        (cx - half_size, cy + half_size, cz + half_size),  # 7: front-top-left
    ]
    
    # Project all vertices to screen space with depth-based parallax
    projected = []
    for vx, vy, vz in vertices:
        screen_x, screen_y, scale = project_3d_point(vx, vy, vz, frustum, eye_x_base, eye_y_base, parallax_sens)
        projected.append((screen_x, screen_y, vz, scale))
    
    # Define edges of the cube (vertex index pairs)
    edges = [
        # Back face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Front face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Connecting edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    # Draw edges
    base_color = cube['color']
    for v1_idx, v2_idx in edges:
        x1, y1, z1, scale1 = projected[v1_idx]
        x2, y2, z2, scale2 = projected[v2_idx]
        
        # Check if on screen
        if (-200 < x1 < WIDTH + 200 and -200 < y1 < HEIGHT + 200 and
                -200 < x2 < WIDTH + 200 and -200 < y2 < HEIGHT + 200):
            
            # Adjust brightness based on depth
            avg_z = (z1 + z2) / 2
            depth_factor = max(0.3, 1 - (abs(avg_z) / GRID_DEPTH))
            
            color = (
                int(base_color[0] * depth_factor),
                int(base_color[1] * depth_factor),
                int(base_color[2] * depth_factor)
            )
            
            avg_scale = (scale1 + scale2) / 2
            thickness = max(2, int(3 * avg_scale))
            
            try:
                pygame.draw.line(surface, color,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                thickness)
            except:
                pass
    
    # Draw faces with transparency (optional, for solid look)
    # Define faces as vertex indices (in counter-clockwise order)
    faces = [
        ([4, 5, 6, 7], 0.3),  # Front face (brightest)
        ([1, 0, 3, 2], 0.15),  # Back face (dimmest)
        ([5, 1, 2, 6], 0.25),  # Right face
        ([0, 4, 7, 3], 0.25),  # Left face
        ([7, 6, 2, 3], 0.2),   # Top face
        ([0, 1, 5, 4], 0.2),   # Bottom face
    ]
    
    for face_verts, alpha_factor in faces:
        points = [projected[i][:2] for i in face_verts]
        
        # Check if face is visible (all points on screen roughly)
        if all(-500 < x < WIDTH + 500 and -500 < y < HEIGHT + 500 for x, y in points):
            # Calculate average depth for this face
            avg_z = sum(projected[i][2] for i in face_verts) / len(face_verts)
            depth_factor = max(0.3, 1 - (abs(avg_z) / GRID_DEPTH))
            
            color = (
                int(base_color[0] * depth_factor * alpha_factor),
                int(base_color[1] * depth_factor * alpha_factor),
                int(base_color[2] * depth_factor * alpha_factor)
            )
            
            try:
                pygame.draw.polygon(surface, color, [(int(x), int(y)) for x, y in points])
            except:
                pass


def draw_ui(surface, face_pos, face_z, parallax_sens):
    """Draw UI overlay"""
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    title = font.render("3D OFF-AXIS PROJECTION GRID", True, (0, 255, 200))
    surface.blit(title, (10, 10))

    face_text = small_font.render(
        f"Face: ({face_pos[0]:.2f}, {face_pos[1]:.2f}) | Depth: {face_z * 100:.1f}cm",
        True, (200, 200, 200)
    )
    surface.blit(face_text, (10, 50))

    instructions = [
        "Move your head to change perspective!",
        "Uses proper off-axis projection technique",
        "Press UP/DOWN to adjust viewing distance",
        f"Parallax Sensitivity: {parallax_sens:.2f} (LEFT/RIGHT to adjust)",
        "Press Q or ESC to quit"
    ]

    y_offset = HEIGHT - 110
    for i, line in enumerate(instructions):
        text = small_font.render(line, True, (200, 200, 200))
        surface.blit(text, (10, y_offset + i * 25))

    # Face position indicator
    pygame.draw.circle(surface, (0, 255, 200), (WIDTH - 100, 100), 50, 2)
    indicator_x = int(WIDTH - 150 + face_pos[0] * 100)
    indicator_y = int(50 + face_pos[1] * 100)
    pygame.draw.circle(surface, (255, 100, 100), (indicator_x, indicator_y), 8)

    # Depth indicator
    depth_bar_x = WIDTH - 50
    depth_bar_y = HEIGHT // 2 - 100
    depth_bar_height = 200

    pygame.draw.rect(surface, (100, 100, 100),
                     (depth_bar_x - 10, depth_bar_y, 20, depth_bar_height), 2)

    # Current depth position (0.2m to 0.8m range)
    depth_norm = (face_z - 0.2) / 0.6
    depth_norm = max(0, min(1, depth_norm))
    depth_y = depth_bar_y + depth_bar_height - int(depth_norm * depth_bar_height)

    pygame.draw.circle(surface, (255, 100, 100), (depth_bar_x, depth_y), 8)

    # Depth labels
    near_text = small_font.render("Near", True, (150, 150, 150))
    far_text = small_font.render("Far", True, (150, 150, 150))
    surface.blit(near_text, (depth_bar_x - 35, depth_bar_y + depth_bar_height + 5))
    surface.blit(far_text, (depth_bar_x - 25, depth_bar_y - 25))


def main():
    global face_position, face_depth, parallax_sensitivity, screen, WIDTH, HEIGHT

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
        face_detected = False  # Track if face is currently detected
        is_fullscreen = False  # Track fullscreen state
        windowed_width, windowed_height = WIDTH, HEIGHT  # Store windowed size

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize (only in windowed mode)
                    if not is_fullscreen:
                        WIDTH, HEIGHT = event.w, event.h
                        windowed_width, windowed_height = WIDTH, HEIGHT
                        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_ESCAPE:
                        # ESC exits fullscreen first, then exits program
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
                        # Toggle fullscreen
                        is_fullscreen = not is_fullscreen
                        if is_fullscreen:
                            # Store current windowed size before going fullscreen
                            windowed_width, windowed_height = WIDTH, HEIGHT
                            # Get actual screen resolution for fullscreen
                            display_info = pygame.display.Info()
                            WIDTH, HEIGHT = display_info.current_w, display_info.current_h
                            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
                        else:
                            # Return to windowed mode with stored size
                            WIDTH, HEIGHT = windowed_width, windowed_height
                            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process face
            face_results = face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                face_detected = True
                face_landmarks = face_results.multi_face_landmarks[0]

                # Use nose tip for tracking
                nose = face_landmarks.landmark[1]
                face_smoothing.append((nose.x, nose.y))

                avg_x = sum(f[0] for f in face_smoothing) / len(face_smoothing)
                avg_y = sum(f[1] for f in face_smoothing) / len(face_smoothing)
                face_position = [avg_x, avg_y]

                # Manual control with UP/DOWN keys
            else:
                face_detected = False

            # Clear screen to dark color
            screen.fill((5, 5, 15))

            # Only draw if face is detected
            if face_detected:
                # Calculate eye position for projection
                eye_x = (face_position[0] - 0.5) * SCREEN_WIDTH_M * parallax_sensitivity
                eye_y = -(face_position[1] - 0.5) * SCREEN_HEIGHT_M * parallax_sensitivity
                eye_z = face_depth
                
                # Create off-axis projection frustum
                near = 0.01
                far = eye_z + 10.0
                frustum = create_off_axis_projection(eye_x, eye_y, eye_z, near, far)
                
                # RENDERING ORDER (back to front):
                # 1. Draw skybox background (farthest)
                #draw_skybox_background(screen, frustum, eye_x, eye_y, parallax_sensitivity)
                
                # 2. Draw 3D grid (intermediate depth)
                frustum, eye_x, eye_y, parallax_sens = draw_3d_grid(screen, face_position[0], face_position[1], face_depth, parallax_sensitivity)
                
                # 3. Draw UI overlay (nearest, no projection)
                draw_ui(screen, face_position, face_depth, parallax_sensitivity)

            # Display FPS
            fps = int(clock.get_fps())
            fps_font = pygame.font.Font(None, 24)
            fps_text = fps_font.render(f"FPS: {fps}", True, (100, 255, 100))
            screen.blit(fps_text, (WIDTH - 100, 10))

            pygame.display.flip()
            clock.tick(144)  # 144 FPS for 144Hz displays

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()