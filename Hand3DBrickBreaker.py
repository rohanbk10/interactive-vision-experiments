import cv2
import numpy as np
import pygame
import mediapipe as mp
from collections import deque
import random

# Optimize OpenCV for better camera performance
cv2.setNumThreads(4)  # Use multiple CPU threads

# Initialize Pygame with hardware acceleration hints
import os
os.environ['SDL_VIDEO_CENTERED'] = '1'
pygame.init()

# Get display info
display_info = pygame.display.Info()
WIDTH, HEIGHT = display_info.current_w - 100, display_info.current_h - 100

# Create resizable windowed mode with hardware surface hint
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE | pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption("3D Hand Brick Breaker")
clock = pygame.time.Clock()

# Cache fonts for better performance
FONT_CACHE = {
    'large': pygame.font.Font(None, 72),
    'medium': pygame.font.Font(None, 48),
    'small': pygame.font.Font(None, 32),
    'tiny': pygame.font.Font(None, 24),
    'countdown': pygame.font.Font(None, 120),
}

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# MediaPipe Hands
mp_hands = mp.solutions.hands

# 3D Grid settings
GRID_SIZE = 6  # Reduced from 10 for better performance
GRID_DEPTH = 1.5

# Screen physical dimensions
SCREEN_WIDTH_M = 0.285
SCREEN_HEIGHT_M = 0.160

# Face tracking
face_position = [0.5, 0.5]
face_depth = 0.2  # Default to near position (close view)
face_smoothing = deque(maxlen=5)  # Reduced from 10 for better performance
parallax_sensitivity = 1.0
use_face_tracking = False  # Disable by default for better performance

# Hand tracking (for both hands)
hand_smoothing = [deque(maxlen=3), deque(maxlen=3)]  # Reduced from 5 for better performance

# Brick settings
BRICK_ROWS = 4  # Reduced from 5 for better performance
BRICK_COLS = 6  # Reduced from 8 for better performance
BRICK_DEPTH = -1.4  # At the back of the tunnel
BRICK_SIZE = 0.09  # Size in meters (slightly larger since fewer bricks)

# Game state
LIVES = 3
lives = LIVES
game_started = False
game_over = False
game_won = False
countdown_active = False
countdown_time = 3.0
countdown_timer = 0.0

# Ball
ball_position_3d = [0.0, 0.0, -0.3]  # 3D position in world space (meters)
ball_velocity_3d = [0.0, 0.0, 0.0]  # 3D velocity
ball_radius = 0.06  # Radius in meters (4x increased from 0.015)
ball_speed = 0.8  # meters per second

# Hand collision squares
hand_squares = []  # List of hand collision squares

# Bricks (3D positions)
bricks = []


def create_off_axis_projection(eye_x, eye_y, eye_z, near, far):
    """Create an off-axis projection matrix based on eye position"""
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
    """Project a 3D point to 2D screen space"""
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


def draw_3d_grid(surface, face_x, face_y, face_z, parallax_sens):
    """Draw a 3D grid tunnel"""
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

    # Horizontal lines
    for k in range(GRID_SIZE + 1):
        for i in range(GRID_SIZE):
            lines.append(((i, 0, k), (i + 1, 0, k), k))
        for i in range(GRID_SIZE):
            lines.append(((i, GRID_SIZE, k), (i + 1, GRID_SIZE, k), k))

    # Vertical lines
    for k in range(GRID_SIZE + 1):
        for j in range(GRID_SIZE):
            lines.append(((0, j, k), (0, j + 1, k), k))
        for j in range(GRID_SIZE):
            lines.append(((GRID_SIZE, j, k), (GRID_SIZE, j + 1, k), k))

    # Depth lines
    for i in range(GRID_SIZE + 1):
        for k in range(GRID_SIZE):
            lines.append(((i, 0, k), (i, 0, k + 1), k))
    for i in range(GRID_SIZE + 1):
        for k in range(GRID_SIZE):
            lines.append(((i, GRID_SIZE, k), (i, GRID_SIZE, k + 1), k))
    for j in range(1, GRID_SIZE):
        for k in range(GRID_SIZE):
            lines.append(((0, j, k), (0, j, k + 1), k))
    for j in range(1, GRID_SIZE):
        for k in range(GRID_SIZE):
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

        if start[2] == 0:
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


def draw_back_wall(surface, frustum, eye_x, eye_y, parallax_sens):
    """Draw a grey opaque wall at the back of the tunnel"""
    wall_z = BRICK_DEPTH - 0.15  # Slightly behind the bricks
    wall_width = SCREEN_WIDTH_M * 2.5
    wall_height = SCREEN_HEIGHT_M * 2.5
    
    # Define wall corners
    corners_3d = [
        (-wall_width/2, -wall_height/2, wall_z),  # bottom-left
        (wall_width/2, -wall_height/2, wall_z),   # bottom-right
        (wall_width/2, wall_height/2, wall_z),    # top-right
        (-wall_width/2, wall_height/2, wall_z),   # top-left
    ]
    
    # Project to screen
    projected = []
    for vx, vy, vz in corners_3d:
        sx, sy, scale = project_3d_point(vx, vy, vz, frustum, eye_x, eye_y, parallax_sens)
        projected.append((int(sx), int(sy)))
    
    # Draw opaque grey wall
    try:
        if all(-1000 < x < WIDTH + 1000 and -1000 < y < HEIGHT + 1000 for x, y in projected):
            pygame.draw.polygon(surface, (60, 60, 80), projected)
            pygame.draw.polygon(surface, (100, 100, 120), projected, 3)
    except:
        pass


def setup_bricks():
    """Create bricks at the back of the tunnel"""
    global bricks
    bricks.clear()
    
    colors = [
        (255, 100, 100),
        (255, 150, 100),
        (255, 255, 100),
        (100, 255, 100),
        (100, 150, 255),
    ]
    
    # Calculate brick grid dimensions
    total_width = BRICK_COLS * BRICK_SIZE
    total_height = BRICK_ROWS * BRICK_SIZE
    
    for row in range(BRICK_ROWS):
        for col in range(BRICK_COLS):
            x = -total_width / 2 + col * BRICK_SIZE + BRICK_SIZE / 2
            y = -total_height / 2 + row * BRICK_SIZE + BRICK_SIZE / 2
            z = BRICK_DEPTH
            
            color = colors[row % len(colors)]
            bricks.append({
                'x': x,
                'y': y,
                'z': z,
                'size': BRICK_SIZE,
                'color': color,
                'active': True
            })


def draw_brick(surface, brick, frustum, eye_x, eye_y, parallax_sens):
    """Draw a single brick in 3D - returns points for batch rendering"""
    if not brick['active']:
        return None
    
    cx, cy, cz = brick['x'], brick['y'], brick['z']
    size = brick['size']
    half = size / 2
    
    # Only project front face vertices (optimization)
    front_vertices = [
        (cx - half, cy - half, cz + half),
        (cx + half, cy - half, cz + half),
        (cx + half, cy + half, cz + half),
        (cx - half, cy + half, cz + half),
    ]
    
    # Project vertices
    projected = []
    for vx, vy, vz in front_vertices:
        sx, sy, scale = project_3d_point(vx, vy, vz, frustum, eye_x, eye_y, parallax_sens)
        projected.append((int(sx), int(sy)))
    
    # Check if visible
    if all(-500 < x < WIDTH + 500 and -500 < y < HEIGHT + 500 for x, y in projected):
        return (projected, brick['color'])
    
    return None


def draw_ball_3d(surface, frustum, eye_x, eye_y, parallax_sens):
    """Draw the ball in 3D space - red with black outline, size scales with depth"""
    bx, by, bz = ball_position_3d
    
    sx, sy, scale = project_3d_point(bx, by, bz, frustum, eye_x, eye_y, parallax_sens)
    
    # Calculate size scale based on depth position
    # Base size = 1.0x
    # Near screen (z close to 0): increase by 0.5 → 1.5x
    # Near bricks (z close to -1.4): reduce by 0.5 → 0.5x
    
    # Normalize depth: 0 at screen, 1 at bricks
    depth_normalized = abs(bz) / 1.4  # -1.4 is brick depth
    depth_normalized = max(0.0, min(1.0, depth_normalized))
    
    # Size scaling: 1.5x at screen, 1.0x at middle, 0.5x at bricks
    # Linear interpolation: 1.5 - (depth_normalized * 1.0)
    size_scale = 1.5 - (depth_normalized * 1.0)
    
    radius_screen = int(ball_radius * scale * 4000 * size_scale)
    radius_screen = max(5, min(60, radius_screen))  # Larger range for bigger ball
    
    if -100 < sx < WIDTH + 100 and -100 < sy < HEIGHT + 100:
        # Red ball
        pygame.draw.circle(surface, (255, 50, 50), (int(sx), int(sy)), radius_screen)
        # Black outline (scales with size)
        outline_thickness = max(2, radius_screen // 6)
        pygame.draw.circle(surface, (0, 0, 0), (int(sx), int(sy)), radius_screen, outline_thickness)


def draw_hand_square(surface, hand_pos, hand_size, frustum, eye_x, eye_y, parallax_sens):
    """Draw transparent collision square at hand position - simplified"""
    # Hand is at screen plane (z = 0)
    hx, hy = hand_pos
    size = hand_size
    
    # Convert screen position to world position at z=0
    world_x = ((hx / WIDTH) - 0.5) * SCREEN_WIDTH_M * 2.5
    world_y = (0.5 - (hy / HEIGHT)) * SCREEN_HEIGHT_M * 2.5
    world_z = 0.0  # At screen plane
    
    # Simplified rendering - just draw at hand screen position
    half_pixels = int((size * WIDTH / SCREEN_WIDTH_M) / 2.5)
    rect = pygame.Rect(int(hx) - half_pixels, int(hy) - half_pixels, half_pixels * 2, half_pixels * 2)
    
    # Draw transparent square (simpler, faster)
    try:
        s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        s.fill((0, 255, 100, 80))
        surface.blit(s, rect.topleft)
        pygame.draw.rect(surface, (0, 255, 100), rect, 3)
    except:
        pass
    
    return world_x, world_y, world_z, size


def check_brick_collision():
    """Check if ball hits any brick"""
    global ball_velocity_3d
    
    for brick in bricks:
        if not brick['active']:
            continue
        
        bx, by, bz = ball_position_3d
        cx, cy, cz = brick['x'], brick['y'], brick['z']
        size = brick['size']
        half = size / 2
        
        # Simple box collision
        if (cx - half - ball_radius < bx < cx + half + ball_radius and
            cy - half - ball_radius < by < cy + half + ball_radius and
            cz - half - ball_radius < bz < cz + half + ball_radius):
            
            brick['active'] = False
            
            # Bounce ball based on which face was hit
            dx = abs(bx - cx)
            dy = abs(by - cy)
            dz = abs(bz - cz)
            
            if dz > dx and dz > dy:
                # Hit front/back face
                ball_velocity_3d[2] = -ball_velocity_3d[2]
            elif dx > dy:
                # Hit left/right face
                ball_velocity_3d[0] = -ball_velocity_3d[0]
            else:
                # Hit top/bottom face
                ball_velocity_3d[1] = -ball_velocity_3d[1]
            
            return True
    
    return False


def check_hand_collision(hand_squares_data):
    """Check if ball hits any hand square"""
    global ball_velocity_3d, ball_speed
    
    bx, by, bz = ball_position_3d
    
    for world_x, world_y, world_z, size in hand_squares_data:
        half = size / 2
        
        # Check collision with hand square (at z=0 plane)
        if (world_x - half < bx < world_x + half and
            world_y - half < by < world_y + half and
            -ball_radius < bz < ball_radius):
            
            # Bounce ball back
            if ball_velocity_3d[2] < 0:  # Only if moving toward screen
                ball_velocity_3d[2] = abs(ball_velocity_3d[2])
                
                # Add angle based on hit position
                offset_x = (bx - world_x) / half
                offset_y = (by - world_y) / half
                
                ball_velocity_3d[0] += offset_x * 0.2
                ball_velocity_3d[1] += offset_y * 0.2
                
                # Increase speed by 1%
                ball_speed *= 1.01
                
                return True
    
    return False


def reset_game():
    """Reset the game"""
    global lives, game_started, game_over, game_won, ball_position_3d, ball_velocity_3d, ball_speed
    global countdown_active, countdown_timer
    
    lives = LIVES
    game_started = False
    game_over = False
    game_won = False
    ball_speed = 0.8
    countdown_active = False
    countdown_timer = 0.0
    
    # Reset ball position
    ball_position_3d = [0.0, 0.0, -0.3]
    ball_velocity_3d = [0.0, 0.0, 0.0]
    
    setup_bricks()


def draw_ui(surface, fps):
    """Draw UI elements"""
    global use_face_tracking
    
    # Use cached fonts
    font = FONT_CACHE['medium']
    small_font = FONT_CACHE['small']
    tiny_font = FONT_CACHE['tiny']
    
    # Lives
    lives_text = font.render(f"Lives: {lives}", True, (255, 255, 255))
    surface.blit(lives_text, (10, 10))
    
    # FPS
    fps_color = (100, 255, 100) if fps > 50 else (255, 255, 100) if fps > 30 else (255, 100, 100)
    fps_text = tiny_font.render(f"FPS: {int(fps)}", True, fps_color)
    surface.blit(fps_text, (10, 60))
    
    # Face tracking status
    face_status = "ON" if use_face_tracking else "OFF"
    face_color = (100, 255, 100) if use_face_tracking else (255, 100, 100)
    face_text = tiny_font.render(f"Face Parallax: {face_status} (F)", True, face_color)
    surface.blit(face_text, (10, 85))
    
    # Bricks
    active_bricks = sum(1 for b in bricks if b['active'])
    bricks_text = small_font.render(f"Bricks: {active_bricks}", True, (200, 200, 200))
    surface.blit(bricks_text, (WIDTH - 200, 10))
    
    # Speed
    speed_text = small_font.render(f"Speed: {ball_speed:.2f}", True, (200, 200, 200))
    surface.blit(speed_text, (WIDTH - 200, 40))
    
    # Countdown
    if countdown_active and not game_started:
        countdown_font = FONT_CACHE['countdown']
        # Show remaining time (counting down from 3 to 1)
        countdown_num = int(countdown_time - countdown_timer)
        if countdown_num < 1:
            countdown_num = 1
        countdown_text = countdown_font.render(str(countdown_num), True, (255, 255, 100))
        rect = countdown_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        surface.blit(countdown_text, rect)
        
        # Progress bar
        progress = countdown_timer / countdown_time
        bar_width = 400
        bar_height = 20
        bar_x = WIDTH // 2 - bar_width // 2
        bar_y = HEIGHT // 2 + 80
        
        # Background bar
        pygame.draw.rect(surface, (60, 60, 60), (bar_x, bar_y, bar_width, bar_height))
        # Progress bar
        pygame.draw.rect(surface, (255, 255, 100), (bar_x, bar_y, int(bar_width * progress), bar_height))
        # Border
        pygame.draw.rect(surface, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height), 2)
        
        # Show "Keep hand visible" message
        tiny_font = FONT_CACHE['tiny']
        msg = tiny_font.render("Keep hand visible...", True, (200, 200, 200))
        msg_rect = msg.get_rect(center=(WIDTH // 2, bar_y + 40))
        surface.blit(msg, msg_rect)
    
    # Instructions
    if not game_started and not game_over and not game_won and not countdown_active:
        inst_font = small_font
        inst = inst_font.render("Hold hand steady for 3 seconds to start!", True, (255, 255, 100))
        rect = inst.get_rect(center=(WIDTH // 2, HEIGHT - 100))
        surface.blit(inst, rect)
        
        inst2 = tiny_font.render("Press F to toggle face parallax for better FPS", True, (150, 150, 150))
        rect2 = inst2.get_rect(center=(WIDTH // 2, HEIGHT - 60))
        surface.blit(inst2, rect2)
    
    # Game Over
    if game_over:
        over_font = FONT_CACHE['large']
        over_text = over_font.render("GAME OVER!", True, (255, 50, 50))
        rect = over_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        surface.blit(over_text, rect)
        
        restart = small_font.render("Press R to Restart | Q to Quit", True, (200, 200, 200))
        rect2 = restart.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 60))
        surface.blit(restart, rect2)
    
    # Win
    if game_won:
        win_font = FONT_CACHE['large']
        win_text = win_font.render("YOU WIN!", True, (100, 255, 100))
        rect = win_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        surface.blit(win_text, rect)
        
        restart = small_font.render("Press R to Restart | Q to Quit", True, (200, 200, 200))
        rect2 = restart.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 60))
        surface.blit(restart, rect2)


def main():
    global face_position, face_depth, parallax_sensitivity, screen, WIDTH, HEIGHT
    global lives, game_started, game_over, game_won, ball_position_3d, ball_velocity_3d
    global countdown_active, countdown_timer, use_face_tracking, ball_speed
    
    cap = cv2.VideoCapture(0)
    # Use DirectShow backend on Windows for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for lower latency
    
    setup_bricks()
    
    # Try to use GPU delegate for MediaPipe if available
    # Note: GPU support requires mediapipe-gpu build on some systems
    try:
        # Attempt GPU configuration
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,  # Disable refinement for speed
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,  # Fastest model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("MediaPipe initialized (using available acceleration)")
        
    except Exception as e:
        print(f"MediaPipe initialization warning: {e}")
        # Fallback to standard initialization
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    try:
        
        running = True
        face_detected = False
        frame_count = 0
        hand_detect_interval = 2
        face_detect_interval = 5  # Process face less frequently (higher = better FPS)
        last_hand_results = None
        last_face_results = None
        target_fps = 30  # Lower target FPS for better performance
        dt = 1/target_fps
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        reset_game()
                    elif event.key == pygame.K_f:
                        use_face_tracking = not use_face_tracking
                        print(f"Face tracking: {'ON' if use_face_tracking else 'OFF'} (Better FPS when OFF)")
                    elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                        hand_detect_interval = min(5, hand_detect_interval + 1)
                        print(f"Hand detection interval: {hand_detect_interval} (Higher = Better FPS, Lower = Smoother)")
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_KP_PLUS:
                        hand_detect_interval = max(1, hand_detect_interval - 1)
                        print(f"Hand detection interval: {hand_detect_interval} (Higher = Better FPS, Lower = Smoother)")
                    elif event.key == pygame.K_UP:
                        face_depth = min(0.8, face_depth + 0.05)
                    elif event.key == pygame.K_DOWN:
                        face_depth = max(0.2, face_depth - 0.05)
                    elif event.key == pygame.K_LEFT:
                        parallax_sensitivity = max(0.0, parallax_sensitivity - 0.1)
                    elif event.key == pygame.K_RIGHT:
                        parallax_sensitivity = min(3.0, parallax_sensitivity + 0.1)
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame_count += 1
            
            # Process face only if enabled (disable for better FPS)
            if use_face_tracking:
                if frame_count % face_detect_interval == 0:
                    face_results = face_mesh.process(rgb_frame)
                    last_face_results = face_results
                else:
                    face_results = last_face_results
                
                if face_results and face_results.multi_face_landmarks:
                    face_detected = True
                    face_landmarks = face_results.multi_face_landmarks[0]
                    nose = face_landmarks.landmark[1]
                    face_smoothing.append((nose.x, nose.y))
                    avg_x = sum(f[0] for f in face_smoothing) / len(face_smoothing)
                    avg_y = sum(f[1] for f in face_smoothing) / len(face_smoothing)
                    face_position = [avg_x, avg_y]
                else:
                    face_detected = False
            else:
                # Always render when face tracking is off
                face_detected = True
            
            # Process hands
            if frame_count % hand_detect_interval == 0:
                results = hands.process(rgb_frame)
                last_hand_results = results
            else:
                results = last_hand_results
            
            # Clear previous hand squares
            hand_squares.clear()
            
            # Process hand positions for collision squares
            hand_squares_temp = []
            if results and results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if hand_idx < 2:
                        x_coords = [lm.x for lm in hand_landmarks.landmark]
                        y_coords = [lm.y for lm in hand_landmarks.landmark]
                        
                        hand_center_x = sum(x_coords) / len(x_coords) * WIDTH
                        hand_center_y = sum(y_coords) / len(y_coords) * HEIGHT
                        
                        hand_smoothing[hand_idx].append((hand_center_x, hand_center_y))
                        if len(hand_smoothing[hand_idx]) > 0:
                            avg_x = sum(h[0] for h in hand_smoothing[hand_idx]) / len(hand_smoothing[hand_idx])
                            avg_y = sum(h[1] for h in hand_smoothing[hand_idx]) / len(hand_smoothing[hand_idx])
                            
                            # Calculate hand size
                            min_x = min(x_coords) * WIDTH
                            max_x = max(x_coords) * WIDTH
                            hand_width = max_x - min_x
                            square_size = max(0.05, min(0.15, hand_width / WIDTH * 0.3))
                            
                            hand_squares.append((avg_x, avg_y, square_size))
                            hand_squares_temp.append((avg_x, avg_y, square_size))
            
            # Handle countdown - count continuous hand presence
            if not game_started and not game_over and not game_won:
                if hand_squares_temp:  # Hand(s) detected
                    if not countdown_active:
                        # First detection - start countdown
                        countdown_active = True
                        countdown_timer = 0.0
                    
                    # Increment timer while hand is present
                    countdown_timer += dt
                    
                    if countdown_timer >= countdown_time:
                        # Countdown finished - start game
                        game_started = True
                        countdown_active = False
                        countdown_timer = 0.0
                        # Launch ball into tunnel
                        angle_x = random.uniform(-0.3, 0.3)
                        angle_y = random.uniform(-0.3, 0.3)
                        ball_velocity_3d = [angle_x, angle_y, -1.0]
                else:
                    # No hand detected - reset countdown
                    if countdown_active:
                        countdown_active = False
                        countdown_timer = 0.0
            
            # Update game physics
            if game_started and not game_over and not game_won:
                # Update ball position
                ball_position_3d[0] += ball_velocity_3d[0] * ball_speed * dt
                ball_position_3d[1] += ball_velocity_3d[1] * ball_speed * dt
                ball_position_3d[2] += ball_velocity_3d[2] * ball_speed * dt
                
                # Check collisions
                check_brick_collision()
                if hand_squares_data:
                    check_hand_collision(hand_squares_data)
                
                # Bounce off tunnel walls
                tunnel_half_width = SCREEN_WIDTH_M * 1.5
                tunnel_half_height = SCREEN_HEIGHT_M * 1.5
                
                if abs(ball_position_3d[0]) > tunnel_half_width:
                    ball_velocity_3d[0] = -ball_velocity_3d[0]
                    ball_position_3d[0] = np.sign(ball_position_3d[0]) * tunnel_half_width
                
                if abs(ball_position_3d[1]) > tunnel_half_height:
                    ball_velocity_3d[1] = -ball_velocity_3d[1]
                    ball_position_3d[1] = np.sign(ball_position_3d[1]) * tunnel_half_height
                
                # Bounce off back wall
                if ball_position_3d[2] < BRICK_DEPTH - 0.2:
                    ball_velocity_3d[2] = abs(ball_velocity_3d[2])
                
                # Check if ball passed through screen (lost a life)
                if ball_position_3d[2] > 0.5:
                    lives -= 1
                    
                    if lives <= 0:
                        game_over = True
                    else:
                        # Reset for next round - require countdown again
                        game_started = False
                        countdown_active = False
                        countdown_timer = 0.0
                        ball_position_3d = [0.0, 0.0, -0.3]
                        ball_velocity_3d = [0.0, 0.0, 0.0]
                        ball_speed = 0.8
                
                # Check win condition
                if all(not b['active'] for b in bricks):
                    game_won = True
            
            # Clear screen
            screen.fill((5, 5, 15))
            
            # Draw everything if face detected
            if face_detected:
                frustum, eye_x, eye_y, parallax_sens = draw_3d_grid(
                    screen, face_position[0], face_position[1], face_depth, parallax_sensitivity
                )
                
                # Draw back wall (farthest back)
                draw_back_wall(screen, frustum, eye_x, eye_y, parallax_sens)
                
                # Draw bricks (in front of wall) - batch rendering for performance
                # Create single transparent surface for all bricks
                brick_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                brick_data = []
                for brick in bricks:
                    result = draw_brick(screen, brick, frustum, eye_x, eye_y, parallax_sens)
                    if result:
                        brick_data.append(result)
                
                # Draw all bricks on transparent surface at once
                for points, color in brick_data:
                    try:
                        color_with_alpha = (*color, 150)
                        pygame.draw.polygon(brick_surface, color_with_alpha, points)
                        pygame.draw.polygon(screen, (255, 255, 255), points, 2)
                    except:
                        pass
                
                screen.blit(brick_surface, (0, 0))
                
                # Draw ball
                draw_ball_3d(screen, frustum, eye_x, eye_y, parallax_sens)
                
                # Draw hand squares (at screen plane) - batch rendering
                hand_squares_data = []
                # Create single transparent surface for all hand squares
                hand_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                
                for hx, hy, size in hand_squares:
                    # Simplified rendering
                    world_x = ((hx / WIDTH) - 0.5) * SCREEN_WIDTH_M * 2.5
                    world_y = (0.5 - (hy / HEIGHT)) * SCREEN_HEIGHT_M * 2.5
                    world_z = 0.0
                    
                    half_pixels = int((size * WIDTH / SCREEN_WIDTH_M) / 2.5)
                    rect = pygame.Rect(int(hx) - half_pixels, int(hy) - half_pixels, half_pixels * 2, half_pixels * 2)
                    
                    try:
                        # Draw on transparent surface
                        pygame.draw.rect(hand_surface, (0, 255, 100, 80), rect)
                        pygame.draw.rect(screen, (0, 255, 100), rect, 3)
                    except:
                        pass
                    
                    hand_squares_data.append((world_x, world_y, world_z, size))
                
                screen.blit(hand_surface, (0, 0))
            
            # Draw UI
            fps = clock.get_fps()
            draw_ui(screen, fps)
            
            pygame.display.flip()
            clock.tick(target_fps)
    
    except Exception as e:
        print(f"Error during game: {e}")
    finally:
        # Clean up MediaPipe
        face_mesh.close()
        hands.close()
        cap.release()
        pygame.quit()


if __name__ == "__main__":
    main()
