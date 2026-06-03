import cv2
import numpy as np
import pygame
import mediapipe as mp
import pymunk
import pymunk.pygame_util
from collections import deque
import random

# Initialize Pygame
pygame.init()

# Get display info and set window to nearly full screen size
display_info = pygame.display.Info()
WIDTH, HEIGHT = display_info.current_w - 100, display_info.current_h - 100

# Create resizable windowed mode
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("3D Grid Parallax with Particle Physics")
clock = pygame.time.Clock()

# Initialize Pymunk physics
space = pymunk.Space()
space.gravity = (0, 0)  # No gravity for floating particles
space.damping = 0.95  # Air resistance

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# MediaPipe Hands
mp_hands = mp.solutions.hands

# 3D Grid settings
GRID_SIZE = 10  # 10x10 grid
GRID_DEPTH = 1.5  # Depth in meters (scene depth)

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

# Particle settings
NUM_PARTICLES = 200
PARTICLE_DEPTH = -0.8  # Center depth in the tunnel

# Hand colliders (fingertips)
fingertip_colliders = []


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
    far = frustum['far']
    eye_z = frustum['eye_z']

    # Calculate depth from screen plane
    depth_from_screen = abs(z)
    
    # Parallax offset: farther objects appear to shift more when viewing angle changes
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
    screen_y = (1 - y_norm) * HEIGHT

    return screen_x, screen_y, scale


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

    # Grid dimensions in world space (meters)
    grid_width = SCREEN_WIDTH_M * 3.0
    grid_height = SCREEN_HEIGHT_M * 3.0

    # Generate all grid points
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

    # Horizontal lines at each depth
    for k in range(GRID_SIZE + 1):
        for i in range(GRID_SIZE):
            lines.append(((i, 0, k), (i + 1, 0, k), k))
        for i in range(GRID_SIZE):
            lines.append(((i, GRID_SIZE, k), (i + 1, GRID_SIZE, k), k))

    # Vertical lines at each depth
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

    # Sort by depth
    lines_with_depth = []
    for start, end, depth_layer in lines:
        if start in points and end in points:
            avg_z = (points[start][2] + points[end][2]) / 2
            lines_with_depth.append((avg_z, start, end))

    lines_with_depth.sort()

    # Draw all lines
    for avg_z, start, end in lines_with_depth:
        x1, y1, z1, scale1 = points[start]
        x2, y2, z2, scale2 = points[end]

        depth_ratio = abs(avg_z) / GRID_DEPTH
        brightness = int(80 + 120 * depth_ratio)
        brightness = max(50, min(255, brightness))

        if start[2] == 0:  # Front layer
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


def create_physics_particle(x, y):
    """Create a particle with physics at a specific 2D screen position"""
    mass = random.uniform(0.5, 1.0)
    radius = random.randint(3, 6)
    moment = pymunk.moment_for_circle(mass, 0, radius)

    body = pymunk.Body(mass, moment)
    body.position = x, y
    body.velocity = (random.uniform(-30, 30), random.uniform(-30, 30))

    shape = pymunk.Circle(body, radius)
    shape.friction = 0.3
    shape.elasticity = 0.8
    
    # Random color
    base_color = (100, 150, 255)
    shape.color = (
        max(0, min(255, random.randint(base_color[0] - 50, base_color[0] + 50))),
        max(0, min(255, random.randint(base_color[1] - 50, base_color[1] + 50))),
        max(0, min(255, random.randint(base_color[2] - 50, base_color[2] + 50)))
    )
    
    # Store depth for rendering
    body.depth_z = PARTICLE_DEPTH

    space.add(body, shape)
    return body, shape


def create_fingertip_collider(x, y, radius):
    """Create a spherical collider at a fingertip position"""
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    body.position = x, y
    
    shape = pymunk.Circle(body, radius)
    shape.friction = 0.1
    shape.elasticity = 0.5
    
    return body, shape


def spawn_initial_particles():
    """Spawn all particles at start"""
    for _ in range(NUM_PARTICLES):
        x = random.randint(100, WIDTH - 100)
        y = random.randint(100, HEIGHT - 100)
        create_physics_particle(x, y)


def main():
    global face_position, face_depth, parallax_sensitivity, screen, WIDTH, HEIGHT

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create boundaries
    boundaries = [
        pymunk.Segment(space.static_body, (0, HEIGHT), (WIDTH, HEIGHT), 5),
        pymunk.Segment(space.static_body, (0, 0), (WIDTH, 0), 5),
        pymunk.Segment(space.static_body, (0, 0), (0, HEIGHT), 5),
        pymunk.Segment(space.static_body, (WIDTH, 0), (WIDTH, HEIGHT), 5),
    ]
    for boundary in boundaries:
        boundary.friction = 0.3
        boundary.elasticity = 0.8
        space.add(boundary)

    # Spawn initial particles
    spawn_initial_particles()

    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh, mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
    ) as hands:

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

            # Process face
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

            # Process hands
            hand_results = hands.process(rgb_frame)
            
            # Remove old fingertip colliders
            for body, shape in fingertip_colliders:
                space.remove(shape)
                space.remove(body)
            fingertip_colliders.clear()

            # Create new fingertip colliders
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Fingertip landmarks: thumb(4), index(8), middle(12), ring(16), pinky(20)
                    fingertip_indices = [4, 8, 12, 16, 20]
                    
                    # Estimate hand depth based on hand size
                    wrist = hand_landmarks.landmark[0]
                    middle_tip = hand_landmarks.landmark[12]
                    hand_size = abs(middle_tip.y - wrist.y)
                    
                    # Larger hand = closer, smaller hand = farther
                    # Radius scales inversely with hand size
                    base_radius = 25
                    radius = int(base_radius * (hand_size / 0.3))
                    radius = max(10, min(40, radius))
                    
                    for tip_idx in fingertip_indices:
                        lm = hand_landmarks.landmark[tip_idx]
                        x = lm.x * WIDTH
                        y = lm.y * HEIGHT
                        
                        body, shape = create_fingertip_collider(x, y, radius)
                        space.add(body, shape)
                        fingertip_colliders.append((body, shape))

            # Update physics
            space.step(1 / 144.0)

            # Clear screen
            screen.fill((5, 5, 15))

            # Only draw grid if face is detected
            if face_detected:
                frustum, eye_x, eye_y, parallax_sens = draw_3d_grid(screen, face_position[0], face_position[1], face_depth, parallax_sensitivity)
            
            # Draw particles (they interact in 2D physics space, we just add depth visual effects)
            for body in space.bodies:
                if body.body_type == pymunk.Body.DYNAMIC:
                    for shape in body.shapes:
                        if isinstance(shape, pymunk.Circle):
                            # Use actual physics position for rendering
                            phys_x = body.position.x
                            phys_y = body.position.y
                            
                            if face_detected:
                                # Apply subtle depth-based parallax offset and scaling
                                # The depth is just visual - physics stays in 2D
                                world_x = ((phys_x / WIDTH) - 0.5) * SCREEN_WIDTH_M * 3.0
                                world_y = (0.5 - (phys_y / HEIGHT)) * SCREEN_HEIGHT_M * 3.0
                                world_z = body.depth_z
                                
                                # Project to screen with parallax (just for visual offset)
                                screen_x, screen_y, scale = project_3d_point(
                                    world_x, world_y, world_z, 
                                    frustum, eye_x, eye_y, parallax_sens
                                )
                                
                                # Draw particle with depth-based size
                                particle_radius = int(shape.radius * scale * 1.5)
                                particle_radius = max(3, min(12, particle_radius))
                            else:
                                # If no face detected, draw at physics position
                                screen_x = phys_x
                                screen_y = phys_y
                                particle_radius = int(shape.radius)
                            
                            if -50 < screen_x < WIDTH + 50 and -50 < screen_y < HEIGHT + 50:
                                pygame.draw.circle(screen, shape.color, 
                                                 (int(screen_x), int(screen_y)), 
                                                 particle_radius)
            
            # Draw fingertip colliders
            for body, shape in fingertip_colliders:
                pos = int(body.position.x), int(body.position.y)
                
                # Semi-transparent circle
                s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                pygame.draw.circle(s, (0, 255, 100, 60), pos, int(shape.radius))
                screen.blit(s, (0, 0))
                
                # Outline
                pygame.draw.circle(screen, (0, 255, 100), pos, int(shape.radius), 2)

            pygame.display.flip()
            clock.tick(144)

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
