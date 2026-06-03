import cv2
import numpy as np
import mediapipe as mp
import pygame
import pymunk
import random
from collections import deque

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Parallax Particle Physics")
clock = pygame.time.Clock()

# Initialize Pymunk physics
space = pymunk.Space()
space.gravity = (0, 200)
space.damping = 0.98

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Particle settings
NUM_PARTICLES = 200
particle_color = (100, 150, 255)

# 3D depth levels (z-axis)
DEPTH_LEVELS = 5
MIN_DEPTH = 0.3  # closest (largest)
MAX_DEPTH = 1.5  # farthest (smallest)

# Face tracking for parallax
face_position = [0.5, 0.5]  # normalized x, y (center of screen)
face_smoothing = deque(maxlen=5)

# Hand colliders
hand_colliders = []


class Particle3D:
    """Particle with depth information"""

    def __init__(self, body, shape, depth):
        self.body = body
        self.shape = shape
        self.depth = depth  # 0.3 (close) to 1.5 (far)
        self.base_color = shape.color

    def get_screen_pos(self, parallax_offset_x, parallax_offset_y):
        """Get screen position with parallax offset based on depth"""
        # More depth = more parallax movement
        parallax_factor = (self.depth - 0.5) * 100
        x = self.body.position.x + parallax_offset_x * parallax_factor
        y = self.body.position.y + parallax_offset_y * parallax_factor
        return x, y

    def get_visual_radius(self):
        """Get radius scaled by depth (closer = bigger)"""
        base_radius = self.shape.radius
        scale = 1.0 / self.depth
        return int(base_radius * scale)

    def get_visual_color(self):
        """Get color adjusted by depth (farther = darker)"""
        r, g, b = self.base_color
        brightness = 1.0 / self.depth
        return (
            int(min(255, r * brightness)),
            int(min(255, g * brightness)),
            int(min(255, b * brightness))
        )


particles_3d = []


def clamp_color(r, g, b):
    """Clamp RGB values to valid range 0-255"""
    return (
        max(0, min(255, r)),
        max(0, min(255, g)),
        max(0, min(255, b))
    )


def create_physics_particle(x, y, depth):
    """Create a particle with physics and depth"""
    mass = random.uniform(0.3, 1.0)
    radius = random.randint(8, 15)
    moment = pymunk.moment_for_circle(mass, 0, radius)

    body = pymunk.Body(mass, moment)
    body.position = x, y
    body.velocity = (random.uniform(-50, 50), random.uniform(-50, 50))

    shape = pymunk.Circle(body, radius)
    shape.friction = 0.3
    shape.elasticity = 0.7
    shape.color = clamp_color(
        random.randint(particle_color[0] - 50, particle_color[0] + 50),
        random.randint(particle_color[1] - 50, particle_color[1] + 50),
        random.randint(particle_color[2] - 50, particle_color[2] + 50)
    )

    space.add(body, shape)

    # Create 3D particle wrapper
    particle = Particle3D(body, shape, depth)
    particles_3d.append(particle)

    return particle


def create_hand_collider(landmarks, w, h):
    """Create a polygon collider around the hand"""
    key_points = [0, 4, 8, 12, 16, 20, 17, 5]

    vertices = []
    for idx in key_points:
        lm = landmarks.landmark[idx]
        x = lm.x * w
        y = lm.y * h
        vertices.append((x, y))

    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)

    centroid_x = sum(v[0] for v in vertices) / len(vertices)
    centroid_y = sum(v[1] for v in vertices) / len(vertices)
    body.position = centroid_x, centroid_y

    relative_vertices = [(v[0] - centroid_x, v[1] - centroid_y) for v in vertices]

    try:
        shape = pymunk.Poly(body, relative_vertices)
        shape.friction = 0.1
        shape.elasticity = 0.3
        return body, shape
    except:
        shape = pymunk.Circle(body, 80)
        shape.friction = 0.1
        shape.elasticity = 0.3
        return body, shape


def spawn_initial_particles():
    """Spawn all particles at start with different depths"""
    particles_3d.clear()
    for _ in range(NUM_PARTICLES):
        x = random.randint(100, WIDTH - 100)
        y = random.randint(100, HEIGHT - 300)
        # Random depth between MIN_DEPTH and MAX_DEPTH
        depth = random.uniform(MIN_DEPTH, MAX_DEPTH)
        create_physics_particle(x, y, depth)


def draw_3d_grid(surface, parallax_offset_x, parallax_offset_y):
    """Draw perspective grid that recedes into the screen"""
    # Multiple grid layers at different depths
    for depth_layer in range(5):
        depth = 0.3 + depth_layer * 0.3  # 0.3, 0.6, 0.9, 1.2, 1.5

        # Grid parameters
        grid_spacing = int(80 / depth)

        # Parallax offset based on depth
        parallax_factor = (depth - 0.5) * 150
        offset_x = parallax_offset_x * parallax_factor
        offset_y = parallax_offset_y * parallax_factor

        # Color fades with depth
        brightness = int(100 / depth)
        color_rgb = (
            max(0, min(255, brightness)),
            max(0, min(255, brightness)),
            max(0, min(255, brightness + 30))
        )

        # Perspective scaling
        scale = 1.0 / depth

        # Alpha based on depth
        alpha = max(0, min(255, 255 - depth_layer * 50))

        # Draw vertical lines
        for i in range(-5, 25):
            x_base = i * grid_spacing + WIDTH // 2

            # Apply parallax
            x = x_base + offset_x

            # Perspective: lines converge to center as they go back
            top_x = WIDTH // 2 + (x - WIDTH // 2) * 0.5
            bottom_x = x

            # Only draw if on screen
            if -200 < bottom_x < WIDTH + 200:
                s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

                # Adjust y with parallax
                top_y = -200 + offset_y * 0.3
                bottom_y = HEIGHT + offset_y

                # Create color with alpha
                line_color = (color_rgb[0], color_rgb[1], color_rgb[2], alpha)

                pygame.draw.line(s, line_color,
                                 (top_x, top_y),
                                 (bottom_x, bottom_y),
                                 max(1, int(2 / depth)))
                surface.blit(s, (0, 0))

        # Draw horizontal lines
        for j in range(-3, 15):
            y_base = j * grid_spacing + HEIGHT // 2
            y = y_base + offset_y

            # Perspective width
            left_margin = (y / HEIGHT) * WIDTH * 0.3
            right_margin = WIDTH - left_margin

            if -100 < y < HEIGHT + 100:
                s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

                left_x = left_margin + offset_x * 0.7
                right_x = right_margin + offset_x * 0.7

                # Create color with alpha
                line_color = (color_rgb[0], color_rgb[1], color_rgb[2], alpha)

                pygame.draw.line(s, line_color,
                                 (left_x, y),
                                 (right_x, y),
                                 max(1, int(2 / depth)))
                surface.blit(s, (0, 0))


def draw_ui(surface, particle_count):
    """Draw UI overlay"""
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    count_text = font.render(f"Particles: {particle_count}", True, (255, 255, 255))
    surface.blit(count_text, (10, 10))

    depth_text = small_font.render("3D PARALLAX MODE", True, (0, 255, 200))
    surface.blit(depth_text, (10, 50))

    instructions = [
        "Move your face to control parallax!",
        "Move your hands to push particles!",
        "",
        "Controls:",
        "R: Respawn Particles",
        "G: Toggle Gravity",
        "C: Change Color",
        "Q: Quit"
    ]

    y_offset = HEIGHT - 180
    for i, line in enumerate(instructions):
        text = small_font.render(line, True, (200, 200, 200))
        surface.blit(text, (10, y_offset + i * 25))


def main():
    global particle_color, face_position

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

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

    spawn_initial_particles()

    gravity_on = True

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
    ) as hands, mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        for body in list(space.bodies):
                            if body.body_type == pymunk.Body.DYNAMIC:
                                for shape in body.shapes:
                                    space.remove(shape)
                                space.remove(body)
                        spawn_initial_particles()
                    elif event.key == pygame.K_g:
                        gravity_on = not gravity_on
                        space.gravity = (0, 200) if gravity_on else (0, 0)
                    elif event.key == pygame.K_c:
                        particle_color = (
                            random.randint(50, 255),
                            random.randint(50, 255),
                            random.randint(50, 255)
                        )
                        for p in particles_3d:
                            p.base_color = clamp_color(
                                random.randint(particle_color[0] - 50, particle_color[0] + 50),
                                random.randint(particle_color[1] - 50, particle_color[1] + 50),
                                random.randint(particle_color[2] - 50, particle_color[2] + 50)
                            )

            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process face for parallax
            face_results = face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                # Use nose tip (landmark 1) for face position
                nose = face_landmarks.landmark[1]
                face_smoothing.append((nose.x, nose.y))

                # Average for smooth movement
                avg_x = sum(f[0] for f in face_smoothing) / len(face_smoothing)
                avg_y = sum(f[1] for f in face_smoothing) / len(face_smoothing)
                face_position = [avg_x, avg_y]

            # Process hands
            hand_results = hands.process(rgb_frame)

            # Remove old hand colliders
            for body, shape in hand_colliders:
                space.remove(shape)
                space.remove(body)
            hand_colliders.clear()

            # Create new hand colliders
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    body, shape = create_hand_collider(hand_landmarks, WIDTH, HEIGHT)
                    space.add(body, shape)
                    hand_colliders.append((body, shape))

            # Update physics
            space.step(1 / 60.0)

            # Apply hand collision effects
            for body, shape in hand_colliders:
                for particle in particles_3d:
                    particle_body = particle.body
                    dx = particle_body.position.x - body.position.x
                    dy = particle_body.position.y - body.position.y
                    distance = (dx ** 2 + dy ** 2) ** 0.5

                    if distance < 100:
                        current_speed = particle_body.velocity.length
                        if current_speed > 0:
                            new_speed = random.uniform(0, current_speed * 0.5)
                            if distance > 0:
                                angle = np.arctan2(dy, dx)
                                angle += random.uniform(-0.5, 0.5)
                                particle_body.velocity = (
                                    np.cos(angle) * new_speed,
                                    np.sin(angle) * new_speed
                                )

            # Calculate parallax offset based on face position
            # Center is (0.5, 0.5), so offset from center
            parallax_offset_x = (face_position[0] - 0.5) * 2  # -1 to 1
            parallax_offset_y = (face_position[1] - 0.5) * 2

            # Clear screen with dark gradient
            screen.fill((5, 5, 15))

            # Draw 3D grid with parallax
            draw_3d_grid(screen, parallax_offset_x, parallax_offset_y)

            # Sort particles by depth (far to near) for proper layering
            sorted_particles = sorted(particles_3d, key=lambda p: p.depth, reverse=True)

            # Draw particles with depth and parallax
            particle_count = 0
            for particle in sorted_particles:
                screen_x, screen_y = particle.get_screen_pos(parallax_offset_x, parallax_offset_y)
                radius = particle.get_visual_radius()
                color = particle.get_visual_color()

                # Only draw if on screen
                if -100 < screen_x < WIDTH + 100 and -100 < screen_y < HEIGHT + 100:
                    pos = (int(screen_x), int(screen_y))

                    # Draw glow for depth effect
                    glow_radius = radius + 4
                    s = pygame.Surface((glow_radius * 4, glow_radius * 4), pygame.SRCALPHA)
                    glow_alpha = max(0, min(255, int(100 / particle.depth)))
                    glow_color = (color[0], color[1], color[2], glow_alpha)
                    pygame.draw.circle(s, glow_color, (glow_radius * 2, glow_radius * 2), glow_radius)
                    screen.blit(s, (pos[0] - glow_radius * 2, pos[1] - glow_radius * 2))

                    # Draw particle
                    pygame.draw.circle(screen, color, pos, radius)
                    pygame.draw.circle(screen, (255, 255, 255), pos, radius, 1)
                    particle_count += 1

            # Draw hand colliders
            for body, shape in hand_colliders:
                if isinstance(shape, pymunk.Poly):
                    vertices = [v.rotated(body.angle) + body.position for v in shape.get_vertices()]
                    points = [(int(v.x), int(v.y)) for v in vertices]

                    s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                    pygame.draw.polygon(s, (0, 255, 100, 100), points)
                    screen.blit(s, (0, 0))
                    pygame.draw.polygon(screen, (0, 255, 100), points, 3)
                elif isinstance(shape, pymunk.Circle):
                    pos = int(body.position.x), int(body.position.y)
                    s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                    pygame.draw.circle(s, (0, 255, 100, 100), pos, int(shape.radius))
                    screen.blit(s, (0, 0))
                    pygame.draw.circle(screen, (0, 255, 100), pos, int(shape.radius), 3)

            # Draw boundaries
            pygame.draw.rect(screen, (40, 40, 60), (0, 0, WIDTH, HEIGHT), 8)

            draw_ui(screen, particle_count)

            pygame.display.flip()
            clock.tick(60)

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()