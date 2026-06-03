import cv2
import numpy as np
import mediapipe as mp
import pygame
import pymunk
import pymunk.pygame_util
from collections import deque
import random

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Particle Physics - Hand Colliders")
clock = pygame.time.Clock()

# Initialize Pymunk physics
space = pymunk.Space()
space.gravity = (0, 200)  # Moderate gravity
space.damping = 0.98  # Air resistance

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Particle settings
particle_color = (100, 150, 255)
NUM_PARTICLES = 1000  # Lots of particles!

# Hand collider bodies (will be updated each frame)
hand_colliders = []


def clamp_color(r, g, b):
    """Clamp RGB values to valid range 0-255"""
    return (
        max(0, min(255, r)),
        max(0, min(255, g)),
        max(0, min(255, b))
    )


def create_physics_particle(x, y):
    """Create a particle with physics"""
    mass = random.uniform(0.3, 1.0)
    radius = random.randint(5, 10)
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
    return body, shape


def create_hand_collider(landmarks, w, h):
    """Create a polygon collider around the hand"""
    # Key landmarks for hand outline
    # We'll use: wrist, thumb tip, index tip, middle tip, ring tip, pinky tip, pinky base
    key_points = [0, 4, 8, 12, 16, 20, 17, 5]  # Creates a rough hand shape

    vertices = []
    for idx in key_points:
        lm = landmarks.landmark[idx]
        x = lm.x * w
        y = lm.y * h
        vertices.append((x, y))

    # Create kinematic body (moves but isn't affected by physics)
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)

    # Calculate centroid for body position
    centroid_x = sum(v[0] for v in vertices) / len(vertices)
    centroid_y = sum(v[1] for v in vertices) / len(vertices)
    body.position = centroid_x, centroid_y

    # Convert vertices to be relative to body position
    relative_vertices = [(v[0] - centroid_x, v[1] - centroid_y) for v in vertices]

    # Create convex hull from points
    try:
        shape = pymunk.Poly(body, relative_vertices)
        shape.friction = 0.1
        shape.elasticity = 0.3  # Low elasticity for softer collisions
        return body, shape
    except:
        # Fallback to circle if polygon fails
        shape = pymunk.Circle(body, 80)
        shape.friction = 0.1
        shape.elasticity = 0.3
        return body, shape


def spawn_initial_particles():
    """Spawn all particles at start"""
    for _ in range(NUM_PARTICLES):
        x = random.randint(100, WIDTH - 100)
        y = random.randint(100, HEIGHT - 300)
        create_physics_particle(x, y)


def draw_ui(surface, particle_count):
    """Draw UI overlay"""
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    # Particle count
    count_text = font.render(f"Particles: {particle_count}", True, (255, 255, 255))
    surface.blit(count_text, (10, 10))

    # Instructions
    instructions = [
        "Move your hands to push particles!",
        "Particles bounce at random speeds (0 to 50% of collision speed)",
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
    global particle_color

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # Create boundaries (box the particles in)
    boundaries = [
        pymunk.Segment(space.static_body, (0, HEIGHT), (WIDTH, HEIGHT), 5),  # bottom
        pymunk.Segment(space.static_body, (0, 0), (WIDTH, 0), 5),  # top
        pymunk.Segment(space.static_body, (0, 0), (0, HEIGHT), 5),  # left
        pymunk.Segment(space.static_body, (WIDTH, 0), (WIDTH, HEIGHT), 5),  # right
    ]
    for boundary in boundaries:
        boundary.friction = 0.3
        boundary.elasticity = 0.8
        space.add(boundary)

    # Spawn initial particles
    spawn_initial_particles()

    gravity_on = True

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
    ) as hands:

        running = True

        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        # Respawn particles
                        for body in list(space.bodies):
                            if body.body_type == pymunk.Body.DYNAMIC:
                                for shape in body.shapes:
                                    space.remove(shape)
                                space.remove(body)
                        spawn_initial_particles()
                    elif event.key == pygame.K_g:
                        # Toggle gravity
                        gravity_on = not gravity_on
                        space.gravity = (0, 200) if gravity_on else (0, 0)
                    elif event.key == pygame.K_c:
                        # Change color
                        particle_color = (
                            random.randint(50, 255),
                            random.randint(50, 255),
                            random.randint(50, 255)
                        )
                        # Update existing particles
                        for body in space.bodies:
                            if body.body_type == pymunk.Body.DYNAMIC:
                                for shape in body.shapes:
                                    shape.color = clamp_color(
                                        random.randint(particle_color[0] - 50, particle_color[0] + 50),
                                        random.randint(particle_color[1] - 50, particle_color[1] + 50),
                                        random.randint(particle_color[2] - 50, particle_color[2] + 50)
                                    )

            # Get webcam frame
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Remove old hand colliders
            for body, shape in hand_colliders:
                space.remove(shape)
                space.remove(body)
            hand_colliders.clear()

            # Create new hand colliders
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    body, shape = create_hand_collider(hand_landmarks, WIDTH, HEIGHT)
                    space.add(body, shape)
                    hand_colliders.append((body, shape))

            # Update physics
            space.step(1 / 60.0)

            # Apply random velocity reduction on hand collision
            for body, shape in hand_colliders:
                # Check for particles near hand
                for particle_body in space.bodies:
                    if particle_body.body_type == pymunk.Body.DYNAMIC:
                        # Calculate distance between hand and particle
                        dx = particle_body.position.x - body.position.x
                        dy = particle_body.position.y - body.position.y
                        distance = (dx ** 2 + dy ** 2) ** 0.5

                        # If particle is close to hand (collision zone)
                        if distance < 100:
                            # Get current velocity
                            current_speed = particle_body.velocity.length

                            if current_speed > 0:
                                # Randomize speed between 0 and half of current speed
                                new_speed = random.uniform(0, current_speed * 0.5)

                                # Calculate direction away from hand
                                if distance > 0:
                                    angle = np.arctan2(dy, dx)
                                    # Add randomness
                                    angle += random.uniform(-0.5, 0.5)

                                    # Set new velocity
                                    particle_body.velocity = (
                                        np.cos(angle) * new_speed,
                                        np.sin(angle) * new_speed
                                    )

            # Clear screen
            screen.fill((15, 15, 25))

            # Draw boundaries
            pygame.draw.rect(screen, (60, 60, 80), (0, 0, WIDTH, HEIGHT), 8)

            # Draw particles
            particle_count = 0
            for body in space.bodies:
                if body.body_type == pymunk.Body.DYNAMIC:
                    for shape in body.shapes:
                        if isinstance(shape, pymunk.Circle):
                            pos = int(body.position.x), int(body.position.y)
                            pygame.draw.circle(screen, shape.color, pos, int(shape.radius))
                            particle_count += 1

            # Draw hand colliders (semi-transparent)
            for body, shape in hand_colliders:
                if isinstance(shape, pymunk.Poly):
                    # Get world coordinates of vertices
                    vertices = [v.rotated(body.angle) + body.position for v in shape.get_vertices()]
                    points = [(int(v.x), int(v.y)) for v in vertices]

                    # Draw filled polygon with transparency
                    s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                    pygame.draw.polygon(s, (0, 255, 0, 80), points)
                    screen.blit(s, (0, 0))

                    # Draw outline
                    pygame.draw.polygon(screen, (0, 255, 0), points, 3)
                elif isinstance(shape, pymunk.Circle):
                    pos = int(body.position.x), int(body.position.y)

                    # Draw filled circle with transparency
                    s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                    pygame.draw.circle(s, (0, 255, 0, 80), pos, int(shape.radius))
                    screen.blit(s, (0, 0))

                    # Draw outline
                    pygame.draw.circle(screen, (0, 255, 0), pos, int(shape.radius), 3)

            # Draw UI
            draw_ui(screen, particle_count)

            # Update display
            pygame.display.flip()
            clock.tick(60)

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()