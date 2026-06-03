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
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand-Controlled Brick Breaker")
clock = pygame.time.Clock()

# Initialize Pymunk physics
space = pymunk.Space()
space.gravity = (0, 0)  # No gravity for brick breaker

# MediaPipe Hands
mp_hands = mp.solutions.hands

# Game settings
LIVES = 3
BRICK_ROWS = 5
BRICK_COLS = 10
BRICK_WIDTH = WIDTH // BRICK_COLS - 10
BRICK_HEIGHT = 30
BRICK_PADDING = 5

# Game state
lives = LIVES
game_started = False
game_over = False
game_won = False

# Ball
ball_body = None
ball_shape = None
ball_radius = 10

# Paddles (for 2 hands)
paddles = []  # List of (body, shape, center_pos) tuples
paddle_width = 150
paddle_height = 20

# Bricks
bricks = []

# Hand tracking (for both hands)
hand_smoothing = [deque(maxlen=5), deque(maxlen=5)]  # One for each hand

# Ball speed management
base_ball_speed = 400
current_ball_speed = base_ball_speed

# Paddle height limit (bottom half of screen)
PADDLE_Y_LIMIT = HEIGHT // 2 + 50


def create_ball(x, y):
    """Create the ball"""
    global ball_body, ball_shape
    
    # Remove old ball if exists
    if ball_body:
        if ball_shape in space.shapes:
            space.remove(ball_shape)
        if ball_body in space.bodies:
            space.remove(ball_body)
    
    mass = 1
    moment = pymunk.moment_for_circle(mass, 0, ball_radius)
    
    ball_body = pymunk.Body(mass, moment)
    ball_body.position = x, y
    ball_body.velocity = (0, 0)  # Start stationary
    
    ball_shape = pymunk.Circle(ball_body, ball_radius)
    ball_shape.elasticity = 1.0  # Perfect bounce
    ball_shape.friction = 0.0
    ball_shape.collision_type = 1  # Ball collision type
    
    space.add(ball_body, ball_shape)


def create_paddle(x, y, width):
    """Create a paddle at hand position"""
    # Limit paddle to bottom half of screen
    y = max(y, PADDLE_Y_LIMIT)
    
    # Create kinematic body (controlled, not affected by physics)
    paddle_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    paddle_body.position = x, y
    
    # Create rectangular shape
    vertices = [
        (-width/2, -paddle_height/2),
        (width/2, -paddle_height/2),
        (width/2, paddle_height/2),
        (-width/2, paddle_height/2)
    ]
    
    paddle_shape = pymunk.Poly(paddle_body, vertices)
    paddle_shape.elasticity = 1.0
    paddle_shape.friction = 0.0
    paddle_shape.collision_type = 2  # Paddle collision type
    
    space.add(paddle_body, paddle_shape)
    
    return paddle_body, paddle_shape, (x, y)


def create_brick(x, y, color):
    """Create a brick"""
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = x, y
    
    vertices = [
        (-BRICK_WIDTH/2, -BRICK_HEIGHT/2),
        (BRICK_WIDTH/2, -BRICK_HEIGHT/2),
        (BRICK_WIDTH/2, BRICK_HEIGHT/2),
        (-BRICK_WIDTH/2, BRICK_HEIGHT/2)
    ]
    
    shape = pymunk.Poly(body, vertices)
    shape.elasticity = 1.0
    shape.friction = 0.0
    shape.collision_type = 3  # Brick collision type
    shape.color = color
    
    space.add(body, shape)
    bricks.append((body, shape))
    return body, shape


def setup_bricks():
    """Create the brick grid"""
    colors = [
        (255, 100, 100),  # Red
        (255, 150, 100),  # Orange
        (255, 255, 100),  # Yellow
        (100, 255, 100),  # Green
        (100, 150, 255),  # Blue
    ]
    
    start_y = 80
    
    for row in range(BRICK_ROWS):
        for col in range(BRICK_COLS):
            x = col * (BRICK_WIDTH + BRICK_PADDING) + BRICK_WIDTH/2 + BRICK_PADDING + 50
            y = row * (BRICK_HEIGHT + BRICK_PADDING) + BRICK_HEIGHT/2 + start_y
            
            color = colors[row % len(colors)]
            create_brick(x, y, color)


def check_brick_collisions():
    """Check and handle ball-brick collisions manually - only hit 1 brick at a time"""
    if not ball_body:
        return
    
    closest_brick = None
    closest_distance = float('inf')
    
    # Find the closest brick that's colliding
    for i, (body, shape) in enumerate(bricks):
        # Check distance between ball and brick center
        dx = ball_body.position.x - body.position.x
        dy = ball_body.position.y - body.position.y
        distance = (dx**2 + dy**2)**0.5
        
        # Simple collision detection - check if ball is near brick
        collision_dist = ball_radius + max(BRICK_WIDTH, BRICK_HEIGHT) / 2
        
        if distance < collision_dist:
            # Check more precise collision using bounding box
            ball_x = ball_body.position.x
            ball_y = ball_body.position.y
            brick_x = body.position.x
            brick_y = body.position.y
            
            # Brick bounds
            left = brick_x - BRICK_WIDTH / 2
            right = brick_x + BRICK_WIDTH / 2
            top = brick_y - BRICK_HEIGHT / 2
            bottom = brick_y + BRICK_HEIGHT / 2
            
            # Check if ball overlaps brick
            if (left - ball_radius < ball_x < right + ball_radius and
                top - ball_radius < ball_y < bottom + ball_radius):
                
                # Track closest colliding brick
                if distance < closest_distance:
                    closest_distance = distance
                    closest_brick = (i, body, shape)
    
    # Only remove and bounce off the closest brick
    if closest_brick:
        i, body, shape = closest_brick
        
        ball_x = ball_body.position.x
        ball_y = ball_body.position.y
        brick_x = body.position.x
        brick_y = body.position.y
        
        # Brick bounds
        left = brick_x - BRICK_WIDTH / 2
        right = brick_x + BRICK_WIDTH / 2
        top = brick_y - BRICK_HEIGHT / 2
        bottom = brick_y + BRICK_HEIGHT / 2
        
        # Bounce ball - determine which side was hit
        if abs(ball_x - left) < abs(ball_y - top) and abs(ball_x - left) < abs(ball_y - bottom):
            # Hit from left
            ball_body.velocity = (-abs(ball_body.velocity.x), ball_body.velocity.y)
        elif abs(ball_x - right) < abs(ball_y - top) and abs(ball_x - right) < abs(ball_y - bottom):
            # Hit from right
            ball_body.velocity = (abs(ball_body.velocity.x), ball_body.velocity.y)
        elif ball_y < brick_y:
            # Hit from top
            ball_body.velocity = (ball_body.velocity.x, -abs(ball_body.velocity.y))
        else:
            # Hit from bottom
            ball_body.velocity = (ball_body.velocity.x, abs(ball_body.velocity.y))
        
        # Remove the brick
        space.remove(shape)
        space.remove(body)
        bricks.pop(i)


def check_paddle_collisions():
    """Check if ball hits any paddle and increase speed"""
    global current_ball_speed
    
    if not ball_body:
        return
    
    for paddle_body, paddle_shape, center_pos in paddles:
        # Get paddle bounds
        vertices = [v.rotated(paddle_body.angle) + paddle_body.position for v in paddle_shape.get_vertices()]
        
        # Simple bounding box collision
        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]
        
        left = min(x_coords)
        right = max(x_coords)
        top = min(y_coords)
        bottom = max(y_coords)
        
        ball_x = ball_body.position.x
        ball_y = ball_body.position.y
        
        # Check collision
        if (left - ball_radius < ball_x < right + ball_radius and
            top - ball_radius < ball_y < bottom + ball_radius):
            
            # Ball hit paddle - bounce upward and increase speed
            if ball_body.velocity.y > 0:  # Only if moving downward
                # Increase speed by 1%
                current_ball_speed *= 1.01
                
                # Calculate bounce angle based on where ball hits paddle
                paddle_center_x = paddle_body.position.x
                paddle_w = right - left
                hit_offset = (ball_x - paddle_center_x) / (paddle_w / 2)
                hit_offset = max(-1, min(1, hit_offset))
                
                # Bounce with angle
                angle = hit_offset * 0.7  # Max 0.7 radians angle
                ball_body.velocity = (
                    current_ball_speed * np.sin(angle),
                    -current_ball_speed * abs(np.cos(angle))
                )


def check_ball_out_of_bounds():
    """Check if ball went below paddle"""
    global lives, game_started, game_over, current_ball_speed
    
    if ball_body and ball_body.position.y > HEIGHT + 50:
        lives -= 1
        game_started = False
        current_ball_speed = base_ball_speed  # Reset speed
        
        if lives <= 0:
            game_over = True
        else:
            # Reset ball below the paddle limit line
            create_ball(WIDTH // 2, PADDLE_Y_LIMIT + 80)


def draw_ui(surface, fps):
    """Draw UI elements"""
    font = pygame.font.Font(None, 48)
    small_font = pygame.font.Font(None, 32)
    tiny_font = pygame.font.Font(None, 24)
    
    # Lives
    lives_text = font.render(f"Lives: {lives}", True, (255, 255, 255))
    surface.blit(lives_text, (10, 10))
    
    # FPS counter
    fps_color = (100, 255, 100) if fps > 50 else (255, 255, 100) if fps > 30 else (255, 100, 100)
    fps_text = tiny_font.render(f"FPS: {int(fps)}", True, fps_color)
    surface.blit(fps_text, (10, 60))
    
    # Bricks remaining
    bricks_text = small_font.render(f"Bricks: {len(bricks)}", True, (200, 200, 200))
    surface.blit(bricks_text, (WIDTH - 200, 10))
    
    # Ball speed
    speed_text = small_font.render(f"Speed: {int(current_ball_speed)}", True, (200, 200, 200))
    surface.blit(speed_text, (WIDTH - 200, 40))
    
    # Instructions
    if not game_started and not game_over and not game_won:
        instruction_font = pygame.font.Font(None, 36)
        instruction = instruction_font.render("Move hand(s) under ball to start!", True, (255, 255, 100))
        text_rect = instruction.get_rect(center=(WIDTH // 2, HEIGHT - 120))
        surface.blit(instruction, text_rect)
        
        instruction2 = small_font.render("Use 1 or 2 hands | Speed increases 1% per paddle hit", True, (200, 200, 200))
        text_rect2 = instruction2.get_rect(center=(WIDTH // 2, HEIGHT - 80))
        surface.blit(instruction2, text_rect2)
        
        instruction3 = tiny_font.render("+/- keys: Adjust hand tracking frequency (for FPS)", True, (150, 150, 150))
        text_rect3 = instruction3.get_rect(center=(WIDTH // 2, HEIGHT - 50))
        surface.blit(instruction3, text_rect3)
    
    # Game Over
    if game_over:
        game_over_font = pygame.font.Font(None, 72)
        game_over_text = game_over_font.render("GAME OVER!", True, (255, 50, 50))
        text_rect = game_over_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        surface.blit(game_over_text, text_rect)
        
        restart_text = small_font.render("Press R to Restart | Q to Quit", True, (200, 200, 200))
        restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 60))
        surface.blit(restart_text, restart_rect)
    
    # Game Won
    if game_won:
        win_font = pygame.font.Font(None, 72)
        win_text = win_font.render("YOU WIN!", True, (100, 255, 100))
        text_rect = win_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        surface.blit(win_text, text_rect)
        
        restart_text = small_font.render("Press R to Restart | Q to Quit", True, (200, 200, 200))
        restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 60))
        surface.blit(restart_text, restart_rect)


def reset_game():
    """Reset the game"""
    global lives, game_started, game_over, game_won, bricks, current_ball_speed
    
    # Clear all physics objects
    for body, shape in bricks:
        if shape in space.shapes:
            space.remove(shape)
        if body in space.bodies:
            space.remove(body)
    
    bricks.clear()
    
    # Reset game state
    lives = LIVES
    game_started = False
    game_over = False
    game_won = False
    current_ball_speed = base_ball_speed
    
    # Create ball (below the paddle limit line)
    create_ball(WIDTH // 2, PADDLE_Y_LIMIT + 80)
    
    # Setup bricks
    setup_bricks()


def main():
    global game_started, game_won, paddles
    
    cap = cv2.VideoCapture(0)
    # Lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Limit camera FPS
    
    # Create initial ball (below the paddle limit line)
    create_ball(WIDTH // 2, PADDLE_Y_LIMIT + 80)
    
    # Setup bricks
    setup_bricks()
    
    # Create boundaries (walls only, not bottom)
    boundaries = [
        pymunk.Segment(space.static_body, (0, 0), (WIDTH, 0), 5),  # top
        pymunk.Segment(space.static_body, (0, 0), (0, HEIGHT), 5),  # left
        pymunk.Segment(space.static_body, (WIDTH, 0), (WIDTH, HEIGHT), 5),  # right
    ]
    for boundary in boundaries:
        boundary.elasticity = 1.0
        boundary.friction = 0.0
        space.add(boundary)
    
    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Track both hands
            model_complexity=0,  # Fastest model (0 is faster than 1)
            min_detection_confidence=0.5,  # Lower threshold for faster detection
            min_tracking_confidence=0.5
    ) as hands:
        
        running = True
        frame_count = 0
        hand_detect_interval = 2  # Process hands every N frames for performance
        last_hand_results = None
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        reset_game()
                    elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                        # Decrease hand detection frequency (better performance)
                        hand_detect_interval = min(5, hand_detect_interval + 1)
                        print(f"Hand detection interval: {hand_detect_interval} (Higher = Better FPS)")
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_KP_PLUS:
                        # Increase hand detection frequency (more responsive)
                        hand_detect_interval = max(1, hand_detect_interval - 1)
                        print(f"Hand detection interval: {hand_detect_interval} (Lower = More Responsive)")
            
            # Get webcam frame
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand detection only every N frames for performance
            frame_count += 1
            if frame_count % hand_detect_interval == 0:
                results = hands.process(rgb_frame)
                last_hand_results = results
            else:
                results = last_hand_results
            
            # Remove old paddles
            for paddle_body, paddle_shape, center_pos in paddles:
                space.remove(paddle_shape)
                space.remove(paddle_body)
            paddles.clear()
            
            # Process hand tracking for both hands
            if results and results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get hand center (average of all landmarks)
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    
                    hand_center_x = sum(x_coords) / len(x_coords) * WIDTH
                    hand_center_y = sum(y_coords) / len(y_coords) * HEIGHT
                    
                    # Smooth hand position
                    if hand_idx < 2:  # Only track up to 2 hands
                        hand_smoothing[hand_idx].append((hand_center_x, hand_center_y))
                        if len(hand_smoothing[hand_idx]) > 0:
                            avg_x = sum(h[0] for h in hand_smoothing[hand_idx]) / len(hand_smoothing[hand_idx])
                            avg_y = sum(h[1] for h in hand_smoothing[hand_idx]) / len(hand_smoothing[hand_idx])
                            
                            # Calculate hand width for paddle sizing
                            min_x = min(x_coords) * WIDTH
                            max_x = max(x_coords) * WIDTH
                            hand_width = max_x - min_x
                            
                            # Scale paddle to hand size (with limits)
                            paddle_w = max(100, min(250, hand_width * 1.2))
                            
                            # Create paddle at hand position
                            paddle_data = create_paddle(avg_x, avg_y, paddle_w)
                            paddles.append(paddle_data)
                            
                            # Check if paddle hits ball to start game
                            if not game_started and not game_over and not game_won:
                                if ball_body:
                                    paddle_body, paddle_shape, center_pos = paddle_data
                                    # Check if ball is near paddle
                                    dx = ball_body.position.x - paddle_body.position.x
                                    dy = ball_body.position.y - paddle_body.position.y
                                    distance = (dx**2 + dy**2)**0.5
                                    
                                    # Start game if hand touches ball from below
                                    if distance < 50 and ball_body.position.y < paddle_body.position.y:
                                        game_started = True
                                        # Launch ball upward with random angle
                                        angle = random.uniform(-0.5, 0.5)
                                        speed = base_ball_speed
                                        ball_body.velocity = (speed * np.sin(angle), -speed * np.cos(angle))
            
            # Update physics only if game is running
            if game_started and not game_over and not game_won:
                space.step(1 / 60.0)
                
                # Keep ball speed constant at current speed
                if ball_body:
                    current_speed = ball_body.velocity.length
                    if current_speed > 0:
                        ball_body.velocity = ball_body.velocity.normalized() * current_ball_speed
                
                # Check paddle collisions (increases speed)
                check_paddle_collisions()
                
                # Check brick collisions
                check_brick_collisions()
                
                # Check if ball is out of bounds
                check_ball_out_of_bounds()
                
                # Check win condition
                if len(bricks) == 0:
                    game_won = True
            
            # Clear screen
            screen.fill((20, 20, 40))
            
            # Draw paddle height limit line (subtle)
            pygame.draw.line(screen, (60, 60, 80), (0, PADDLE_Y_LIMIT), (WIDTH, PADDLE_Y_LIMIT), 2)
            
            # Draw bricks
            for body, shape in bricks:
                if isinstance(shape, pymunk.Poly):
                    vertices = [v.rotated(body.angle) + body.position for v in shape.get_vertices()]
                    points = [(int(v.x), int(v.y)) for v in vertices]
                    pygame.draw.polygon(screen, shape.color, points)
                    pygame.draw.polygon(screen, (255, 255, 255), points, 2)
            
            # Draw ball
            if ball_body:
                pos = int(ball_body.position.x), int(ball_body.position.y)
                pygame.draw.circle(screen, (255, 255, 255), pos, ball_radius)
                pygame.draw.circle(screen, (200, 200, 255), pos, ball_radius - 2)
            
            # Draw paddles
            for paddle_body, paddle_shape, center_pos in paddles:
                vertices = [v.rotated(paddle_body.angle) + paddle_body.position for v in paddle_shape.get_vertices()]
                points = [(int(v.x), int(v.y)) for v in vertices]
                
                # Draw semi-transparent fill
                s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                pygame.draw.polygon(s, (100, 255, 100, 120), points)
                screen.blit(s, (0, 0))
                
                # Draw outline
                pygame.draw.polygon(screen, (100, 255, 100), points, 4)
                
                # Draw center indicator (small sphere)
                center_x, center_y = center_pos
                pygame.draw.circle(screen, (255, 255, 100, 100), (int(center_x), int(center_y)), 8)
                pygame.draw.circle(screen, (255, 255, 150), (int(center_x), int(center_y)), 6)
            
            # Draw UI
            fps = clock.get_fps()
            draw_ui(screen, fps)
            
            pygame.display.flip()
            clock.tick(60)
    
    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
