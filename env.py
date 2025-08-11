import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class DriftSimEnv(gym.Env):
    def __init__(self, width=200, height=200, slipperiness=0.9):
        super().__init__()
        pygame.init()
        self.width = width
        self.height = height
        # Slipperiness factor controls how much the car drifts (0.0-1.0)
        # Higher values make the car slide more in its previous direction
        self.slipperiness = slipperiness

        # Continuous action space: throttle [0,1], steering [-1,1]
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]),
                                       high=np.array([1.0, 1.0]),
                                       dtype=np.float32)

        # Observation space: 84x84 grayscale image
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width),
                                            dtype=np.uint8)

        self.screen = pygame.Surface((self.width, self.height))
        
        # Initialize track and car state by calling reset
        self.reset()

    def reset(self):
        # Reset car state
        self.car_x = self.width // 2
        self.car_y = self.height // 2
        self.car_velocity = 0.0
        self.car_angle = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        
        # Generate a random track (closed loop with random curvature)
        self.track_points = []
        
        # Generate track using parametric equation with random variations
        center_x, center_y = self.width // 2, self.height // 2
        radius = min(self.width, self.height) * 0.35  # Base radius
        
        # Random coefficients for the parametric equation
        a1 = np.random.uniform(0.1, 0.3)
        a2 = np.random.uniform(0.1, 0.2)
        b1 = np.random.uniform(2, 4)
        b2 = np.random.uniform(3, 6)
        
        # Generate the track points
        num_points = 60
        for i in range(num_points):
            t = 2 * np.pi * i / num_points
            r = radius * (1 + a1 * np.sin(b1 * t) + a2 * np.cos(b2 * t))
            x = center_x + r * np.cos(t)
            y = center_y + r * np.sin(t)
            self.track_points.append((int(x), int(y)))
        
        # Close the loop
        self.track_points.append(self.track_points[0])
        
        # Generate track inner and outer borders
        self.track_width = 20
        self.track_inner = []
        self.track_outer = []
        
        for i in range(len(self.track_points) - 1):
            p1 = np.array(self.track_points[i])
            p2 = np.array(self.track_points[(i + 1) % len(self.track_points)])
            
            # Calculate normal vector
            tangent = p2 - p1
            normal = np.array([-tangent[1], tangent[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-8)  # Normalize
            
            # Calculate inner and outer points
            inner = p1 - normal * (self.track_width / 2)
            outer = p1 + normal * (self.track_width / 2)
            
            self.track_inner.append((int(inner[0]), int(inner[1])))
            self.track_outer.append((int(outer[0]), int(outer[1])))
        
        return self.render_frame()

    def step(self, action):
        # Unpack and clamp action
        throttle = float(np.clip(action[0], 0.0, 1.0))
        steering = float(np.clip(action[1], -1.0, 1.0))

        # Lazy init state
        if not hasattr(self, "steer_angle"):
            self.steer_angle = 0.0
        if not hasattr(self, "steer_rate"):
            self.steer_rate = 0.0
        if not hasattr(self, "speed"):
            self.speed = 0.0

        # Params
        max_speed = 4.0
        throttle_accel = 0.5     # accel per step at full throttle
        drag_coeff = 0.08        # linear drag on speed

        max_steer_angle = 1.0    # clamp for steering angle
        steer_accel = 0.5       # steering input accelerates steering angle (via steer_rate)
        steer_drag_coeff = 0.1 # drag for steering
        steer_damping = 0.8     # damping on steering rate
        yaw_gain = 0.2          # turn rate per unit steering angle

        # Update steering dynamics:
        # steering input accelerates steering angle; yaw rate proportional to steer_angle
        self.steer_rate += steer_accel * steering
        self.steer_rate *= (1.0 - steer_damping)
        self.steer_angle += self.steer_rate
        self.steer_angle -= steer_drag_coeff * self.steer_angle
        self.steer_angle = float(np.clip(self.steer_angle, -max_steer_angle, max_steer_angle))

        # Update heading (turn rate proportional to current steering angle)
        yaw_rate = yaw_gain * self.steer_angle
        self.car_angle += yaw_rate
        # keep angle bounded
        if self.car_angle > np.pi:
            self.car_angle -= 2 * np.pi
        elif self.car_angle < -np.pi:
            self.car_angle += 2 * np.pi

        # Update forward speed (always move in heading direction)
        self.speed += throttle_accel * throttle
        self.speed -= drag_coeff * self.speed
        self.speed = float(np.clip(self.speed, 0.0, max_speed))

        # Desired velocity strictly along heading
        fwd = np.array([np.sin(self.car_angle), -np.cos(self.car_angle)], dtype=np.float32)
        desired_v = self.speed * fwd

        # Only position is affected by slipperiness: blend desired velocity with previous
        s = float(np.clip(self.slipperiness, 0.0, 1.0))
        prev_v = np.array([getattr(self, "velocity_x", 0.0), getattr(self, "velocity_y", 0.0)], dtype=np.float32)
        v = (1.0 - s) * desired_v + s * prev_v

        # Update state
        self.velocity_x, self.velocity_y = float(v[0]), float(v[1])
        self.car_velocity = float(np.linalg.norm(v))
        self.car_x += self.velocity_x
        self.car_y += self.velocity_y

        # Reward/termination
        reward = 1.0
        done = False
        info = {}

        obs = self.render_frame()
        return obs, reward, done, info
    
    def render_car_perspective(self):
        # Create a surface for the perspective view
        perspective_surface = pygame.Surface((self.width, self.height))
        perspective_surface.fill((0, 0, 0))
        
        # Car will be fixed at this position
        fixed_car_x = self.width // 2
        fixed_car_y = self.height - 20
        
        # Calculate transformation matrix for rendering track from car's perspective
        # Translate everything relative to car position and rotation
        def transform_point(point):
            # Translate point relative to car
            x, y = point[0] - self.car_x, point[1] - self.car_y
            
            # Rotate point by negative car angle (to align with car's forward direction)
            cos_theta = np.cos(-self.car_angle)
            sin_theta = np.sin(-self.car_angle)
            x_rot = x * cos_theta - y * sin_theta
            y_rot = x * sin_theta + y * cos_theta
            
            # Translate to fixed car position
            return (int(x_rot + fixed_car_x), int(y_rot + fixed_car_y))
        
        # Transform track points
        transformed_inner = [transform_point(p) for p in self.track_inner]
        transformed_outer = [transform_point(p) for p in self.track_outer]
        
        # Draw the transformed track
        border_color = (128, 128, 128)  # Gray color
        pygame.draw.lines(perspective_surface, border_color, True, transformed_inner, 2)
        pygame.draw.lines(perspective_surface, border_color, True, transformed_outer, 2)
        
        # Draw the car (fixed at bottom center)
        car_surf = pygame.Surface((5, 10))
        car_surf.fill((255, 255, 255))
        car_surf.set_colorkey((0, 0, 0))
        
        # Car is always pointing up in this view
        rect = car_surf.get_rect(center=(fixed_car_x, fixed_car_y))
        perspective_surface.blit(car_surf, rect)
        
        # Convert to grayscale observation
        frame = pygame.surfarray.array3d(perspective_surface)
        frame = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])  # RGB to grayscale
        frame = frame.astype(np.uint8)
        frame = np.transpose(frame, (1, 0))
        
        return frame

    def render_frame(self):
        self.screen.fill((0, 0, 0))
        
        # Draw track with inner and outer borders
        if len(self.track_inner) > 1 and len(self.track_outer) > 1:
            # Draw track borders
            border_color = (128, 128, 128)  # Gray color
            # Draw inner border
            pygame.draw.lines(self.screen, border_color, True, self.track_inner, 2)
            # Draw outer border
            pygame.draw.lines(self.screen, border_color, True, self.track_outer, 2)

        # Draw rotated car rectangle
        car_surf = pygame.Surface((5, 10))
        car_surf.fill((255, 255, 255))
        car_surf.set_colorkey((0, 0, 0))
        rotated = pygame.transform.rotate(car_surf, np.degrees(-self.car_angle))

        rect = rotated.get_rect(center=(self.car_x, self.car_y))
        self.screen.blit(rotated, rect)

        frame = pygame.surfarray.array3d(self.screen)
        frame = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])  # RGB to grayscale
        frame = frame.astype(np.uint8)
        frame = np.transpose(frame, (1, 0))

        return frame

    def close(self):
        pygame.quit()
