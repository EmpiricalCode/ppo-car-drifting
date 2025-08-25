import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from util import smooth_closed_loop

class DriftSimEnv(gym.Env):
    def __init__(self, width=300, height=300, cam_width=60, cam_height=60, max_speed=3, max_steer_angle=1, num_next_points=15, slipperiness=0.9, num_track_points=16, track_windiness=0.5):

        super().__init__()
        pygame.init()

        self.width = width
        self.height = height

        self.cam_width = cam_width
        self.cam_height = cam_height

        #######################################################################
        # Track 
        #######################################################################

        self.num_track_points = num_track_points
        self.track_windiness = track_windiness
        self.track_points = []
        self.track_points_interpolated = np.array([])
        self.num_next_points = num_next_points

        #######################################################################
        # Car Mechanics
        #######################################################################

        self.steer_angle = 0.0
        self.steer_rate = 0.0
        self.speed = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0

        self.max_speed = max_speed
        self.max_steer_angle = max_steer_angle

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
        
        # Start Coordinates (Middle Right)
        self.start_x = width * 4 // 5
        self.start_y = height // 2

        #######################################################################
        # PyGame
        #######################################################################
        
        self.screen = pygame.Surface((self.width, self.height))
        
        # Initialize track and car state by calling reset
        self.reset()

    def reset(self):
        # Reset car state
        self.car_x = self.start_x
        self.car_y = self.start_y
        
        self.car_velocity = 0.0
        self.car_angle = 0.0
        self.steer_rate = 0.0
        self.steer_angle = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.speed = 0.0

        # Generate a random track 
        # First generate a circle with a bunch of points, one of which is on the car's start coordinate
        # Offset these points (except for the starting point) away/towards the origin
        # Interpolate a track from these points

        self.track_points = []
        self.track_points_interpolated = np.array([])

        radius = self.start_x - self.width // 2

        # This is how far the track point can vary (perpendicularly) from its position on the circle
        max_offset_radius = (self.width // 2 - radius) * self.track_windiness
        
        for index in range(0, self.num_track_points):

            current_angle = (index / self.num_track_points) * 2 * np.pi

            current_x = self.width // 2 + radius * np.cos(current_angle)
            current_y = self.height // 2 + radius * np.sin(current_angle)

            # Offset each point a random value between (-max_offset_radius, max_offset_radius) perpendicularly away/towards the circle origin
            # Offset is 0 if it's the starting point
            offset_radius = np.random.uniform(-max_offset_radius, max_offset_radius)
            
            perp_offset_x = 0 if index == 0 else offset_radius * np.cos(current_angle)
            perp_offset_y = 0 if index == 0 else offset_radius * np.sin(current_angle)

            # append point to track_points
            # (circle_x + circle_y) + (perpendicular_offset_x, perpendicular_offset_y)
            self.track_points.append((current_x + perp_offset_x, current_y + perp_offset_y))
        
        self.track_points_interpolated = smooth_closed_loop(np.array(self.track_points), n_points_per_segment=20).astype(int)
        
        return self.render()

    def step(self, action):
        # Unpack and clamp action
        throttle = float(np.clip(action[0], 0.0, 1.0))
        steering = float(np.clip(action[1], -1.0, 1.0))

        # Params
        throttle_accel = 0.5     # accel per step at full throttle
        drag_coeff = 0.08        # linear drag on speed

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
        self.steer_angle = float(np.clip(self.steer_angle, -self.max_steer_angle, self.max_steer_angle))

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
        self.speed = float(np.clip(self.speed, 0.0, self.max_speed))

        # Desired velocity strictly along heading
        fwd = np.array([np.sin(self.car_angle), -np.cos(self.car_angle)], dtype=np.float32)
        desired_v = self.speed * fwd

        # Only position is affected by slipperiness: blend desired velocity with previous
        s = float(np.clip(self.slipperiness, 0.0, 1.0))
        prev_v = np.array([self.velocity_x, self.velocity_y], dtype=np.float32)
        v = (1.0 - s) * desired_v + s * prev_v

        # Update state
        self.velocity_x, self.velocity_y = float(v[0]), float(v[1])
        self.car_velocity = float(np.linalg.norm(v))
        self.car_x += self.velocity_x
        self.car_y += self.velocity_y

        # Reward/termination
        reward = 1.0
        done = False

        self.get_observation()
        obs = self.render()
        return obs, reward, done

    # Returns obs
    def get_observation(self):

        # Get the distances of all track points from the car's position
        car_pos = np.array([self.car_x, self.car_y])
        point_dist = np.sqrt(np.sum((self.track_points_interpolated - car_pos) ** 2, axis=1))

        # Find the closest point
        closest_point_index = np.argmin(point_dist)

        # Calculate rotation matrix for offsetting points relative to the car
        theta = -self.car_angle
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        # Get the next N points on the track ahead of the car
        # This needs to handle wrapping around the end of the array
        num_points_total = len(self.track_points_interpolated)
        indices = (closest_point_index - np.arange(0, 5*self.num_next_points, 5)) % num_points_total
        next_points = self.track_points_interpolated[indices]

        # Offset points 
        next_points = next_points - np.array([self.car_x, self.car_y])
        transformed_next_points = rotation_matrix @ next_points.T

        # Transforming velocity vector to be relative to the car
        car_velocity = np.array([self.velocity_x, self.velocity_y])
        transformed_car_velocity = rotation_matrix @ car_velocity.T

        # Calculate reward
        tangent = next_points[1] - next_points[0]
        car_speed_tangent = np.dot(car_velocity, tangent) / np.sqrt(np.sum(tangent ** 2))
        reward = car_speed_tangent / self.max_speed
        done = False

        if (point_dist[np.argmin(point_dist)] > 20):
            reward = -20
            done = True

        return (self.steer_rate, self.steer_angle, car_speed_tangent, transformed_car_velocity.T, transformed_next_points.T), (reward, done)
    
    def render(self):

        car_color = (255, 255, 255)
        points_color = (150, 150, 150)
        debug_color = (255, 0, 0)

        # Create a surface for the perspective view
        perspective_surface = pygame.Surface((self.cam_width, self.cam_height))
        perspective_surface.fill((0, 0, 0))
        
        # Car will be fixed at this position
        screen_car_x = self.cam_width // 2
        screen_car_y = self.cam_height * 0.9

        # Visualizing the agent's observations (next N points, velocity, steering)
        env_observation, env_state = self.get_observation()

        steer_rate, steer_angle, car_speed_tangent, car_velocity, next_points = env_observation
        reward, done = env_state
        
        # Draw the transformed points (white circles)
        # These points are relative to the car's perspective
        offset_points = next_points + np.array([screen_car_x, screen_car_y])        

        for point in offset_points:
            pygame.draw.circle(perspective_surface, points_color, (int(point[0]), int(point[1])), 2)

        # Display numbers
        font = pygame.font.Font(None, 16)
        steer_rate_text = font.render(f"Rate: {steer_rate:.2f}", True, debug_color)
        perspective_surface.blit(steer_rate_text, (5, 5))
        steer_angle_text = font.render(f"Angle: {steer_angle:.2f}", True, debug_color)
        perspective_surface.blit(steer_angle_text, (5, 20))
        speed_tan_text = font.render(f"Speed Tan: {car_speed_tangent:.2f}", True, debug_color)
        perspective_surface.blit(speed_tan_text, (5, 35))
        reward_text = font.render(f"Reward: {reward:.2f}", True, debug_color)
        perspective_surface.blit(reward_text, (5, 50))
        done_text = font.render(f"Done: {done}", True, debug_color)
        perspective_surface.blit(done_text, (5, 65))

        # Draw the velocity vector
        velocity_scale = 10
        start_pos = np.array([screen_car_x, screen_car_y])
        end_pos = start_pos + car_velocity * velocity_scale
        pygame.draw.line(perspective_surface, debug_color, start_pos, end_pos, 1)

        # Draw the tangent vector based on car's speed along the track tangent
        tangent_color = (0, 255, 0) # Green
        if len(next_points) > 1:
            tangent_vec = next_points[1] - next_points[0]
            norm = np.linalg.norm(tangent_vec)
            if norm > 0:
                tangent_unit_vec = tangent_vec / norm
                # Scale the tangent vector by the car's speed component along it and the velocity scale
                scaled_tangent = tangent_unit_vec * car_speed_tangent * velocity_scale
                end_pos_tangent = start_pos + scaled_tangent
                pygame.draw.line(perspective_surface, tangent_color, start_pos, end_pos_tangent, 1)

        # Draw the car (fixed at bottom center)
        car_surf = pygame.Surface((5, 10))
        car_surf.fill(car_color)
        car_surf.set_colorkey((0, 0, 0))
        
        # Car is always pointing up in this view
        rect = car_surf.get_rect(center=(screen_car_x, screen_car_y))
        perspective_surface.blit(car_surf, rect)
        
        # Convert to grayscale observation
        frame = pygame.surfarray.array3d(perspective_surface)
        frame = frame.transpose(1, 0, 2)
        frame = frame.astype(np.uint8)
        
        return frame

    def close(self):
        pygame.quit()
