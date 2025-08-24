import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from util import smooth_closed_loop

class DriftSimEnv(gym.Env):
    def __init__(self, width=300, height=300, cam_width=60, cam_height=60, num_next_points=15, slipperiness=0.9, num_track_points=16, track_windiness=0.5):

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

        self.closest_point_index = 0
        self.num_next_points = num_next_points

        #######################################################################
        # Car Mechanics
        #######################################################################

        self.steer_angle = 0.0
        self.steer_rate = 0.0
        self.speed = 0.0

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
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        
        # Generate a random track 
        # First generate a circle with a bunch of points, one of which is on the car's start coordinate
        # Offset these points (except for the starting point) away/towards the origin
        # Interpolate a track from these points

        self.track_points = []
        self.track_points_interpolated = np.array([])

        radius = self.start_x - self.width // 2

        # This is how far the track point can vary (perpendicularly) from its position on the circle
        max_offset_radius = (self.width // 2 - radius) * self.track_windiness
        
        print(radius, self.width)

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
        
        self.track_points_interpolated = smooth_closed_loop(np.array(self.track_points), n_points_per_segment=5).astype(int)
        
        return self.render()

    def step(self, action):
        # Unpack and clamp action
        throttle = float(np.clip(action[0], 0.0, 1.0))
        steering = float(np.clip(action[1], -1.0, 1.0))

        # Params
        max_speed = 3.0
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

        self.get_observation()
        obs = self.render()
        return obs, reward, done

    # Returns obs, rewards, done
    def get_observation(self):

        # Get the distances of all track points from the car's position
        car_pos = np.array([self.car_x, self.car_y])
        point_dist = np.sum((self.track_points_interpolated - car_pos) ** 2, axis=1)

        # Find the closest point
        self.closest_point_index = np.argmin(point_dist)

        # Get the next N points on the track ahead of the car
        # This needs to handle wrapping around the end of the array
        num_points_total = len(self.track_points_interpolated)
        indices = (self.closest_point_index - np.arange(self.num_next_points)) % num_points_total
        next_points = self.track_points_interpolated[indices]

        print(self.closest_point_index, indices)

        # Offset points 
        next_points = next_points - np.array([self.car_x, self.car_y])
        theta = -self.car_angle

        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        transformed_next_points = rotation_matrix @ next_points.T

        return transformed_next_points.T
    
    def render(self):
        # Create a surface for the perspective view
        perspective_surface = pygame.Surface((self.cam_width, self.cam_height))
        perspective_surface.fill((0, 0, 0))
        
        # Car will be fixed at this position
        screen_car_x = self.cam_width // 2
        screen_car_y = self.cam_height * 0.9
        
        # Draw track points
        # if self.track_points_interpolated.size > 0:
        #     pts = self.track_points_interpolated.astype(np.float32)

        #     # Translate points so car is origin
        #     rel = pts - np.array([self.car_x, self.car_y], dtype=np.float32)

        #     # Rotate points so car heading is up in the perspective view
        #     theta = -self.car_angle
        #     c, s = np.cos(theta), np.sin(theta)
        #     R = np.array([[c, -s],
        #           [s,  c]], dtype=np.float32)
        #     rotated = rel @ R.T  # row-vector multiplication: v' = R * v

        #     # Place car at fixed position in perspective surface
        #     screen_car_x = self.cam_width // 2
        #     screen_car_y = self.cam_height * 0.9
        #     screen_pts = rotated + np.array([screen_car_x, screen_car_y], dtype=np.float32)

        #     # Convert to integer points and draw
        #     screen_pts_int = [tuple(p.astype(int)) for p in screen_pts]
        #     if len(screen_pts_int) >= 2:
        #         pygame.draw.lines(perspective_surface, (0, 200, 255), True, screen_pts_int, 20)
        
        # draw circles at each track point in the perspective view
        # if self.track_points_interpolated.size > 0:
        #     pts = self.track_points_interpolated.astype(np.float32)
        #     rel = pts - np.array([self.car_x, self.car_y], dtype=np.float32)
        #     theta = -self.car_angle
        #     c, s = np.cos(theta), np.sin(theta)
        #     R = np.array([[c, -s],
        #                   [s,  c]], dtype=np.float32)
        #     rotated = rel @ R.T
        #     screen_pts = (rotated + np.array([screen_car_x, screen_car_y], dtype=np.float32)).astype(int)

        #     # Drawing points relative to car
        #     for i in range(len(screen_pts)):
        #         point = screen_pts[i]
        #         color = (150, 100, 100)

        #         pygame.draw.circle(perspective_surface, color, (int(point[0]), int(point[1])), 3)

        # Draw the transformed points (white circles)
        # These points are relative to the car's perspective
        transformed_points = self.get_observation() + np.array([screen_car_x, screen_car_y])

        for point in transformed_points:
            pygame.draw.circle(perspective_surface, (255, 255, 255), (int(point[0]), int(point[1])), 2)

        # Draw the car (fixed at bottom center)
        car_surf = pygame.Surface((5, 10))
        car_surf.fill((255, 255, 255))
        car_surf.set_colorkey((0, 0, 0))
        
        # Car is always pointing up in this view
        rect = car_surf.get_rect(center=(screen_car_x, screen_car_y))
        perspective_surface.blit(car_surf, rect)
        
        # Convert to grayscale observation
        frame = pygame.surfarray.array3d(perspective_surface)
        frame = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])  # RGB to grayscale
        frame = frame.astype(np.uint8)
        frame = np.transpose(frame, (1, 0))
        
        return frame

    def close(self):
        pygame.quit()
