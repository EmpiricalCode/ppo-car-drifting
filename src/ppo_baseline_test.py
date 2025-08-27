import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from sim.env import DriftSimEnv

# Loading env
env = DriftSimEnv(cam_height=200, cam_width=200, track_radius=15)

# Create PPO agent
model = PPO.load("ppo_drift_car")

frames = []

# Test the agent
obs, _ = env.reset()
for _ in range(10000):
    
    # Stepping through env
    action, _states = model.predict(obs)
    obs, reward, truncated, done, info = env.step(action)
    
    # Storing rendered frame
    frames.append(env.render())

    # Calculate steps per second
    if _ == 0:
        start_time = time.time()
    if _ % 100 == 0 and _ > 0:
        end_time = time.time()
        steps_per_second = 100 / (end_time - start_time)
        print(f"Steps per second: {steps_per_second:.2f}")
        start_time = time.time()
    
    # If terminated, reset env
    if done or truncated:
        obs, _ = env.reset()

# Display the frames using matplotlib
fig, ax = plt.subplots()
for frame in frames:
    ax.imshow(frame)
    plt.pause(0.05)
    ax.clear()

plt.close(fig)