from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from sim.env import DriftSimEnv

# Create env and wrap it in a tensorboard monitor
env = DriftSimEnv(track_radius=15)
env = Monitor(env)

# Create PPO agent for continuous actions
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="ppo_drift_tensorboard/")

# Train the agent
model.learn(total_timesteps=700000)
model.save("../ppo_drift_car")
