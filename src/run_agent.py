
from matplotlib import pyplot as plt
import torch
from sim.env import DriftSimEnv
from sim.agent import DriftSimAgent

sim_env = DriftSimEnv(track_radius=15)
sim_agent = DriftSimAgent(env=sim_env)

data = sim_agent.learn(1000)

# Show training stats
timesteps, ep_mean_rews = zip(*data)

plt.figure(figsize=(8, 5))
plt.plot(timesteps, ep_mean_rews, marker="o", color="blue", label = "EP Mean Reward")

plt.xlabel("Timestep")
plt.ylabel("Episode Mean Reward")
plt.title("PPO Training Progress")
plt.grid(True)
plt.legend()
plt.show()

# Run the agent in the environment
plt.ion()

for i in range(1000):
    obs, _ = sim_env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = sim_agent.get_action(torch.tensor(obs, dtype=torch.float32))
        obs, reward, _, done, _ = sim_env.step(action)
        total_reward += reward

        plt.imshow(sim_env.render(), cmap='gray')
        plt.axis('off')
        plt.pause(0.05)
        plt.clf()