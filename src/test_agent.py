import gymnasium as gym
import torch

from sim.agent import DriftSimAgent

def run_episode(env, model, render: bool = False) -> float:
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        action, _ = model.get_action(obs_tensor)

        # convert torch tensor actions to numpy if needed
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        if render:
            env.render()
        total_reward += reward
        done = terminated or truncated

    return total_reward

def main(train_steps: int = 300):
    # training environment
    env = gym.make("Pendulum-v1")
    model = DriftSimAgent(env)
    model.learn(train_steps)

    # render a single episode
    env_render = gym.make("Pendulum-v1", render_mode="human")
    try:
        total = run_episode(env_render, model, render=True)
        print(f"Episode total reward: {total}")
    finally:
        env_render.close()

if __name__ == "__main__":
    main()