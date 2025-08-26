import torch

from torch.distributions import MultivariateNormal
from torch.optim import Adam

from sim.env import DriftSimEnv
from sim.network import MLP

class DriftSimAgent:

    def __init__(self, env, num_rollout_episodes=4, discount_factor=0.95, ppo_clip=0.2):

        # Hyperparameters
        self.num_rollout_episodes = num_rollout_episodes
        self.discount_factor = discount_factor
        self.ppo_clip = ppo_clip
        self.learning_rate = 0.005
        self.num_updates_per_rollout = 5

        # Create env
        self.env = env #DriftSimEnv(track_radius=10)

        # Covariance matrix for action sampling
        self.cov_var = torch.full(size=(self.env.action_space.shape[0],), fill_value=0.3) # Fixed std value of 0.3
        self.cov_mat = torch.diag(self.cov_var)

        # Create actor/critic networks
        self.policy_network = MLP(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        # Size 1 since we are predicting the value of a given state (Value ~ Rt or the rewards-to-go)
        self.value_network = MLP(self.env.observation_space.shape[0], 1) 

        # Create optimizers
        self.policy_network_optimizer = Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.value_network_optimizer = Adam(self.value_network.parameters(), lr=self.learning_rate)

    def get_action(self, obs):

        # Create action probability distribution
        # Since cov_mat is just a diagonal matrix filled with our standard deviation value, the actions are independent
        # We are essentially sampling all our actions from n independent normal distributions
        mean_actions = self.policy_network(obs)
        distribution = MultivariateNormal(mean_actions, self.cov_mat)

        # Sample an action
        action = distribution.sample()
        log_prob_action = distribution.log_prob(action)

        return action.detach().numpy(), log_prob_action.detach()
    
    # Gets state action probs under the current policy network
    def get_action_probs(self, states, actions):

        mean_actions = self.policy_network(states)
        distribution = MultivariateNormal(mean_actions, self.cov_mat)

        return distribution.log_prob(actions)

    # Gets state values under the current value network
    def get_values(self, obs):
        return self.value_network(obs).squeeze()
    
    def calculate_rewards_to_go(self, rewards):

        rewards_to_go = []
        
        # Iterate backwards through each episode
        for episode in reversed(rewards):

            cumulative = 0

            # Iterate through each reward in the episode and find the discounted reward-to-go
            for reward in reversed(episode):
                cumulative = reward + self.discount_factor * cumulative
                # We insert it at zero since we're going backwards in rewards
                rewards_to_go.insert(0, cumulative)

        return rewards_to_go

    def rollout(self):
        
        states = []
        actions = []
        log_prob_actions = []

        # This is a 2d array of rewards for each episode
        rewards = []

        # This is what we actually care about, this is flattened like the rest of the arrays
        # Discounted rewards for each episode at each time step
        rewards_to_go = []
        
        # Run episodes
        for _ in range(self.num_rollout_episodes):

            obs, _ = self.env.reset()
            ep_rewards = []

            # Start episode
            while True:

                # Use the policy network to select an action
                action, log_prob_action = self.get_action(torch.tensor(obs, dtype=torch.float32))
                new_obs, reward, truncated, done, _ = self.env.step(action)

                # Store info for this step
                actions.append(action)
                log_prob_actions.append(log_prob_action)
                ep_rewards.append(reward)
                states.append(obs)

                # Update observation
                obs = new_obs

                # End episode
                if (truncated or done):
                    break
            
            rewards.append(ep_rewards)

        # Calculate rewards-to-go from raw rewards
        rewards_to_go = self.calculate_rewards_to_go(rewards)

        # Converting to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        log_prob_actions = torch.tensor(log_prob_actions, dtype=torch.float32)
        rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32)

        return states, actions, log_prob_actions, rewards_to_go, rewards
    
    def learn(self, num_rollouts):

        for _ in range(num_rollouts):

            # Rollout once
            rollout_states, rollout_actions, rollout_log_prob_actions, rollout_rewards_to_go, rewards = self.rollout()

            print(f"Mean episode reward: {sum(map(sum, rewards)) / len(rewards)}")

            # Get the state values
            values = self.get_values(rollout_states)

            # Use the values to calculate advantages
            # A = Q - V ~ Rtg - V
            advantages = rollout_rewards_to_go - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalize

            for _ in range(self.num_updates_per_rollout):

                # OPTIMIZE POLICY NETWORK

                # Calculate PPO ratio
                action_probs = self.get_action_probs(rollout_states, rollout_actions)
                ratios = torch.exp(action_probs - rollout_log_prob_actions)

                # Objective: mean(min(ratio * A, clip(ratio, 1-e, 1+e) * A)) -> core of PPO
                clipped_ratios = torch.clamp(ratios, 1 - self.ppo_clip, 1 + self.ppo_clip)
                # We take the negative here since optimizers minimize loss, but this is an objective that we want to maximize
                loss_policy = -torch.mean(torch.min(ratios * advantages, clipped_ratios * advantages))

                # Run one optimizer step
                self.policy_network_optimizer.zero_grad()
                loss_policy.backward()
                self.policy_network_optimizer.step()

                # OPTIMIZE VALUE NETWORK

                loss_value = torch.mean((rollout_rewards_to_go - self.get_values(rollout_states))**2)

                self.value_network_optimizer.zero_grad()
                loss_value.backward()
                self.value_network_optimizer.step()
