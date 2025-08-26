import torch
from torch.distributions import MultivariateNormal

from sim.env import DriftSimEnv
from sim.network import MLP

class DriftSimAgent:

    def __init__(self, num_rollout_episodes=4, discount_factor=0.95):

        # Hyperparameters
        self.num_rollout_episodes = num_rollout_episodes
        self.discount_factor = discount_factor

        # Covariance matrix for action sampling
        self.cov_var = torch.full(size=self.env.observation_space.shape, fill_value=0.3) # Fixed std value of 0.3
        self.cov_mat = torch.diag(self.cov_var)

        # Create env
        self.env = DriftSimEnv(track_radius=10)

        # Create actor/critic networks
        self.policy_network = MLP(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        # Size 1 since we are predicting the value of a given state (Value ~ Rt or the rewards-to-go)
        self.value_network = MLP(self.env.observation_space.shape[0], 1) 

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

            obs = self.env.reset()
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

        return states, actions, log_prob_actions, rewards_to_go