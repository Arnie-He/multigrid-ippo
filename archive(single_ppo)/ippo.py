from copy import deepcopy
import math
import os
from itertools import count

import gym
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from envs import gym_multigrid
from envs.gym_multigrid import multigrid_envs

# You already have this helper.
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Your PPO agent (actor-critic) used for each independent agent.
class PPOAgent(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        input_dim = int(np.prod(observation_space.shape))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dim, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_dim, 128)), nn.Tanh(),
            layer_init(nn.Linear(128, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, action_space.n), std=0.01)
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class MultiGridFlattenObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to flatten Dict observation space for MultiGrid environment
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Calculate the total flattened observation size
        self.image_shape = env.observation_space["image"].shape
        self.direction_shape = env.observation_space["direction"].shape
        
        # Image will be flattened from (1, 5, 5, 3) to (75,)
        flattened_image_size = np.prod(self.image_shape)
        flattened_direction_size = np.prod(self.direction_shape)
        
        # Define the new flattened observation space
        total_size = flattened_image_size + flattened_direction_size
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(total_size,),
            dtype=np.float32
        )

    def observation(self, obs):
        # Extract components
        image = obs["image"]
        direction = obs["direction"]
        
        # Convert lists to numpy arrays if necessary
        if isinstance(image, list):
            image = np.array(image)
        if isinstance(direction, list):
            direction = np.array(direction)
        
        # Flatten components
        flat_image = image.reshape(-1).astype(np.float32)
        flat_direction = direction.reshape(-1).astype(np.float32)
        
        # Concatenate the flattened components
        return np.concatenate([flat_image, flat_direction])

class MultiGridBoxToDiscreteWrapper(gym.ActionWrapper):
    """
    Wrapper to convert Box action space to Discrete for MultiGrid environment.
    The Box action space is (0, 6, (1,), int64), so we need to convert it to Discrete(7)
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Check that the action space is a Box as expected
        assert isinstance(env.action_space, gym.spaces.Box), "The action space must be a Box"
        
        # Get the bounds of the Box
        low, high = env.action_space.low, env.action_space.high
        
        # Define the new Discrete action space
        self.num_actions = int(high[0] - low[0] + 1)  # 7 actions (0 to 6)
        self.action_space = gym.spaces.Discrete(self.num_actions)
        
    def action(self, act):
        # Convert Discrete action to Box action
        return np.array([act], dtype=self.env.action_space.dtype)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Ensure reward is a float (not a list)
        if isinstance(reward, list):
            reward = float(reward[0])
            
        return obs, reward, done, info

# The multi-agent (IPPO) class.
class MultiAgent():
    """
    A multi-agent (IPPO) implementation that uses independent PPO updates.
    """
    def __init__(self, config, env, device, training=True, with_expert=None, debug=False):
        self.config = config
        self.env = env
        self.device = device
        self.n_agents = env.n_agents  # assumes your env has this attribute

        # Create a PPOAgent and optimizer for each agent.
        self.agents = []
        self.optimizers = []
        for _ in range(self.n_agents):
            agent = PPOAgent(env.observation_space, env.action_space, self.n_agents).to(device)
            self.agents.append(agent)
            self.optimizers.append(optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5))
        self.total_steps = 0
        # If you use opponent modeling, set this flag accordingly.
        self.model_others = False

    def train(self):
        # Parameters from config:
        num_steps = self.config.num_steps
        batch_size = num_steps  # per agent (since each agent collects its own trajectory)
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda
        clip_coef = self.config.clip_coef

        # Determine observation dimension (flattened).
        obs_shape = self.env.observation_space.shape
        obs_dim = int(np.prod(obs_shape))

        # Preallocate trajectory buffers: shape (num_steps, n_agents, ...)
        obs_buf = torch.zeros((num_steps, self.n_agents, obs_dim), device=self.device)
        actions_buf = torch.zeros((num_steps, self.n_agents), dtype=torch.int64, device=self.device)
        logprobs_buf = torch.zeros((num_steps, self.n_agents), device=self.device)
        rewards_buf = torch.zeros((num_steps, self.n_agents), device=self.device)
        dones_buf = torch.zeros((num_steps, self.n_agents), device=self.device)
        values_buf = torch.zeros((num_steps, self.n_agents), device=self.device)

        # Reset the environment and convert the initial observation to tensor.
        next_obs = torch.tensor(self.env.reset(), dtype=torch.float32, device=self.device)
        # Expecting next_obs to be an array of shape (n_agents, obs_dim)
        next_done = torch.zeros(self.n_agents, device=self.device)

        global_step = 0
        num_iterations = self.config.total_timesteps // (num_steps * self.n_agents)

        for iteration in range(1, num_iterations + 1):
            # Collect trajectory data.
            for step in range(num_steps):
                # Store the current observation and done flags.
                obs_buf[step] = next_obs.view(self.n_agents, -1)
                dones_buf[step] = next_done

                current_actions = []
                current_values = []
                current_logprobs = []
                # For each agent, get action, log probability, entropy and value.
                for i in range(self.n_agents):
                    agent = self.agents[i]
                    # Prepare observation for agent i: shape (1, obs_dim)
                    obs_i = next_obs[i].view(1, -1)
                    a, logp, ent, value = agent.get_action_and_value(obs_i)
                    current_actions.append(a.item())
                    current_logprobs.append(logp)
                    current_values.append(value)
                # Store actions, logprobs, and values.
                actions_buf[step] = torch.tensor(current_actions, device=self.device)
                logprobs_buf[step] = torch.stack(current_logprobs).squeeze()
                values_buf[step] = torch.stack(current_values).squeeze()

                global_step += self.n_agents

                # Step the environment with the actions from all agents.
                next_obs_np, rewards, done, info = self.env.step(current_actions)
                if done:
                    next_obs_np = self.env.reset()
                next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device)
                rewards_buf[step] = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                next_done = torch.tensor(done, dtype=torch.float32, device=self.device)

            # Compute advantages and returns.
            advantages_buf = torch.zeros_like(rewards_buf, device=self.device)
            returns_buf = torch.zeros_like(rewards_buf, device=self.device)
            # Bootstrap last value for each agent.
            bootstrap_values = []
            for i in range(self.n_agents):
                obs_i = next_obs[i].view(1, -1)
                bootstrap_values.append(self.agents[i].get_value(obs_i))
            bootstrap_values = torch.stack(bootstrap_values).squeeze()

            lastgaelam = torch.zeros(self.n_agents, device=self.device)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = bootstrap_values
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
                lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                advantages_buf[t] = lastgaelam
            returns_buf = advantages_buf + values_buf

            # PPO update for each agent independently.
            for i in range(self.n_agents):
                # Flatten the trajectory for agent i.
                b_obs = obs_buf[:, i].view(num_steps, -1)
                b_actions = actions_buf[:, i]
                b_logprobs = logprobs_buf[:, i]
                b_advantages = advantages_buf[:, i]
                b_returns = returns_buf[:, i]
                b_values = values_buf[:, i]

                if self.config.norm_adv:
                    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                agent = self.agents[i]
                optimizer = self.optimizers[i]
                # Determine minibatch size (here, batch size is num_steps per agent).
                batch_inds = np.arange(num_steps)
                minibatch_size = num_steps // self.config.num_minibatches

                for epoch in range(self.config.update_epochs):
                    np.random.shuffle(batch_inds)
                    for start in range(0, num_steps, minibatch_size):
                        end = start + minibatch_size
                        mb_inds = batch_inds[start:end]

                        mb_obs = b_obs[mb_inds]
                        mb_actions = b_actions[mb_inds]
                        mb_logprobs = b_logprobs[mb_inds]
                        mb_advantages = b_advantages[mb_inds]
                        mb_returns = b_returns[mb_inds]
                        mb_values = b_values[mb_inds]

                        # Get new log probabilities, entropy and value predictions.
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
                        logratio = newlogprob - mb_logprobs
                        ratio = logratio.exp()

                        # PPO policy loss.
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss (with optional clipping).
                        newvalue = newvalue.view(-1)
                        if self.config.clip_vloss:
                            v_loss_unclipped = (newvalue - mb_returns) ** 2
                            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -clip_coef, clip_coef)
                            v_loss_clipped = (v_clipped - mb_returns) ** 2
                            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                        else:
                            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), self.config.max_grad_norm)
                        optimizer.step()

            # (Optional) Logging & checkpointing here.
            print(f"Iteration {iteration} completed. Global steps: {global_step}")

# Example configuration using a dataclass.
from dataclasses import dataclass

@dataclass
class Args:
    exp_name: str = "ippo_experiment"
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

# Example usage:
config = Args()
env = gym.make("MultiGrid-Cluttered-Fixed-15x15")  # make sure your env has attributes like n_agents, observation_space, etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ippo = MultiAgent(config, env, device)
ippo.train()