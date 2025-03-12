import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from envs import gym_multigrid
from envs.gym_multigrid import multigrid_envs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self, obs_size):
        super().__init__()

        ACTION_DIM = 7

        obs_size = int(obs_size[0])

        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(obs_size, 128)),
            nn.Tanh(),
            self.layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            self.layer_init(nn.Linear(obs_size, 128)),
            nn.Tanh(),
            self.layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, ACTION_DIM), std=0.01),
        )
    
    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class sub_agent:
    def __init__(self, config, next_obs):
        self.obs_size = self.get_obs_size(next_obs)
        self.network = Agent(self.obs_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config["learning_rate"], eps = 1e-5)
        # initialize buffers
        self.init_buffer(config)
        self.next_obs = torch.tensor(self.flatten_obs(next_obs), dtype=torch.float32, device=device)
        self.next_done = 0.0

    def get_obs_size(self, next_obs):
        # get flattened observation size
        image_shape = next_obs["image"].shape
        flattened_image_size = np.prod(image_shape)
        obs_size = flattened_image_size + 1
        return (obs_size, )

    def flatten_obs(self, obs):
        image = np.array(obs["image"])
        direction = np.array(obs["direction"])
        flat_image = image.reshape(-1).astype(np.float32)
        flat_direction = direction.reshape(-1).astype(np.float32)
        return np.concatenate([flat_image, flat_direction])

    def init_buffer(self, config):
        # Initialize rollout buffers.
        self.obs_buffer = torch.zeros((config["num_steps"], ) + self.obs_size, device = device)
        self.actions_buffer = torch.zeros((config["num_steps"]), dtype=torch.long, device=device)
        self.logprobs_buffer = torch.zeros(config["num_steps"], device=device)
        self.rewards_buffer = torch.zeros(config["num_steps"], device=device)
        self.dones_buffer = torch.zeros(config["num_steps"], device=device)
        self.values_buffer = torch.zeros(config["num_steps"], device=device)
    
    def report_action(self, step):
        """ call report_action and use it directly to step the env"""
        self.obs_buffer[step] = self.next_obs
        self.dones_buffer[step] = torch.tensor(self.next_done, dtype=torch.float32, device=device)

        with torch.no_grad():
            action, logprob, _, value = self.network.get_action_and_value(self.next_obs.unsqueeze(0))
            action = action.squeeze(0)
            logprob = logprob.squeeze(0)
            self.values_buffer[step] = value.squeeze(0)
        
        self.actions_buffer[step] = action
        self.logprobs_buffer[step] = logprob

        return action.item()
    
    def update_after_step(self, step, next_obs_np, reward, done, info):
        self.rewards_buffer[step] = torch.tensor(reward, dtype=torch.float32, device=device)

        self.next_done = 1.0 if done else 0.0
        self.next_obs = torch.tensor(self.flatten_obs(next_obs_np), dtype=torch.float32, device=device)
    
    def train_current_iteration(self, config):
        # Assuming you've computed advantages and returns (same as before):
        with torch.no_grad():
            next_value = self.network.get_value(self.next_obs.unsqueeze(0)).reshape(1)
            self.advantages = torch.zeros_like(self.rewards_buffer, device=device)
            lastgaelam = 0
            for t in reversed(range(config["num_steps"])):
                if t == config["num_steps"] - 1:
                    nextnonterminal = 1.0 - self.next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones_buffer[t + 1]
                    nextvalues = self.values_buffer[t + 1]
                delta = self.rewards_buffer[t] + config["gamma"] * nextvalues * nextnonterminal - self.values_buffer[t]
                self.advantages[t] = lastgaelam = delta + config["gamma"] * config["gae_lambda"] * nextnonterminal * lastgaelam
            self.returns = self.advantages + self.values_buffer

        # Flatten buffers for minibatch training.
        b_obs = self.obs_buffer.reshape((-1,) + self.obs_size)
        b_logprobs = self.logprobs_buffer.reshape(-1)
        b_actions = self.actions_buffer.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values_buffer.reshape(-1)

        b_inds = np.arange(config["batch_size"])
        clipfracs = []
        for epoch in range(config["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, config["batch_size"], config["minibatch_size"]):
                end = start + config["minibatch_size"]
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = self.network.get_action_and_value(b_obs[mb_inds], action=b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config["clip_coef"]).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if config["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config["clip_coef"], 1 + config["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if config["clip_vloss"]:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds],
                                                                -config["clip_coef"], config["clip_coef"])
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config["ent_coef"] * entropy_loss + v_loss * config["vf_coef"]

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config["max_grad_norm"])
                self.optimizer.step()

            if config["target_kl"] is not None and approx_kl > config["target_kl"]:
                break
        
    def save_model(self, save_path=None):
        """
        Saves the current network and optimizer state.
        
        Parameters:
            save_path (str): The path where the checkpoint will be saved. 
                             Defaults to "sub_agent_model.pt" if not provided.
        """
        if save_path is None:
            save_path = "sub_agent_model.pt"
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, save_path=None):
        """
        Loads the network and optimizer state from a checkpoint.
        
        Parameters:
            save_path (str): The path from where to load the checkpoint.
                             Defaults to "sub_agent_model.pt" if not provided.
        """
        if save_path is None:
            save_path = "sub_agent_model.pt"
        checkpoint = torch.load(save_path, map_location=device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model loaded from {save_path}")
        
