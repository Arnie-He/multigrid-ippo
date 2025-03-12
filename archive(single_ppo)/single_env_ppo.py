import os
import random
import time
from dataclasses import dataclass

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

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "MultiGrid"
    wandb_entity: str = "rl-power"
    capture_video: bool = False

    # Algorithm specific arguments
    env_id: str = "MultiGrid-Cluttered-Fixed-Single-v0"
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    # For non-vectorized version, we set num_envs to 1.
    num_envs: int = 1  
    num_steps: int = 128
    anneal_lr: bool = True
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
    target_kl: float = None

    # These will be computed at runtime
    batch_size: int = 0  
    minibatch_size: int = 0
    num_iterations: int = 0

def make_env(env_id, idx, capture_video, run_name, seed=0):
    def thunk():
        random.seed(seed + idx)
        np.random.seed(seed + idx)
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        try:
            env.seed(seed + idx)
        except (AttributeError, TypeError):
            pass
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self):
        super().__init__()

        self.NUM_DIRECTIONS = 4
        fc_direction = 8
        ACTION_DIM = 7
        
        self.image_layers = nn.Sequential(
            nn.Conv2d(3, 8, (3, 3)),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, (3, 3)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(16, 16),  
            nn.LeakyReLU()
        )
        self.direction_layers = nn.Sequential(
            nn.Linear(self.NUM_DIRECTIONS, fc_direction),
            nn.ReLU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(16 + fc_direction, 32),
            nn.ReLU(),
            nn.Linear(32, ACTION_DIM),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(16 + fc_direction, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def process_image(self, x):
        if len(x.shape) == 3:
            # Add batch dimension
            x = x.unsqueeze(0)
        if (len(x.shape) == 5 and x.shape[1] ==1):
            x = x.squeeze(1)
        # Change from (B,H,W,C) to (B,C,W,H) (i.e. RGB channel of dim 3 comes first)
        x = x.permute((0, 3, 1, 2))
        x = x.float()
        return x
    
    def get_value(self, image, dir):
        image = torch.tensor(image).to(device)
        image = self.process_image(image)
        batch_dim = image.shape[0]
        image_features = self.image_layers(image)
        image_features = image_features.reshape(batch_dim, -1)
        dirs = torch.tensor(dir).to(device)
        if batch_dim == 1:  # 
            dirs = torch.tensor(dirs).unsqueeze(0)
        dirs_onehot = torch.nn.functional.one_hot(dirs.to(torch.int64), num_classes=self.NUM_DIRECTIONS).reshape((batch_dim, -1)).float()
        dirs_encoding = self.direction_layers(dirs_onehot)
        # Concat
        features = torch.cat([image_features, dirs_encoding], dim=-1)
        return self.critic_head(features)

    def get_action_and_value(self, image, dir, action=None):
        x = torch.tensor(image).to(device)
        x = self.process_image(x)
        batch_dim = x.shape[0]
        image_features = self.image_layers(x)
        image_features = image_features.reshape(batch_dim, -1)
        dirs = torch.tensor(dir).to(device)
        if batch_dim == 1:  # 
            dirs = torch.tensor(dirs).unsqueeze(0)
        dirs_onehot = torch.nn.functional.one_hot(dirs.to(torch.int64), num_classes=self.NUM_DIRECTIONS).reshape((batch_dim, -1)).float()
        dirs_encoding = self.direction_layers(dirs_onehot)
        # Concat
        features = torch.cat([image_features, dirs_encoding], dim=-1)
        logits = self.actor_head(features)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic_head(features)

if __name__ == "__main__":
    args = tyro.cli(Args)
    # For non-vectorized training, our batch size is just num_steps.
    args.batch_size = args.num_envs * args.num_steps  # here num_envs is 1
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create a single environment (non-vectorized)
    env = make_env(args.env_id, 0, args.capture_video, run_name, args.seed)()
    # Ensure we use the correct observation and action space from the wrapped env.
    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Initialize separate buffers for images and directions.
    image_buffer = []
    direction_buffer = []
    actions_buffer = torch.zeros((args.num_steps,), dtype=torch.long, device=device)
    logprobs_buffer = torch.zeros(args.num_steps, device=device)
    rewards_buffer = torch.zeros(args.num_steps, device=device)
    dones_buffer = torch.zeros(args.num_steps, device=device)
    values_buffer = torch.zeros(args.num_steps, device=device)

    global_step = 0
    start_time = time.time()

    # Reset the environment. This returns a dictionary with keys "image" and "direction".
    next_obs = env.reset()
    next_done = 0.0

    for iteration in range(1, args.num_iterations + 1):
        # Collect rollout of fixed length (num_steps) from a single environment.
        for step in range(args.num_steps):
            # Append image and direction separately.
            image_buffer.append(next_obs["image"])
            direction_buffer.append(next_obs["direction"])
            
            dones_buffer[step] = torch.tensor(next_done, dtype=torch.float32, device=device)
            
            # Use the current observation (as a dict) for the agent.
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs["image"], next_obs["direction"])
                values_buffer[step] = value.squeeze(0)
            actions_buffer[step] = action
            logprobs_buffer[step] = logprob

            next_obs_np, reward, done, info = env.step(action.item())
            rewards_buffer[step] = torch.tensor(reward, dtype=torch.float32, device=device)

            if done:
                if "episode" in info:
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                next_obs_np = env.reset()
                next_done = 1.0
            else:
                next_done = 0.0
            next_obs = next_obs_np
            global_step += 1

        # Convert the separate buffers into tensors.
        b_images = torch.stack([torch.tensor(img, dtype=torch.float32, device=device) for img in image_buffer])
        b_directions = torch.stack([torch.tensor(d, dtype=torch.float32, device=device) for d in direction_buffer])
        
        # Assuming you've computed advantages and returns (same as before):
        with torch.no_grad():
            next_value = agent.get_value(next_obs["image"], next_obs["direction"]).reshape(1)
            advantages = torch.zeros_like(rewards_buffer, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t + 1]
                    nextvalues = values_buffer[t + 1]
                delta = rewards_buffer[t] + args.gamma * nextvalues * nextnonterminal - values_buffer[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buffer

        # Flatten buffers for minibatch training.
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buffer.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_images[mb_inds], b_directions[mb_inds], action=b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds],
                                                                -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    env.close()
    writer.close()

