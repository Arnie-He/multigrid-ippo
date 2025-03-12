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
    def __init__(self, env):
        super().__init__()
        # get flattened observation size
        image_shape = env.observation_space["image"].shape
        direction_shape = env.observation_space["direction"].shape
        flattened_image_size = np.prod(image_shape)
        flattened_direction_size = np.prod(direction_shape)
        obs_size = flattened_image_size + flattened_direction_size
        
        self.observation_size = (obs_size, )

        ACTION_DIM = 7
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, ACTION_DIM), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def get_flattened_obs(self, obs):
        image = np.array(obs["image"])
        direction = np.array(obs["direction"])
        flat_image = image.reshape(-1).astype(np.float32)
        flat_direction = direction.reshape(-1).astype(np.float32)
        return np.concatenate([flat_image, flat_direction])

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
    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Rollout buffers for a single environment.
    obs_buffer = torch.zeros((args.num_steps, ) + agent.observation_size, device=device)
    actions_buffer = torch.zeros((args.num_steps,), dtype=torch.long, device=device)
    logprobs_buffer = torch.zeros(args.num_steps, device=device)
    rewards_buffer = torch.zeros(args.num_steps, device=device)
    dones_buffer = torch.zeros(args.num_steps, device=device)
    values_buffer = torch.zeros(args.num_steps, device=device)

    global_step = 0
    start_time = time.time()

    # Initialize the environment.
    next_obs = agent.get_flattened_obs(env.reset())  # numpy array
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.tensor(0.0, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Collect rollout of fixed length (num_steps) from a single environment.
        for step in range(args.num_steps):
            obs_buffer[step] = next_obs
            dones_buffer[step] = next_done
            with torch.no_grad():
                # Unsqueeze to add batch dimension.
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                action = action.squeeze(0)
                logprob = logprob.squeeze(0)
                value = value.squeeze(0)
                values_buffer[step] = value
            actions_buffer[step] = action
            logprobs_buffer[step] = logprob

            # Step the environment with the chosen action.
            next_obs_np, reward, done, info = env.step(action.item())
            next_obs_np = agent.get_flattened_obs(next_obs_np)
            rewards_buffer[step] = torch.tensor(reward, dtype=torch.float32, device=device)

            # If the episode finished, record metrics and reset.
            if done:
                if "episode" in info:
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                next_obs_np = env.reset()
                next_obs_np = agent.get_flattened_obs(next_obs_np)
                next_done = torch.tensor(1.0, device=device)
            else:
                next_done = torch.tensor(0.0, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            global_step += 1

        # Bootstrap the value for the last observation.
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).reshape(1)
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

        # Flatten the rollout buffers.
        b_obs = obs_buffer.reshape((-1,) + agent.observation_size)
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buffer.reshape(-1)

        # PPO optimization.
        b_inds = np.arange(args.batch_size)  # Here batch_size == num_steps
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # When passing actions, we supply them so that log probabilities are recomputed.
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], action=b_actions[mb_inds])
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
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
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
