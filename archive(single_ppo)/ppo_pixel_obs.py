import os
import random
import time
from dataclasses import dataclass

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import tyro
from torch.utils.tensorboard import SummaryWriter

from envs import gym_multigrid
from envs.gym_multigrid import multigrid_envs

NUM_DIRECTIONS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiGridActorCriticNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        """
        Initialize Actor-Critic Network

        Args:
            observation_space (gym.spaces.Dict): Dictionary observation space
            action_space (gym.spaces.Discrete): Discrete action space
        """
        super(MultiGridActorCriticNetwork, self).__init__()
        
        # Extract shapes from observation space
        self.image_shape = observation_space.spaces['image'].shape  # (1, 5, 5, 3)
        self.direction_dim = observation_space.spaces['direction'].shape[0]  # 1
        
        # Number of actions
        self.n_actions = action_space.n  # 7 actions (0-6)
        
        # Image processing layers
        self.image_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        
        # Calculate the flattened size after convolutions
        # For a 5x5 image with the current conv layers (same padding), 
        # the spatial dimensions remain 5x5, so: 32 * 5 * 5 = 800
        conv_output_size = 32 * 5 * 5
        
        self.image_fc = nn.Sequential(
            nn.Linear(conv_output_size, 64),
            nn.ReLU(),
        )
        
        # Direction processing layers
        self.direction_layers = nn.Sequential(
            nn.Linear(NUM_DIRECTIONS, 64),
            nn.ReLU(),
        )
        
        # Combined feature size
        self.feature_size = 64 + 64
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_actions),
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def process_obs(self, obs):
        """
        Process observation dictionary into tensors
        
        Args:
            obs (dict): Observation dictionary with 'image' and 'direction' keys
            
        Returns:
            tuple: Processed image tensor and direction tensor
        """
        # Process image: (B, 1, 5, 5, 3) -> (B, 3, 5, 5)
        if isinstance(obs['image'], np.ndarray):
            image = torch.FloatTensor(obs['image']).to(next(self.parameters()).device)
        else:
            image = obs['image'].float()
            
        # Remove extra dimension if present (B, 1, 5, 5, 3) -> (B, 5, 5, 3)
        if image.dim() == 5:  # (B, 1, H, W, C)
            image = image.squeeze(1)
        
        # Permute from (B, H, W, C) to (B, C, H, W) for CNN
        image = image.permute(0, 3, 1, 2)
        
        # Process direction
        if isinstance(obs['direction'], np.ndarray):
            direction = torch.LongTensor(obs['direction']).to(next(self.parameters()).device)
        else:
            direction = obs['direction'].long()
            
        # Ensure direction has the right shape
        if direction.dim() == 2 and direction.size(1) == 1:
            direction = direction.squeeze(1)  # (B, 1) -> (B)
        
        return image, direction
        
    def get_action_and_value(self, obs, action=None):
        """
        Forward pass to get action distribution, entropy, and value
        
        Args:
            obs (dict): Observation dictionary
            action (tensor, optional): Actions to compute log probabilities
            
        Returns:
            tuple: (action, log_prob, entropy, value)
        """
        image, direction = self.process_obs(obs)
        
        # Process image through convolutions
        image_features = self.image_layers(image)
        image_features = self.image_fc(image_features)
        
        # Convert direction to one-hot and get features
        dirs_onehot = F.one_hot(direction, num_classes=NUM_DIRECTIONS).float()
        dirs_features = self.direction_layers(dirs_onehot)
        
        # Combine features
        features = torch.cat([image_features, dirs_features], dim=1)
        
        # Get action logits and value
        action_logits = self.actor(features)
        value = self.critic(features)
        
        # Create action distribution
        probs = Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), value
        
    def get_value(self, obs):
        """
        Get state value
        
        Args:
            obs (dict): Observation dictionary
            
        Returns:
            tensor: Value of the state
        """
        image, direction = self.process_obs(obs)
        
        # Process image through convolutions
        image_features = self.image_layers(image)
        image_features = self.image_fc(image_features)
        
        # Convert direction to one-hot and get features
        dirs_onehot = F.one_hot(direction, num_classes=NUM_DIRECTIONS).float()
        dirs_features = self.direction_layers(dirs_onehot)
        
        # Combine features
        features = torch.cat([image_features, dirs_features], dim=1)
        
        # Get value
        value = self.critic(features)
        return value


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

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "MultiGrid"
    """the wandb's project name"""
    wandb_entity: str = "rl-power"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "MultiGrid-Cluttered-Fixed-Single-v0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def make_env(env_id, idx, capture_video, run_name, seed=0):
    def thunk():
        # Set the seed manually before creating the environment
        random.seed(seed + idx)
        np.random.seed(seed + idx)
        
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # Convert Box action space to Discrete
        env = MultiGridBoxToDiscreteWrapper(env)
        
        # Manually call env.seed() as some environments may still use this method
        try:
            env.seed(seed + idx)
        except (AttributeError, TypeError):
            pass  # If env.seed() is not available or doesn't take arguments, just pass
        
        return env

    return thunk

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.seed) for i in range(args.num_envs)],
    )
    
    # Make sure we're working with a Discrete action space after wrapping
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Create the agent with our MultiGridActorCriticNetwork
    agent = MultiGridActorCriticNetwork(envs.single_observation_space, envs.single_action_space).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    # Helper function to create empty observations
    def create_empty_obs(num_steps, num_envs, observation_space):
        empty_obs = {}
        for key, space in observation_space.spaces.items():
            empty_obs[key] = torch.zeros((num_steps, num_envs) + space.shape, dtype=torch.float32).to(device)
        return empty_obs
    
    # Create storage for observations as dictionaries
    obs = create_empty_obs(args.num_steps, args.num_envs, envs.single_observation_space)
    
    # Storage for actions, values, etc.
    actions = torch.zeros((args.num_steps, args.num_envs)).long().to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    
    # Reset environments and get initial observations
    next_obs = envs.reset()
    
    # Convert numpy observations to PyTorch tensors
    next_obs_tensor = {}
    for key in next_obs.keys():
        next_obs_tensor[key] = torch.tensor(next_obs[key]).to(device)
    
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            
            # Store current observations and dones
            for key in obs.keys():
                obs[key][step] = next_obs_tensor[key]
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs_tensor)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            # Execute actions in environments
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            
            # Store rewards and update next_done
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_done = torch.tensor(done).to(device)
            
            # Handle episode metrics
            for idx, d in enumerate(done):
                if d:
                    print(f"Episode finished with reward: {info[idx].get('episode', {}).get('r', 0)}")
                    if 'episode' in info[idx]:
                        writer.add_scalar("charts/episodic_return", info[idx]['episode']['r'], global_step)
                        writer.add_scalar("charts/episodic_length", info[idx]['episode']['l'], global_step)
            
            # Convert next_obs to tensor format
            for key in next_obs.keys():
                next_obs_tensor[key] = torch.tensor(next_obs[key]).to(device)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs_tensor).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch - we need to handle the dict observations differently
        b_obs = {}
        for key in obs.keys():
            b_obs[key] = obs[key].reshape((-1,) + envs.single_observation_space.spaces[key].shape)
            
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                # Extract mini-batch observations
                mb_obs = {}
                for key in b_obs.keys():
                    mb_obs[key] = b_obs[key][mb_inds]
                
                # Get new action distributions and values
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    mb_obs, b_actions[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
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

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()