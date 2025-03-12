import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np

NUM_DIRECTIONS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiGridActorCriticNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        """
        Initialize Actor-Critic Network

        Args:
            obs_shape (tuple): Tuple containing image shape and direction shape
            n_actions (int): Number of actions
        """
        super(MultiGridActorCriticNetwork, self).__init__()
        
        # Extract image shape (1, 5, 5, 3)
        self.image_shape = obs_shape[0]  
        
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
            nn.Linear(128, n_actions),
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
            
        # Remove extra dimension if present (1, 5, 5, 3) -> (5, 5, 3)
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

class MultiGridNetwork(nn.Module):
    def __init__(self, obs, config, n_actions, n_agents, agent_id):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(MultiGridNetwork, self).__init__()
        self.obs_shape = obs
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.config = config
        self.agent_id = agent_id

        self.image_layers = nn.Sequential(
            nn.Conv2d(3, 32, (self.config.kernel_size, self.config.kernel_size)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, (self.config.kernel_size, self.config.kernel_size)),
            nn.LeakyReLU(),
            nn.Flatten(),  # [B, 64, 1, 1] -> [B, 64]
            nn.Linear(64, 64),  
            nn.LeakyReLU()
            )

        self.direction_layers = nn.Sequential(
            nn.Linear(NUM_DIRECTIONS * self.n_agents, self.config.fc_direction),
            nn.ReLU(),
            )

        #interm = (obs['image'].shape[1]-self.config.kernel_size)+1
        self.head = nn.Sequential(
            nn.Linear(64 + self.config.fc_direction, 192),
            nn.ReLU(),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions),
        )

    def process_image(self, x):
        if len(x.shape) == 3:
            # Add batch dimension
            x = x.unsqueeze(0)
        # Change from (B,H,W,C) to (B,C,W,H) (i.e. RGB channel of dim 3 comes first)
        x = x.permute((0, 3, 1, 2))
        x = x.float()
        return x
            
    def forward(self, obs):
        # process image
        x = torch.tensor(obs['image']).to(device)
        x = self.process_image(x)
        batch_dim = x.shape[0]

        # Run conv layers on image
        image_features = self.image_layers(x)
        image_features = image_features.reshape(batch_dim, -1)

        # Process direction and run direction layers
        dirs = torch.tensor(obs['direction']).to(device)
        if batch_dim == 1:  # 
            dirs = torch.tensor(dirs).unsqueeze(0)
        dirs_onehot = torch.nn.functional.one_hot(dirs.to(torch.int64), num_classes=NUM_DIRECTIONS).reshape((batch_dim, -1)).float()
        dirs_encoding = self.direction_layers(dirs_onehot)

        # Concat
        features = torch.cat([image_features, dirs_encoding], dim=-1)

        # Run head
        return self.head(features)