from copy import deepcopy
import gym
from itertools import count
import math
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import wandb
import time
import random
import numpy as np

from utils import plot_single_frame, make_video, extract_mode_from_path
from sub_agent import sub_agent

class MultiAgent():
    """This is a meta agent that creates and controls several sub agents. 
    If model_others is True, it enables sharing of buffer experience data between agents.
    """

    def __init__(self, config, env, device, training=True, with_expert=None, debug=False):
        # Set batch parameters.
        # config["batch_size"] = config["num_steps"]
        # config["minibatch_size"] = config["batch_size"] // config["num_minibatches"]
        # config["num_iterations"] = config["total_timesteps"] // config["batch_size"]
        self.config = config
        
        run_name = f"{config['domain']}__{config['mode']}__{config['seed']}__{int(time.time())}"
        wandb.init(
            project=config["wandb_project_name"],
            entity=config["wandb_entity"],
            sync_tensorboard=True,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        self.writer = SummaryWriter(f"runs/{run_name}")

        # Seed everything.
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.backends.cudnn.deterministic = config["torch_deterministic"]
        device = torch.device("cuda" if torch.cuda.is_available() and config["cuda"] else "cpu")
        
        self.combined_next_obs = env.reset()
        # Initialize sub-agents.
        self.agents = []
        self.n_agents = env.n_agents
        for idx in range(self.n_agents):
            next_obs_idx = {"image": self.combined_next_obs["image"][idx],
                            "direction": self.combined_next_obs["direction"][idx]}
            self.agents.append(sub_agent(config, next_obs_idx))
        
        self.total_steps = 0
        # Flag for using a “model of others” (used in visualization).
        self.model_others = config.get("model_others", False)
        self.debug = debug

    def get_agent_state(self, state, idx):
        """Helper to extract the state dictionary for a given agent."""
        return {"image": state["image"][idx], "direction": state["direction"][idx]}
        
    def run_one_episode(self, env, episode, log=True, train=True, save_model=True, visualize=False):
        """Perform a fixed-length rollout.

        At each step:
          - Each agent reports an action based on its own observation.
          - The environment is stepped using the list of actions.
          - The sub-agents update their buffers with the new observations, rewards, and done flag.
          - If the wrapper signals episode termination, log the statistics and reset the env.
        """
        # Optionally anneal learning rates.
        if self.config["anneal_lr"]:
            frac = 1.0 - (episode - 1.0) / self.config["num_iterations"]
            lrnow = frac * self.config["learning_rate"]
            for idx in range(self.n_agents):
                self.agents[idx].optimizer.param_groups[0]["lr"] = lrnow

        if visualize:
            viz_data = self.init_visualization_data(env, 0)

        done = False
        rewards = []
        state = self.combined_next_obs

        for step in range(self.config["num_steps"]):
            self.total_steps += 1

            # For each agent, get an action.
            actions = []
            for idx, agent in enumerate(self.agents):
                # (Optionally update agent.next_obs using state if needed)
                action = agent.report_action(step)
                actions.append(action)

            # Step the environment.
            next_state, reward, done, info = env.step(actions)
            rewards.append(reward)

            # Log episode statistics if the wrapper indicates the episode is over.
            if done:
                if "episode" in info:
                    # Log total reward and per-agent rewards.
                    self.writer.add_scalar("charts/episodic_total_return", 
                                           info["episode"]["total_reward"], self.total_steps)
                    for idx, agent_reward in enumerate(info["episode"]["agent_rewards"]):
                        self.writer.add_scalar(f"charts/episodic_agent_{idx}_return", 
                                               agent_reward, self.total_steps)
                    self.writer.add_scalar("charts/episodic_length", 
                                           info["episode"]["length"], self.total_steps)
                next_state = env.reset()
                done = True  # Mark done so that the agents’ buffers know the episode ended
            else:
                done = False

            # For each agent, update its rollout buffers.
            for idx, agent in enumerate(self.agents):
                next_obs_idx = {"image": next_state["image"][idx],
                                "direction": next_state["direction"][idx]}
                # Here we assume reward is an array with one reward per agent.
                agent_reward = reward[idx]
                agent.update_after_step(step, next_obs_idx, agent_reward, done, info)

            state = next_state  # Update state for next step.

            if visualize:
                viz_data = self.add_visualization_data(viz_data, env, state, actions, next_state)

        self.print_terminal_output(episode, np.sum(rewards))
        if save_model:
            self.save_model_checkpoints(episode)

        if visualize:
            viz_data['rewards'] = np.array(rewards)
            return viz_data

    def get_action_predictions(self, step):
        actions = []
        for idx in range(self.n_agents):
            actions.append(self.agents[idx].report_action(step))
        return actions

    def save_model_checkpoints(self, episode):
        if episode % self.config["save_model_episode"] == 0:
            for i in range(self.n_agents):
                self.agents[i].save_model()

    def print_terminal_output(self, episode, total_reward):
        if episode % self.config["print_every"] == 0:
            print('Total steps: {} \t Episode: {} \t Total reward: {}'.format(
                self.total_steps, episode, total_reward))

    def init_visualization_data(self, env, step):
        viz_data = {
            'agents_partial_images': [],
            'actions': [],
            'full_images': [],
            'predicted_actions': None
        }
        viz_data['full_images'].append(env.render('rgb_array'))
        if self.model_others:
            predicted_actions = []
            predicted_actions.append(self.get_action_predictions(step))
            viz_data['predicted_actions'] = predicted_actions
        return viz_data

    def add_visualization_data(self, viz_data, env, state, actions, next_state):
        viz_data['actions'].append(actions)
        viz_data['agents_partial_images'].append(
            [env.get_obs_render(self.get_agent_state(state, i)['image']) for i in range(self.n_agents)]
        )
        viz_data['full_images'].append(env.render('rgb_array'))
        if self.model_others:
            viz_data['predicted_actions'].append(self.get_action_predictions(0))
        return viz_data
        
    def update_models(self):
        for agent in self.agents:
            agent.train_current_iteration(self.config)
    
    def train(self, env):
        for episode in range(self.config["num_iterations"]):
            if episode % self.config["visualize_every"] == 0 and not(episode == 0):
                viz_data = self.run_one_episode(env, episode, visualize=True)
                self.visualize(env, self.config["mode"] + '_training_step' + str(episode), viz_data=viz_data)
            else:
                self.run_one_episode(env, episode)

            self.update_models()

        env.close()
        return

    def visualize(self, env, mode, video_dir='videos', viz_data=None):
        if not viz_data:
            viz_data = self.run_one_episode(env, episode=0, log=False, train=False, save_model=False, visualize=True)
            env.close()

        video_path = os.path.join(video_dir, self.config["experiment_name"], self.config["model_name"])

        if not os.path.exists(video_path):
            os.makedirs(video_path)

        # Get names of actions.
        action_dict = {}
        for act in env.Actions:
            action_dict[act.value] = act.name

        traj_len = len(viz_data['rewards'])
        for t in range(traj_len):
            self.visualize_one_frame(t, viz_data, action_dict, video_path, self.config["model_name"])
            print('Frame {}/{}'.format(t, traj_len))

        make_video(video_path, mode + '_trajectory_video')

    def visualize_one_frame(self, t, viz_data, action_dict, video_path, model_name):
        plot_single_frame(t, 
                          viz_data['full_images'][t], 
                          viz_data['agents_partial_images'][t], 
                          viz_data['actions'][t], 
                          viz_data['rewards'], 
                          action_dict, 
                          video_path, 
                          self.config["model_name"],
                          predicted_actions=viz_data['predicted_actions'])

    def load_models(self, model_path=None):
        for i in range(self.n_agents):
            if model_path is not None:
                self.agents[i].load_model(save_path=model_path + '_agent_' + str(i))
            else:
                self.agents[i].load_model()