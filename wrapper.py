import time
from collections import deque
import numpy as np
import gym

class RecordMultiAgentEpisodeStatistics(gym.Wrapper):
    """
    A gym wrapper that records episode statistics for multi-agent environments.
    
    It tracks:
      - The per-agent episode rewards.
      - The total episode reward (sum over agents).
      - The episode length.
      - The elapsed time.
    
    When an episode ends, it adds an "episode" key to the info dict with these statistics.
    """
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.n_agents = getattr(env, "n_agents", 1)
        self.t0 = time.perf_counter()
        self.episode_count = 0
        self.agent_episode_returns = np.zeros(self.n_agents, dtype=np.float32)
        self.episode_length = 0
        self.total_return_queue = deque(maxlen=deque_size)
        self.agent_return_queues = [deque(maxlen=deque_size) for _ in range(self.n_agents)]
    
    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.agent_episode_returns = np.zeros(self.n_agents, dtype=np.float32)
        self.episode_length = 0
        return observation

    def step(self, action):
        observation, rewards, done, info = super().step(action)
        rewards = np.array(rewards)  # ensure rewards is an array
        self.agent_episode_returns += rewards
        self.episode_length += 1

        if done:
            total_reward = np.sum(self.agent_episode_returns)
            episode_info = {
                "total_reward": total_reward,
                "agent_rewards": self.agent_episode_returns.copy().tolist(),
                "length": self.episode_length,
                "time": round(time.perf_counter() - self.t0, 6)
            }
            # Update info with episode statistics.
            if isinstance(info, dict):
                info = info.copy()
                info["episode"] = episode_info
            elif isinstance(info, list):
                info = [dict(episode=episode_info) if x is None else x for x in info]
                for i in range(len(info)):
                    if isinstance(info[i], dict):
                        info[i]["episode"] = episode_info
                    else:
                        info[i] = {"episode": episode_info}

            self.total_return_queue.append(total_reward)
            for i in range(self.n_agents):
                self.agent_return_queues[i].append(self.agent_episode_returns[i])
            self.episode_count += 1

            # Reset counters for the next episode.
            self.agent_episode_returns = np.zeros(self.n_agents, dtype=np.float32)
            self.episode_length = 0

        return observation, rewards, done, info