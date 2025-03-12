import gym
# Import needed to trigger env registration
from envs import gym_multigrid
from envs.gym_multigrid import multigrid_envs
import random

# Replace with your actual environment ID
env_id = "MultiGrid-Cluttered-Fixed-15x15"  # Use your environment ID here
env = gym.make(env_id)

print(f"Environment ID: {env_id}")
print("\nAction Space:")
print(f"Type: {type(env.action_space)}")
print(f"Space: {env.action_space}")
if hasattr(env.action_space, 'n'):
    print("fffffffffffff")
    print(f"Number of actions: {env.action_space.n}")

print("\nObservation Space:")
print(f"Type: {type(env.observation_space)}")
print(f"Space: {env.observation_space['image'].shape}")
if hasattr(env.observation_space, 'spaces'):
    print("Observation space components:")
    for key, space in env.observation_space.spaces.items():
        print(f"  {key}: {space}")

print(env.reset()["image"][0].shape)

def random_policy():
    return random.randint(0, 6)

# env.reset()
# while(True):
#     next_obs, reward, done, info = env.step([random_policy(), random_policy(), random_policy()])
#     print(f"next_obs: {next_obs}, reward: {reward}, done: {done}, info: {info}")


# Close the environment
env.close()