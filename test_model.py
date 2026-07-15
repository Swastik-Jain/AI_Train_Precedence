import torch
import numpy as np
from stable_baselines3 import PPO

try:
    model = PPO.load("backend/models/ppo_multi_agent")
    print("Model loaded")
    obs = np.zeros((1, model.observation_space.shape[0]))
    obs_tensor = torch.tensor(obs).float()
    dist = model.policy.get_distribution(obs_tensor)
    print("Type of dist:", type(dist))
    print("Has attribute 'distribution':", hasattr(dist, 'distribution'))
    if hasattr(dist, 'distribution'):
        print("dist.distribution:", type(dist.distribution))
except Exception as e:
    print("Error:", e)
