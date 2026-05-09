import torch
from sb3_contrib import MaskablePPO
import gymnasium as gym
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env):
    return env.get_action_mask()

# Dummy env
class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10, 10))
        self.action_space = gym.spaces.MultiDiscrete([3]*10)
    def reset(self, seed=None): return self.observation_space.sample(), {}
    def step(self, action): return self.observation_space.sample(), 0, False, False, {}
    def get_action_mask(self): return [True]*30

env = ActionMasker(DummyEnv(), mask_fn)
model = MaskablePPO("MlpPolicy", env, verbose=0)

# Test extraction
obs = torch.randn(1, 10, 10)
try:
    features = model.policy.extract_features(obs)
    print(f"Features shape: {features.shape}")
    latent_pi, latent_vf = model.policy.mlp_extractor(features)
    print(f"Latent PI shape: {latent_pi.shape}")
    
    action_logits = model.policy.action_net(latent_pi)
    print(f"Action Logits shape: {action_logits.shape}")
except Exception as e:
    print(f"Failed: {e}")
