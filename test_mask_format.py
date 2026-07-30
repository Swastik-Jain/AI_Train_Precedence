import torch
import numpy as np
from sb3_contrib import MaskablePPO

try:
    model_path = "/Users/sj/Desktop/AI_Train_Precedence/backend/ai/models/Phase3/ppo_L6_25Trains_final.zip"
    model = MaskablePPO.load(model_path, device="cpu")
    obs = np.zeros((1, *model.observation_space.shape))
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    
    dist = model.policy.get_distribution(obs_tensor)
    
    if hasattr(dist, "apply_masking"):
        print("apply_masking is present.")
        # Try passing a numpy array (as returned by env_method)
        if hasattr(model.action_space, 'nvec'):
            mask = np.ones((1, sum(model.action_space.nvec)), dtype=bool)
        else:
            mask = np.ones((1, model.action_space.n), dtype=bool)
            
        print("Raw mask type:", type(mask), mask.shape)
        # Try just dist.apply_masking(mask)
        try:
            dist.apply_masking(mask)
            print("Successfully applied mask as raw numpy array.")
        except Exception as e:
            print("Failed raw:", e)
            
        try:
            # SB3 contrib's action_masks is often a list of arrays for MultiDiscrete
            import collections
            # what if we try what the library does:
            print("Library requires tensor lists or what?")
            
            # Let's inspect the signature
            import inspect
            print("Signature:", inspect.signature(dist.apply_masking))
        except Exception as e:
            pass
            
except Exception as e:
    import traceback
    traceback.print_exc()
