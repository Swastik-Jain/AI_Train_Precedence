import torch
import numpy as np
from sb3_contrib import MaskablePPO

try:
    model_path = "backend/ai/models/Phase3/ppo_L6_25Trains_final.zip"
    model = MaskablePPO.load(model_path, device="cpu")
    obs_shape = model.observation_space.shape
    obs = np.zeros((1, *obs_shape))
    obs_tensor = torch.tensor(obs).float()
    
    # We bypass get_distribution shape checks by just using what we know
    dist = model.policy.get_distribution(obs_tensor)
    
    print("Action dist class:", type(dist))
    print("dir dist:", dir(dist))
    
    if hasattr(dist, "distribution"):
        print("dist.distribution type:", type(dist.distribution))
        if isinstance(dist.distribution, list):
            print("IS A LIST of length", len(dist.distribution))
        else:
            print("NOT A LIST")
            
    # Try the manual method from the code:
    act_list = [0] * sum(model.action_space.nvec) if hasattr(model.action_space, 'nvec') else [0] * model.action_space.n
    if hasattr(model.action_space, 'nvec'):
        act_list = [0] * len(model.action_space.nvec)
    
    action_tensor = torch.tensor([act_list])
    
    probs_list = []
    if hasattr(dist, "distribution") and isinstance(dist.distribution, list):
        print("Using MultiDiscrete list approach")
        for d, a in zip(dist.distribution, act_list):
            p = torch.exp(d.log_prob(torch.tensor(a).to(model.device))).item()
            probs_list.append(p)
        print("Extracted probs:", probs_list[:5])
    else:
        print("No list approach available. How to get marginals?")
        # Let's see if there is action_dims
        if hasattr(dist, "action_dims"):
            print("Has action_dims:", dist.action_dims)
            
except Exception as e:
    import traceback
    traceback.print_exc()
