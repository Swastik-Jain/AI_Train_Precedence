import torch
import numpy as np
from sb3_contrib import MaskablePPO

try:
    model_path = "backend/ai/models/Phase3/ppo_L6_25Trains_final.zip"
    model = MaskablePPO.load(model_path, device="cpu")
    obs = np.zeros((1, model.observation_space.shape[0]))
    obs_tensor = torch.tensor(obs).float()
    
    # Bypass get_distribution because we need masks. Let's just use action_dist
    features = model.policy.extract_features(obs_tensor)
    latent_pi = model.policy.mlp_extractor.forward_actor(features)
    action_logits = model.policy.action_net(latent_pi)
    
    dist = model.policy.action_dist.proba_distribution(action_logits=action_logits)
    
    act_list = [0] * len(dist.action_dims)
    action_tensor = torch.tensor([act_list])
    
    print("distributions:", getattr(dist, "distributions", None))
    if hasattr(dist, "distributions") and isinstance(dist.distributions, list):
        print("dist.distributions is a list of length:", len(dist.distributions))
        probs_list = []
        for d, a in zip(dist.distributions, act_list):
            p = torch.exp(d.log_prob(torch.tensor(a).to(model.device))).item()
            probs_list.append(p)
        print("probs_list len:", len(probs_list))
        print("probs_list:", probs_list[:5])
        
except Exception as e:
    import traceback
    traceback.print_exc()
