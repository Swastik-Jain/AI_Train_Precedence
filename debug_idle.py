import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.ai.hybrid_connector import _build_env_and_model
from collections import namedtuple

Args = namedtuple('Args', ['load', 'trains'])
args = Args(load='backend/ai/models/hybrid_step3_FINAL.zip', trains=7)

norm_env, raw_env, model = _build_env_and_model(args, 'backend/ai/models', 'backend/ai/logs')

obs = norm_env.reset()
for i in range(1516):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = norm_env.step(action)
    
    env_instance = norm_env.venv.envs[0].unwrapped
    
    max_idle = max([t.get('idle_time', 0) for t in env_instance.trains])
    if i % 100 == 0:
        print(f"Step {i}, max idle_time: {max_idle}, max speed: {max([t.get('speed', 0) for t in env_instance.trains])}")
        for t in env_instance.trains:
            if t.get('idle_time', 0) > 50:
                print(f"  Train {t['id']} idle: {t['idle_time']}, speed: {t['speed']}, pos: {t['position']}, m_acc: {env_instance._movement_acc[env_instance.trains.index(t)]}")
    if dones[0]:
        print(f"Episode done at step {i}")
        break
