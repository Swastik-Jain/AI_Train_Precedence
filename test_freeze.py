import sys
sys.path.append('backend')
import copy
from backend.main import _get_sim_brain

model, env = _get_sim_brain()
try:
    cloned = copy.deepcopy(env)
    print("DEEPCOPY SUCCESS")
except Exception as e:
    print("DEEPCOPY FAILED", e)
