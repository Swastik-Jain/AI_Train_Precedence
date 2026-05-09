"""
run_incremental_scaling.py
==========================
Phase 3.5: The 7-Train Leap

This script bridges the gap between the 5-train optimized model and the 
new 7-train complexity. It generates a 7-train expert schedule and uses 
Behavioral Cloning (BC) for a quick "state space adjustment" before 
full PPO training begins.
"""

import os
import sys
import subprocess
import logging
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym
import numpy as np

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_env import TrainDispatchEnv
from hybrid_connector import pretrain_from_expert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScalingScript")

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_action_mask()

def run_scaling():
    # 1. Paths
    LOAD_PATH = "models/ppo_phase3_L2_Manual.zip" # The L2-5T best model
    EXPERT_PATH = "expert_actions_7T.json"
    STATS_PATH = "models/vec_normalize_L2.pkl"
    OUTPUT_MODEL = "models/ppo_phase3_L2_7T_Warmed.zip"
    
    # 2. Generate 7-Train Expert Data
    logger.info("🚀 Generating 7-Train expert trajectories...")
    subprocess.run([
        "python", "generate_golden_trajectories.py", 
        "--trains", "7", 
        "--out", EXPERT_PATH
    ], check=True)

    # 3. Initialise Environment (7 Trains)
    def make_env():
        env = TrainDispatchEnv()
        env.set_difficulty(7)
        return ActionMasker(env, mask_fn)

    raw_env = make_vec_env(make_env, n_envs=1)
    
    # IMPORTANT: Delete stale 5-train stats to avoid normalisation noise
    if os.path.exists(STATS_PATH):
        logger.info(f"🗑️  Removing stale normalization stats: {STATS_PATH}")
        os.remove(STATS_PATH)
        
    env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # 4. Load Model
    if not os.path.exists(LOAD_PATH):
        logger.error(f"❌ Base model {LOAD_PATH} not found! Run Level 2 training first.")
        return

    logger.info(f"🔄 Loading base model from {LOAD_PATH}...")
    model = MaskablePPO.load(LOAD_PATH, env=env)

    # 5. Execute Behavioral Cloning (20,000 steps equivalent)
    # Each ep is ~120 mins. 7 trains * 120 steps = ~840 samples per ep.
    # 20,000 steps / 840 = ~24 epochs of the expert data.
    logger.info("🎓 Running Incremental BC Warm-up (20,000 global steps)...")
    model = pretrain_from_expert(
        model=model,
        env=raw_env.envs[0].unwrapped, # Unwrap to access .trains attribute
        expert_data_path=EXPERT_PATH,
        bc_epochs=25,
        batch_size=256,
        lr=3e-5
    )

    # 6. Final Save
    model.save(OUTPUT_MODEL)
    env.save(STATS_PATH)
    logger.info(f"✅ 7-Train Warmed Model saved to: {OUTPUT_MODEL}")
    logger.info(f"👓 Normalization stats reset and saved to: {STATS_PATH}")

if __name__ == "__main__":
    run_scaling()
