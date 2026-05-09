"""
train_manual.py — Curriculum Training for CSMT-Manmad Corridor
MaskablePPO + ForceConstantLR + GhatTokenSystem environment

Curriculum:
  Level 1: 2 trains  (1 UP + 1 DOWN)  → 500k steps   — learn basic bidirectional
  Level 2: 5 trains  mixed direction   → 700k steps   — learn ghat token conflicts
  Level 3: 7 trains  mixed direction   → 700k steps   — learn overtaking + banker
  Level 4: 10 trains mixed direction   → 500k steps   — full complexity

Usage:
  # Fresh start
  python train_manual.py --level 1 --trains 2 --steps 500000

  # Continue from checkpoint
  python train_manual.py --level 2 --trains 5 --steps 700000 \
      --load models/L1_2Trains_Best/best_model.zip

CRITICAL: NEVER load VecNormalize stats across levels — obs shape is fixed
at (10, 23) throughout, but reward distributions shift. Always start fresh
normalization at each level (default behaviour here).
"""

import os
import argparse

os.environ['TORCH_COMPILE_DISABLE'] = '1'

import numpy as np
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    StopTrainingOnRewardThreshold,
)
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from train_env import TrainDispatchEnv

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
LOGS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# These worked well on the toy map. Keep stable — don't tune until Phase 3.
# ─────────────────────────────────────────────────────────────────────────────
FIXED_LR        = 3e-5
CLIP_RANGE      = 0.3
GAMMA           = 0.995
GAE_LAMBDA      = 0.95
N_STEPS         = 2048
BATCH_SIZE      = 256
N_EPOCHS        = 10
ENT_COEF        = 0.01
VF_COEF         = 0.5
MAX_GRAD_NORM   = 0.5

# Number of parallel envs — 8 gives 3-4x speedup over 2
# Requires SubprocVecEnv (spawns separate processes)
N_ENVS_TRAIN    = 8
N_ENVS_EVAL     = 2

# Early stopping reward thresholds per level
# These are conservative — set high enough to not stop prematurely
# Adjust down if training stalls (reward never reaches threshold)
REWARD_THRESHOLDS = {
    1: 5.0,    # 2 trains: basic movement reward
    2: 15.0,   # 5 trains: ghat conflicts resolved
    3: 25.0,   # 7 trains: overtaking working
    4: 35.0,   # 10 trains: full curriculum
}


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class ForceConstantLR(BaseCallback):
    """
    Locks the optimizer LR to a fixed value every rollout.
    SB3 decays LR by default via lr_schedule — this overrides that.
    Solved a convergence bug in the toy map training.
    Keep it. Don't remove it.
    """

    def __init__(self, lr: float):
        super().__init__()
        self.lr = lr

    def _apply_lr(self):
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
            for pg in self.model.policy.optimizer.param_groups:
                pg['lr'] = self.lr

    def _on_training_start(self):
        self._apply_lr()
        print(f"✅ ForceConstantLR: LR locked to {self.lr}")

    def _on_rollout_start(self):
        self._apply_lr()

    def _on_step(self) -> bool:
        return True


class GhatUtilizationLogger(BaseCallback):
    """
    Logs token block utilization every eval_freq steps.
    Tracks how often the ghat mid-line is occupied — key throughput metric.
    """

    def __init__(self, eval_freq: int = 10000):
        super().__init__()
        self.eval_freq    = eval_freq
        self._step_count  = 0
        self._token_occ   = 0   # steps where ghat token was held
        self._total_steps = 0

    def _on_step(self) -> bool:
        self._step_count  += 1
        self._total_steps += 1

        # Sample token status from first env
        try:
            envs = self.training_env.env_method('ghat_token')
        except Exception:
            return True

        if self._step_count >= self.eval_freq:
            util = self._token_occ / max(self._step_count, 1)
            if self.logger:
                self.logger.record('ghat/token_utilization', util)
            self._step_count = 0
            self._token_occ  = 0

        return True


# ─────────────────────────────────────────────────────────────────────────────
# ACTION MASK FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_action_mask()


# ─────────────────────────────────────────────────────────────────────────────
# ENV FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_env_fn(num_trains: int):
    """Returns a callable that creates one masked env with the right difficulty."""
    def _make():
        env = TrainDispatchEnv()
        env.set_difficulty(num_trains)
        return ActionMasker(env, mask_fn)
    return _make


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train_manual(
    level:          str,
    num_trains:     int,
    total_steps:    int,
    load_path:      str   = None,
    learning_rate:  float = FIXED_LR,
    no_early_stop:  bool  = False,
    ent_coef:       float = ENT_COEF,
    chaos:          bool  = False,
):
    print(f"\n{'='*60}")
    print(f"🚂 TRAINING: Level {level} | {num_trains} Trains | {total_steps} Steps")
    print(f"   LR={learning_rate}  ent_coef={ent_coef}  chaos={chaos}")
    print(f"{'='*60}\n")

    # ── Create vectorised environments ────────────────────────────────────
    # SubprocVecEnv spawns separate processes — 3-4x faster than DummyVecEnv
    # Requires if __name__ == '__main__' guard (handled at bottom of file)
    train_env_fns = [make_env_fn(num_trains) for _ in range(N_ENVS_TRAIN)]
    eval_env_fns  = [make_env_fn(num_trains) for _ in range(N_ENVS_EVAL)]

    raw_train_env = make_vec_env(
        make_env_fn(num_trains),
        n_envs=N_ENVS_TRAIN,
        vec_env_cls=SubprocVecEnv,
    )
    raw_eval_env = make_vec_env(
        make_env_fn(num_trains),
        n_envs=N_ENVS_EVAL,
        vec_env_cls=SubprocVecEnv,
    )

    # ── VecNormalize ──────────────────────────────────────────────────────
    # CRITICAL: Never load VecNormalize stats from a previous level.
    # Obs shape is stable at (10, 23) across all levels, but reward
    # distributions shift significantly. Always start fresh normalization.
    #
    # stats_path is saved at end of training but NOT loaded on resume —
    # loading across levels causes reward normalizer drift.

    stats_path = os.path.join(
        MODELS_DIR, f"vec_normalize_L{level}_{num_trains}Trains.pkl"
    )

    train_env = VecNormalize(
        raw_train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    eval_env = VecNormalize(
        raw_eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
        training=False,
    )

    # ── Model: load or create ─────────────────────────────────────────────
    if load_path and os.path.exists(load_path):
        print(f"🔄 Loading model from: {load_path}")
        print(f"⚠️  Starting FRESH VecNormalize (new level — don't reuse old stats)")

        model = MaskablePPO.load(
            load_path,
            env=train_env,
            tensorboard_log=LOGS_DIR,
            custom_objects={
                'learning_rate': learning_rate,
                'lr_schedule':   lambda _: learning_rate,
                'clip_range':    CLIP_RANGE,
                'ent_coef':      ent_coef,
            },
        )
        print(f"✅ Model loaded | LR overridden → {learning_rate}")

    else:
        print("✨ Creating NEW MaskablePPO model")
        model = MaskablePPO(
            'MlpPolicy',
            train_env,
            verbose=1,
            tensorboard_log=LOGS_DIR,
            learning_rate=learning_rate,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ent_coef,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            device='auto',
        )

    # ── Enable chaos mode if requested ───────────────────────────────────
    if chaos:
        print("🐒 Chaos mode enabled on training envs")
        train_env.env_method('set_chaos_mode', True)

    # ── Callbacks ─────────────────────────────────────────────────────────
    threshold = REWARD_THRESHOLDS.get(int(level) if str(level).isdigit() else 1, 10.0)

    stop_callback = None
    if not no_early_stop:
        print(f"⏳ Early stopping at reward > {threshold}")
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=threshold, verbose=1
        )
    else:
        print("⏳ Early stopping DISABLED — full step budget")

    eval_callback = MaskableEvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=max(10000 // N_ENVS_TRAIN, 1000),
        n_eval_episodes=20,
        best_model_save_path=os.path.join(
            MODELS_DIR, f"L{level}_{num_trains}Trains_Best"
        ),
        log_path=os.path.join(LOGS_DIR, f"Eval_L{level}_{num_trains}Trains"),
        deterministic=True,
        render=False,
    )

    lr_callback = ForceConstantLR(learning_rate)

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n🚀 Training for {total_steps:,} steps on {N_ENVS_TRAIN} parallel envs...")
    print(f"   Effective throughput: ~{N_ENVS_TRAIN * N_STEPS:,} steps/rollout\n")

    model.learn(
        total_timesteps=total_steps,
        reset_num_timesteps=True,
        tb_log_name=f"PPO_L{level}_{num_trains}Trains",
        callback=[eval_callback, lr_callback],
    )

    # ── Save ──────────────────────────────────────────────────────────────
    save_name = f"ppo_L{level}_{num_trains}Trains_final.zip"
    save_path = os.path.join(MODELS_DIR, save_name)
    model.save(save_path)
    train_env.save(stats_path)

    print(f"\n✅ Training complete")
    print(f"   Model  → {save_path}")
    print(f"   Stats  → {stats_path}")
    print(f"\n   To continue to next level:")

    next_level  = int(level) + 1 if str(level).isdigit() else 2
    next_trains = {1: 5, 2: 7, 3: 10, 4: 10}.get(int(level), num_trains + 2)
    best_path   = os.path.join(
        MODELS_DIR, f"L{level}_{num_trains}Trains_Best", "best_model.zip"
    )
    print(f"   python train_manual.py "
          f"--level {next_level} "
          f"--trains {next_trains} "
          f"--steps 700000 "
          f"--load {best_path}")

    # Clean up subprocesses
    train_env.close()
    eval_env.close()

    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Required for SubprocVecEnv on Linux/macOS
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description='Curriculum Training — CSMT-Manmad Train Dispatcher'
    )
    parser.add_argument('--level',         type=str,   required=True,
                        help='Curriculum level (1-4)')
    parser.add_argument('--trains',        type=int,   required=True,
                        help='Number of trains in this level')
    parser.add_argument('--steps',         type=int,   default=500000,
                        help='Training timesteps (default: 500000)')
    parser.add_argument('--load',          type=str,   default=None,
                        help='Path to model checkpoint to resume from')
    parser.add_argument('--lr',            type=float, default=FIXED_LR,
                        help=f'Learning rate (default: {FIXED_LR})')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='Disable early stopping on reward threshold')
    parser.add_argument('--ent-coef',      type=float, default=ENT_COEF,
                        help=f'Entropy coefficient (default: {ENT_COEF})')
    parser.add_argument('--chaos',         action='store_true',
                        help='Enable chaos monkey (delayed starts, speed snags)')

    args = parser.parse_args()

    train_manual(
        level=args.level,
        num_trains=args.trains,
        total_steps=args.steps,
        load_path=args.load,
        learning_rate=args.lr,
        no_early_stop=args.no_early_stop,
        ent_coef=args.ent_coef,
        chaos=args.chaos,
    )