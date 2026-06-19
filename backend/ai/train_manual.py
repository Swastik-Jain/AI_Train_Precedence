"""
train_manual.py — Curriculum Training for CSMT-Manmad Corridor
MaskablePPO + ForceConstantLR + GhatTokenSystem environment

Curriculum:
  Level 1: 2 trains  (1 UP + 1 DOWN)  → 500k steps   — learn basic bidirectional
  Level 2: 5 trains  mixed direction   → 700k steps   — learn ghat token conflicts
  Level 3: 7 trains  mixed direction   → 700k steps   — learn overtaking + banker
  Level 4: 10 trains mixed direction   → 700k steps   — full 10-train complexity
  Level 5: 15 trains mixed direction   → 1M  steps    — Phase 3 saturation target

Usage:
  # Fresh start
  python train_manual.py --level 1 --trains 2 --steps 500000

  # Continue from checkpoint
  python train_manual.py --level 2 --trains 5 --steps 700000 \
      --load models/L1_2Trains_Best/best_model.zip

CRITICAL: NEVER load VecNormalize stats across levels — obs shape is now
(10, 24) throughout Phase 3 (added required_speed_norm urgency feature).
Always start fresh normalization at each level (default behaviour here).
"""

import os
import sys
import argparse
import math

os.environ['TORCH_COMPILE_DISABLE'] = '1'

# Ensure backend/ is on the path so train_env.py is always importable
# regardless of the working directory the script is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from or_tools.feasibility_shield import FeasibilityShield

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "Phase3")
LOGS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs",   "Phase3")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS — Tuned for fresh bidirectional training from scratch
# ─────────────────────────────────────────────────────────────────────────────
PEAK_LR         = 3e-4   # default (overridden per level below)
LR_MIN          = 1e-5   # cosine decay floor
LR_WARMUP_FRAC  = 0.05   # first 5% = linear warm-up
CLIP_RANGE      = 0.15   # default (overridden per level)
GAMMA           = 0.99   # default (overridden per level)
GAE_LAMBDA      = 0.92   # default (overridden per level)
N_STEPS         = 4096   # shorter rollouts = more frequent updates
BATCH_SIZE      = 1024   # consistent with N_STEPS
N_EPOCHS        = 8      # default (overridden per level)
ENT_COEF        = 0.05   # default (overridden per level)
VF_COEF         = 0.5
MAX_GRAD_NORM   = 0.5

# Per-level hyperparameters
# Keys: curriculum level integer (1–6)
# Each level has its own exploration/exploitation balance:
#   Early levels: high entropy, short gamma, fast LR — wide exploration needed
#   Late levels:  low entropy, long gamma, slow LR  — fine-tune precise decisions
PER_LEVEL_HPARAMS = {
    1: dict(ent_coef=0.050, gamma=0.990, gae_lambda=0.92, n_epochs=8,  peak_lr=3.0e-4, clip_range=0.20),
    2: dict(ent_coef=0.040, gamma=0.992, gae_lambda=0.93, n_epochs=8,  peak_lr=2.5e-4, clip_range=0.18),
    3: dict(ent_coef=0.030, gamma=0.993, gae_lambda=0.94, n_epochs=10, peak_lr=2.0e-4, clip_range=0.15),
    4: dict(ent_coef=0.020, gamma=0.995, gae_lambda=0.95, n_epochs=10, peak_lr=1.5e-4, clip_range=0.15),
    5: dict(ent_coef=0.015, gamma=0.996, gae_lambda=0.95, n_epochs=12, peak_lr=1.0e-4, clip_range=0.12),
    6: dict(ent_coef=0.010, gamma=0.997, gae_lambda=0.95, n_epochs=12, peak_lr=8.0e-5, clip_range=0.10),
}

# Number of parallel envs
N_ENVS_TRAIN    = 8
N_ENVS_EVAL     = 2

# Early stopping reward thresholds per level
REWARD_THRESHOLDS = {
    1: 5.0,
    2: 15.0,
    3: 25.0,
    4: 35.0,
    5: 50.0,
    6: 65.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineDecayLR(BaseCallback):
    """
    Learning rate schedule: linear warm-up → cosine decay.

    Phase 1 (0 → warmup_steps): LR rises linearly from ~0 to lr_peak.
      Gives the fresh model stable early gradients before full-speed updates.

    Phase 2 (warmup_steps → total_steps): LR decays via cosine annealing
      from lr_peak down to lr_min.  Provides smooth convergence without a
      hard LR cliff that caused oscillations in toy-map training.

    Replaces the old ForceConstantLR which hard-locked LR and prevented
    the late-training refinement needed for bidirectional conflict learning.
    """

    def __init__(
        self,
        total_steps:   int,
        lr_peak:       float = PEAK_LR,
        lr_min:        float = LR_MIN,
        warmup_frac:   float = LR_WARMUP_FRAC,
    ):
        super().__init__()
        self.total_steps   = total_steps
        self.lr_peak       = lr_peak
        self.lr_min        = lr_min
        self.warmup_steps  = max(1, int(total_steps * warmup_frac))

    def _get_lr(self) -> float:
        t = self.num_timesteps
        if t < self.warmup_steps:
            # Linear warm-up
            return self.lr_min + (self.lr_peak - self.lr_min) * t / self.warmup_steps
        else:
            # Cosine decay
            progress = (t - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            return self.lr_min + 0.5 * (self.lr_peak - self.lr_min) * (
                1.0 + math.cos(math.pi * progress)
            )

    def _apply_lr(self):
        lr = self._get_lr()
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
            for pg in self.model.policy.optimizer.param_groups:
                pg['lr'] = lr

    def _on_training_start(self):
        self._apply_lr()
        print(
            f"✅ WarmupCosineDecayLR: "
            f"peak={self.lr_peak}  min={self.lr_min}  "
            f"warmup={self.warmup_steps} steps"
        )

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
        env.set_mixed_mode(num_trains)
        return ActionMasker(env, mask_fn)
    return _make


# ─────────────────────────────────────────────────────────────────────────────
# SHIELD-AWARE ENV FACTORY
# Used when training WITH the FeasibilityShield active so the RL Agent learns
# a policy that is always compatible with the Shield's safety constraints.
# The Shield's vetoes are merged into the action mask at every step, meaning
# the agent NEVER sees Shield-blocked actions as available options during
# training. After training, attaching the Shield at inference is seamless
# because the agent's policy was built entirely within the Shield's guardrails.
# ─────────────────────────────────────────────────────────────────────────────

def mask_fn_with_shield(env: gym.Env) -> np.ndarray:
    """
    Combines the environment's base action mask with the FeasibilityShield's
    real-time vetoes. The Shield instance is stored on the env so it persists
    across steps and can accumulate statistics.
    """
    base_mask = env.get_action_mask()          # shape: (n_trains, n_actions)

    # Lazy-init shield on the env instance so it is created once per worker
    if not hasattr(env, '_shield'):
        env._shield = FeasibilityShield(
            track_map=env.track_map,
            station_nodes=env.station_nodes,
            token_blocks=list(env.token_blocks),
        )

    shield_mask = env._shield.get_masked_actions(
        sim_time=env.sim_time,
        current_mask=base_mask,
        trains=env.trains,
        schedule=env.schedule,
        ghat_token=env.ghat_token,
    )                                          # shape: (n_trains, n_actions)

    # Combine: action is legal only if BOTH masks allow it
    combined = base_mask & shield_mask

    # Safety fallback: if Shield masks ALL actions for a train,
    # fall back to base mask to avoid an impossible action space
    for i in range(combined.shape[0]):
        if not combined[i].any():
            combined[i] = base_mask[i]

    return combined


def make_env_fn_with_shield(num_trains: int):
    """Returns a callable that creates one Shield-aware masked env."""
    def _make():
        env = TrainDispatchEnv()
        env.set_mixed_mode(num_trains)
        return ActionMasker(env, mask_fn_with_shield)
    return _make


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train_manual(
    level:          str,
    num_trains:     int,
    total_steps:    int,
    load_path:      str   = None,
    learning_rate:  float = None,   # None = use PER_LEVEL_HPARAMS
    no_early_stop:  bool  = False,
    ent_coef:       float = None,   # None = use PER_LEVEL_HPARAMS
    chaos:          bool  = False,
    hardcore:       bool  = False,
    incident:       bool  = False,
):
    # Resolve per-level hyperparameters
    lvl_int  = int(level) if str(level).isdigit() else 1
    lvl_hp   = PER_LEVEL_HPARAMS.get(lvl_int, PER_LEVEL_HPARAMS[6])
    # CLI overrides take priority; otherwise use per-level table
    peak_lr   = learning_rate if learning_rate is not None else lvl_hp['peak_lr']
    _ent_coef = ent_coef      if ent_coef      is not None else lvl_hp['ent_coef']
    _gamma    = lvl_hp['gamma']
    _gae_lam  = lvl_hp['gae_lambda']
    _n_epochs = lvl_hp['n_epochs']
    _clip_rng = lvl_hp['clip_range']
    print(f"\n{'='*60}")
    print(f"🚂 TRAINING: Level {level} | {num_trains} Trains | {total_steps} Steps")
    print(f"   LR={peak_lr}  ent_coef={_ent_coef}  gamma={_gamma}  "
          f"gae_lambda={_gae_lam}  n_epochs={_n_epochs}  clip={_clip_rng}")
    print(f"   chaos={chaos}  hardcore={hardcore}  incident={incident}")
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
    # Obs shape is now (10, 24) in Phase 3 (added required_speed_norm).
    # Reward distributions also shift per level. Always start fresh.
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
                'learning_rate': peak_lr,
                'lr_schedule':   lambda _: peak_lr,
                'clip_range':    _clip_rng,
                'ent_coef':      _ent_coef,
                'gamma':         _gamma,
                'gae_lambda':    _gae_lam,
                'n_epochs':      _n_epochs,
            },
        )
        print(f"✅ Model loaded | LR overridden → {peak_lr} | ent_coef={_ent_coef} | gamma={_gamma}")

    else:
        print("✨ Creating NEW MaskablePPO model")
        model = MaskablePPO(
            'MlpPolicy',
            train_env,
            verbose=1,
            tensorboard_log=LOGS_DIR,
            learning_rate=peak_lr,   # WarmupCosineDecayLR callback overrides this
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=_n_epochs,
            gamma=_gamma,
            gae_lambda=_gae_lam,
            clip_range=_clip_rng,
            ent_coef=_ent_coef,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            device='auto',
        )

    # ── Enable chaos mode if requested ───────────────────────────────────
    if chaos or hardcore:
        print("🐒 Chaos mode enabled on training envs")
        train_env.env_method('set_chaos_mode', True, hardcore)

    if incident:
        print("🚨 Incident mode enabled on training envs")
        train_env.env_method('set_incident_mode', True)

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

    lr_callback = WarmupCosineDecayLR(
        total_steps=total_steps,
        lr_peak=peak_lr,
        lr_min=LR_MIN,
        warmup_frac=LR_WARMUP_FRAC,
    )

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
    next_trains = {1: 5, 2: 7, 3: 10, 4: 15, 5: 25, 6: 25}.get(int(level), num_trains + 2)
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
    parser.add_argument('--lr',            type=float, default=PEAK_LR,
                        help=f'Peak learning rate for cosine schedule (default: {PEAK_LR})')
    parser.add_argument('--no-early-stop', action='store_true',
                        help='Disable early stopping on reward threshold')
    parser.add_argument('--ent-coef',      type=float, default=ENT_COEF,
                        help=f'Entropy coefficient (default: {ENT_COEF})')
    parser.add_argument('--chaos',         action='store_true',
                        help='Enable chaos monkey (delayed starts, speed snags)')
    parser.add_argument('--hardcore',      action='store_true',
                        help='Enable hardcore chaos (60% delay, up to 30m)')
    parser.add_argument('--incident',      action='store_true',
                        help='Enable incident mode (broken tracks)')

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
        hardcore=args.hardcore,
        incident=args.incident,
    )