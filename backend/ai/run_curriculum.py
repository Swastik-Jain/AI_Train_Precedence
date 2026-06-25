"""
run_curriculum.py — Adaptive Curriculum Orchestrator
CSMT-Manmad Train Dispatcher — Phase 4

Replaces run_training.sh with an intelligent training loop that:
  1. Trains each level for a minimum step budget
  2. After each eval checkpoint, checks if the model has truly mastered the level:
       - Rolling avg reward > mastery threshold
       - Completion % > mastery completion target
       - Reward trend is flat (plateau) — not still improving
  3. If IMPROVING → extend training (up to per-level max budget)
  4. If MASTERED (plateau + threshold met) → advance to next level
  5. If STUCK (plateau + threshold NOT met) → extend by one block (up to max), log warning
  6. If MAX BUDGET reached → advance with a warning (don't loop forever)

Usage:
    python ai/run_curriculum.py                          # full L1→L6 curriculum
    python ai/run_curriculum.py --start-level 4          # resume from L4
    python ai/run_curriculum.py --start-level 5 \\
        --end-level 5 \\
        --load ai/models/Phase3/L4_10Trains_Best/best_model.zip \\
        --tag v2    # saves to L5_15Trains_Best_v2/ — never overwrites original
"""

import os
import sys
import argparse
import time
import json

os.environ['TORCH_COMPILE_DISABLE'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from train_env import TrainDispatchEnv
from ai.train_manual import (
    WarmupCosineDecayLR,
    PER_LEVEL_HPARAMS,
    N_STEPS, BATCH_SIZE, VF_COEF, MAX_GRAD_NORM, LR_MIN, LR_WARMUP_FRAC,
    N_ENVS_TRAIN, MODELS_DIR, LOGS_DIR,
    mask_fn, make_env_fn_with_shield,
)

# ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

CURRICULUM = [
    {
        'level':       1,
        'trains':      2,
        'min_steps':   400_000,    # must train at least this long
        'step_block':  200_000,    # extend in blocks of this size
        'max_steps':   1_500_000,  # hard ceiling — advance regardless
        # Mastery criteria
        'reward_threshold':    8.0,    # rolling avg reward (post √n normalisation)
        'completion_target':   1.00,   # 100% trains completing
        # Plateau detection
        'plateau_window':  5,          # number of evals to check for plateau
        'plateau_epsilon': 0.5,        # max improvement to be called "plateau"
    },
    {
        'level':       2,
        'trains':      5,
        'min_steps':   800_000,
        'step_block':  400_000,
        'max_steps':   3_000_000,
        'reward_threshold':    20.0,
        'completion_target':   0.95,
        'plateau_window':  5,
        'plateau_epsilon': 1.0,
    },
    {
        'level':       3,
        'trains':      7,
        'min_steps':   1_000_000,
        'step_block':  500_000,
        'max_steps':   4_000_000,
        'reward_threshold':    28.0,
        'completion_target':   0.90,
        'plateau_window':  5,
        'plateau_epsilon': 1.5,
    },
    {
        'level':       4,
        'trains':      10,
        'min_steps':   1_500_000,
        'step_block':  500_000,
        'max_steps':   6_000_000,
        'reward_threshold':    15.0,
        'completion_target':   0.80,
        'plateau_window':  6,
        'plateau_epsilon': 1.5,
    },
    {
        'level':       5,
        'trains':      15,
        # Realistic targets: we already proved 95.6% is achievable.
        # Don't over-stretch to 98% — that caused over-specialisation last time.
        # plateau_window=6 means 6 flat evals (~4.2M steps of flat reward) before advancing.
        'min_steps':   2_000_000,  # 3 blocks to establish baseline from L4 warm-start
        'step_block':  700_000,
        'max_steps':   8_000_000,  # enough to converge without over-training
        'reward_threshold':    80.0,   # realistic for √n-normalised 15-train rewards
        'completion_target':   0.95,   # proven achievable; not over-stretched to 0.98
        'plateau_window':  6,
        'plateau_epsilon': 2.0,        # slightly stricter than original to confirm convergence
    },
    {
        'level':       6,
        'trains':      25,
        'min_steps':   2_000_000,  # 2 blocks minimum
        'step_block':  1_000_000,  # 1M steps per eval cycle
        'max_steps':   40_000_000, # Raised to 40M for extended baking
        # We know it can hit ~260 reward and 91% completion.
        # Set targets high so it keeps baking until it plateaus.
        'reward_threshold':    200.0,
        'completion_target':   0.90,
        'plateau_window':  8,           # 8 flat evals (~8M steps) before stopping
        'plateau_epsilon': 5.0,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# MASTERY CALLBACK
# Wraps MaskableEvalCallback and tracks rolling eval history.
# Exposes is_mastered() and is_improving() so the orchestrator can decide
# whether to extend, advance, or abort.
# ─────────────────────────────────────────────────────────────────────────────

class MasteryTracker(BaseCallback):
    """
    Hooks into the eval callback result and records rolling reward history.
    Does NOT stop training itself — the orchestrator loop decides that by
    calling model.learn() in successive step blocks.
    """

    def __init__(
        self,
        plateau_window:  int   = 5,
        plateau_epsilon: float = 1.0,
        reward_threshold: float = 10.0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.plateau_window   = plateau_window
        self.plateau_epsilon  = plateau_epsilon
        self.reward_threshold = reward_threshold
        self.eval_rewards     = []          # list of mean rewards from each eval
        self.eval_completions = []          # list of completion fractions
        self.best_reward      = -np.inf

    def _on_step(self) -> bool:
        return True

    def record_eval(self, mean_reward: float, completion_frac: float):
        """Called by the orchestrator after each eval block."""
        self.eval_rewards.append(mean_reward)
        self.eval_completions.append(completion_frac)
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
        if self.verbose >= 1:
            w = self.plateau_window
            recent = self.eval_rewards[-w:]
            trend  = (recent[-1] - recent[0]) if len(recent) >= 2 else float('nan')
            print(f"   [MasteryTracker] eval #{len(self.eval_rewards):>3} | "
                  f"reward={mean_reward:>8.2f} | best={self.best_reward:>8.2f} | "
                  f"completion={completion_frac*100:>5.1f}% | "
                  f"trend(last {min(w,len(recent))} evals)={trend:>+.2f}")

    def is_plateau(self) -> bool:
        """True when reward improvement over the last plateau_window evals
        is less than plateau_epsilon — the model has stopped learning."""
        if len(self.eval_rewards) < self.plateau_window:
            return False
        recent = self.eval_rewards[-self.plateau_window:]
        improvement = max(recent) - min(recent)
        return improvement < self.plateau_epsilon

    def is_mastered(self, completion_target: float) -> bool:
        """True when rolling avg reward and completion both meet the threshold."""
        if len(self.eval_rewards) < 3:
            return False
        recent_r = self.eval_rewards[-3:]
        recent_c = self.eval_completions[-3:]
        return (
            np.mean(recent_r) >= self.reward_threshold
            and np.mean(recent_c) >= completion_target
        )

    def is_improving(self) -> bool:
        """True when reward is still meaningfully trending up."""
        return not self.is_plateau()

    def summary(self) -> str:
        if not self.eval_rewards:
            return "no evals yet"
        return (f"evals={len(self.eval_rewards)} "
                f"best={self.best_reward:.2f} "
                f"last={self.eval_rewards[-1]:.2f} "
                f"plateau={self.is_plateau()}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK EVALUATION HELPER
# Runs N deterministic episodes and returns (mean_reward, completion_frac)
# without touching the training environment.
# ─────────────────────────────────────────────────────────────────────────────

def quick_eval(model: MaskablePPO, num_trains: int, n_episodes: int = 25) -> tuple:
    """
    Runs n_episodes deterministic episodes and returns:
        mean_reward    : float
        completion_frac: float  (fraction of trains that finished across all eps)

    25 episodes (up from 15) reduces checkpoint selection noise significantly.
    A 15-episode sample has ~40% more variance than a 25-episode sample,
    meaning the wrong checkpoint could be saved as 'best' in noisy environments.
    """
    rewards     = []
    completions = []

    for _ in range(n_episodes):
        env = TrainDispatchEnv()
        env.set_difficulty(num_trains)
        env = ActionMasker(env, mask_fn)
        obs, _ = env.reset()
        done, ep_r = False, 0.0

        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, _ = env.step(act)
            ep_r += r

        rewards.append(ep_r)

        # Try to count finished trains
        inner = env.env if hasattr(env, 'env') else env
        if hasattr(inner, 'trains'):
            fin = sum(1 for t in inner.trains if t.get('finished', False))
            completions.append(fin / max(num_trains, 1))
        else:
            completions.append(0.0)

    return float(np.mean(rewards)), float(np.mean(completions))


# ─────────────────────────────────────────────────────────────────────────────
# PER-LEVEL TRAINING BLOCK
# Trains the model for exactly `steps` timesteps and returns the model.
# The orchestrator calls this in a loop, extending until mastery or max budget.
# ─────────────────────────────────────────────────────────────────────────────

def train_block(
    model:      MaskablePPO,
    train_env:  VecNormalize,
    steps:      int,
    level:      int,
    run_name:   str,
    total_steps_so_far: int,
    total_budget: int,
):
    """Train for `steps` timesteps. Returns updated model."""
    hp = PER_LEVEL_HPARAMS.get(level, PER_LEVEL_HPARAMS[6])

    lr_callback = WarmupCosineDecayLR(
        total_steps  = total_budget,
        lr_peak      = hp['peak_lr'],
        lr_min       = LR_MIN,
        warmup_frac  = LR_WARMUP_FRAC,
    )
    # Override internal step counter so LR schedule is relative to total budget
    lr_callback.num_timesteps = total_steps_so_far

    model.learn(
        total_timesteps=steps,
        reset_num_timesteps=False,   # CRITICAL: continue from where we left off
        tb_log_name=run_name,
        callback=[lr_callback],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_curriculum(start_level: int = 1, end_level: int = 6, load_path: str = None, tag: str = ""):
    """
    tag: optional suffix appended to best-model save dirs.
         Use a non-empty tag (e.g. 'v2') when extending training on a level
         that already has a proven checkpoint. This PREVENTS overwriting the
         existing best_model.zip with a potentially noisier training checkpoint.
         Example:  L5_15Trains_Best_v2/  instead of  L5_15Trains_Best/
    """
    tag_suffix = f"_{tag}" if tag else ""

    print("\n" + "═" * 66)
    print("  🚂  CSMT-Manmad Adaptive Curriculum — Phase 4")
    print("  Levels: L1(2T) → L2(5T) → L3(7T) → L4(10T) → L5(15T) → L6(25T)")
    print("  Mode: Train until mastered, not just until steps exhausted")
    if tag_suffix:
        print(f"  Tag: '{tag}' — saving to *_Best{tag_suffix}/ (won't overwrite originals)")
    print("═" * 66 + "\n")

    model = None
    current_load = load_path

    # State log — saved after each level for crash recovery
    state_file = os.path.join(MODELS_DIR, "curriculum_state.json")

    for lvl_cfg in CURRICULUM:
        level      = lvl_cfg['level']
        num_trains = lvl_cfg['trains']

        if level < start_level:
            print(f"⏭  Skipping L{level} (start_level={start_level})")
            # Set load path to the best model for this level if it exists
            best = os.path.join(MODELS_DIR, f"L{level}_{num_trains}Trains_Best", "best_model.zip")
            if os.path.exists(best):
                current_load = best
            continue

        if level > end_level:
            print(f"⏭  Stopping at L{end_level} as requested.")
            break

        hp = PER_LEVEL_HPARAMS.get(level, PER_LEVEL_HPARAMS[6])

        print("\n" + "─" * 66)
        print(f"  ▶  LEVEL {level} — {num_trains} Trains")
        print(f"     ent_coef={hp['ent_coef']}  gamma={hp['gamma']}  "
              f"gae_lambda={hp['gae_lambda']}  n_epochs={hp['n_epochs']}")
        print(f"     lr={hp['peak_lr']}  clip={hp['clip_range']}")
        print(f"     min_steps={lvl_cfg['min_steps']:,}  "
              f"step_block={lvl_cfg['step_block']:,}  "
              f"max_steps={lvl_cfg['max_steps']:,}")
        print(f"     mastery: reward>{lvl_cfg['reward_threshold']}  "
              f"completion>{lvl_cfg['completion_target']*100:.0f}%")
        print("─" * 66)

        # ── Build envs ─────────────────────────────────────────────────
        # Use the Shield-aware env factory so the RL Agent learns a policy
        # that is always compatible with the Shield's safety constraints.
        raw_train = make_vec_env(
            make_env_fn_with_shield(num_trains),
            n_envs=N_ENVS_TRAIN,
            vec_env_cls=SubprocVecEnv,
        )
        train_env = VecNormalize(
            raw_train, norm_obs=True, norm_reward=True,
            clip_obs=10.0, clip_reward=10.0,
        )

        # ── Build or load model ───────────────────────────────────────────
        if current_load and os.path.exists(current_load):
            print(f"   🔄 Loading checkpoint: {current_load}")
            model = MaskablePPO.load(
                current_load,
                env=train_env,
                tensorboard_log=LOGS_DIR,
                custom_objects={
                    'learning_rate': hp['peak_lr'],
                    'lr_schedule':   lambda _: hp['peak_lr'],
                    'clip_range':    hp['clip_range'],
                    'ent_coef':      hp['ent_coef'],
                    'gamma':         hp['gamma'],
                    'gae_lambda':    hp['gae_lambda'],
                    'n_epochs':      hp['n_epochs'],
                    'n_steps':       N_STEPS,
                    'batch_size':    BATCH_SIZE,
                },
            )
        else:
            print("   ✨ Creating fresh MaskablePPO model")
            model = MaskablePPO(
                'MlpPolicy', train_env,
                verbose=1,
                tensorboard_log=LOGS_DIR,
                learning_rate=hp['peak_lr'],
                n_steps=N_STEPS,
                batch_size=BATCH_SIZE,
                n_epochs=hp['n_epochs'],
                gamma=hp['gamma'],
                gae_lambda=hp['gae_lambda'],
                clip_range=hp['clip_range'],
                ent_coef=hp['ent_coef'],
                vf_coef=VF_COEF,
                max_grad_norm=MAX_GRAD_NORM,
                device='auto',
            )

        # Save dir uses tag suffix so we never overwrite a proven checkpoint
        best_save_dir = os.path.join(MODELS_DIR, f"L{level}_{num_trains}Trains_Best{tag_suffix}")
        os.makedirs(best_save_dir, exist_ok=True)
        print(f"   💾 Best model will save to: {best_save_dir}/")

        tracker  = MasteryTracker(
            plateau_window   = lvl_cfg['plateau_window'],
            plateau_epsilon  = lvl_cfg['plateau_epsilon'],
            reward_threshold = lvl_cfg['reward_threshold'],
        )
        run_name = f"PPO_L{level}_{num_trains}Trains"

        steps_trained    = 0
        block_num        = 0
        best_reward_seen = -np.inf
        advance_reason   = "max_budget"

        # ── Training loop — extend until mastered or max budget ───────────
        while steps_trained < lvl_cfg['max_steps']:

            # How many steps in this block?
            if steps_trained < lvl_cfg['min_steps']:
                # First block: train to minimum budget
                block_steps = min(
                    lvl_cfg['min_steps'] - steps_trained,
                    lvl_cfg['step_block'],
                )
            else:
                block_steps = lvl_cfg['step_block']

            # Don't overshoot max
            block_steps = min(block_steps, lvl_cfg['max_steps'] - steps_trained)
            if block_steps <= 0:
                break

            block_num    += 1
            steps_trained += block_steps

            print(f"\n   [Block {block_num}] Training {block_steps:,} steps "
                  f"(total so far: {steps_trained:,} / {lvl_cfg['max_steps']:,})")

            model = train_block(
                model=model,
                train_env=train_env,
                steps=block_steps,
                level=level,
                run_name=run_name,
                total_steps_so_far=steps_trained - block_steps,
                total_budget=lvl_cfg['max_steps'],
            )

            # ── Evaluate after this block ─────────────────────────────────
            print(f"   [Block {block_num}] Evaluating ({15} episodes)...")
            mean_r, comp = quick_eval(model, num_trains, n_episodes=15)
            tracker.record_eval(mean_r, comp)

            # Save best model checkpoint
            if mean_r > best_reward_seen:
                best_reward_seen = mean_r
                model.save(os.path.join(best_save_dir, "best_model"))
                print(f"   💾 New best saved: reward={mean_r:.2f}")

            # ── Mastery check (only after minimum budget) ─────────────────
            if steps_trained >= lvl_cfg['min_steps']:
                mastered  = tracker.is_mastered(lvl_cfg['completion_target'])
                improving = tracker.is_improving()
                plateau   = tracker.is_plateau()

                print(f"   [Check] mastered={mastered}  improving={improving}  "
                      f"plateau={plateau}  {tracker.summary()}")

                if mastered and plateau:
                    advance_reason = "mastered_and_plateau"
                    print(f"\n   ✅ MASTERY CONFIRMED at {steps_trained:,} steps — "
                          f"reward={mean_r:.2f} ≥ {lvl_cfg['reward_threshold']}  "
                          f"completion={comp*100:.1f}% ≥ {lvl_cfg['completion_target']*100:.0f}%")
                    break

                if mastered and improving:
                    advance_reason = "mastered_still_improving"
                    print(f"\n   🔥 Mastered BUT still improving — letting it bake "
                          f"(another {lvl_cfg['step_block']:,} steps)...")
                    # Don't break — continue baking!

                if plateau and not mastered:
                    print(f"\n   ⚠️  PLATEAU without mastery at {steps_trained:,} steps. "
                          f"reward={mean_r:.2f} < {lvl_cfg['reward_threshold']}. "
                          f"Extending training...")
                    # Continue loop — will add another block unless at max

        # ── End of level ─────────────────────────────────────────────────
        # Final save
        final_path = os.path.join(MODELS_DIR, f"ppo_L{level}_{num_trains}Trains_final.zip")
        model.save(final_path)
        norm_path  = os.path.join(MODELS_DIR, f"vec_normalize_L{level}_{num_trains}Trains.pkl")
        train_env.save(norm_path)
        train_env.close()

        print(f"\n   {'═'*50}")
        print(f"   Level {level} complete — {advance_reason}")
        print(f"   Steps trained : {steps_trained:,}")
        print(f"   Best reward   : {best_reward_seen:.2f}")
        print(f"   Model saved   : {final_path}")
        print(f"   Best model    : {best_save_dir}/best_model.zip")
        print(f"   {'═'*50}\n")

        # Persist curriculum state for crash recovery
        state = {
            'last_completed_level': level,
            'steps_trained': steps_trained,
            'best_reward': best_reward_seen,
            'advance_reason': advance_reason,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

        # Next level loads from this level's tagged best model
        current_load = os.path.join(best_save_dir, "best_model.zip")

    print("\n" + "═" * 66)
    print("  ✅  Full Curriculum Complete!")
    print(f"  Final model: {MODELS_DIR}/L6_25Trains_Best/best_model.zip")
    print("  Run: python eval_all_levels.py  to verify performance")
    print("═" * 66 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description='Adaptive Curriculum Orchestrator — CSMT-Manmad'
    )
    parser.add_argument(
        '--start-level', type=int, default=1,
        help='Curriculum level to start from (1-6). Skips earlier levels.',
    )
    parser.add_argument(
        '--end-level', type=int, default=6,
        help='Curriculum level to stop at (inclusive). Default: 6.',
    )
    parser.add_argument(
        '--load', type=str, default=None,
        help='Path to a model checkpoint to start the first trained level from.',
    )
    parser.add_argument(
        '--tag', type=str, default='',
        help=(
            'Optional suffix for save directories (e.g. "v2"). '
            'Saves to L5_15Trains_Best_v2/ instead of L5_15Trains_Best/. '
            'Use this when re-training a level that already has a proven checkpoint '
            'so you never accidentally overwrite it.'
        ),
    )
    args = parser.parse_args()

    run_curriculum(
        start_level=args.start_level,
        end_level=args.end_level,
        load_path=args.load,
        tag=args.tag,
    )
