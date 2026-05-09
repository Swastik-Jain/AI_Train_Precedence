"""
run_bc_warmup.py
================
Step 2 of the Hybrid Pipeline — Behaviour Cloning (BC) Warm-up.

Loads expert_actions.json (produced by generate_golden_trajectories.py)
and trains the MaskablePPO policy directly on the expert state-action pairs
using supervised Cross-Entropy Loss.

What you should see:
  Epoch 1  → Policy Loss: ~1.09  (random baseline, log(3) ≈ 1.099)
  Epoch 3  → Policy Loss: ~0.6
  Epoch 8  → Policy Loss: ~0.35
  Epoch 15 → Policy Loss: <0.20  (agent "inherits" OR-solver logic)

Usage:
  cd backend/ai
  python run_bc_warmup.py --expert expert_actions.json --trains 5
  python run_bc_warmup.py --expert expert_actions.json --epochs 20 --lr 1e-4
  python run_bc_warmup.py --expert expert_actions.json --load models/existing.zip
"""

import os
import sys
import json
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ["TORCH_COMPILE_DISABLE"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("BC-Warmup")

NUM_ACTIONS    = 3          # 0=STOP, 1=MAIN, 2=DIVERT
OBS_FEATURES   = 10         # features per train in observation vector


# ─────────────────────────────────────────────────────────────────
# DATASET BUILDER
# Rolls out the environment following the expert actions step-by-step
# so every (obs, action) pair reflects the expert trajectory, not null.
# ─────────────────────────────────────────────────────────────────

def build_bc_dataset(expert_data: dict, env, max_trains: int):
    """
    Returns
    -------
    obs_tensor    : FloatTensor  [T, max_trains * OBS_FEATURES]
    action_tensor : LongTensor   [T, max_trains]
    """
    train_ids   = [t["id"] for t in env.trains]
    expert_acts = expert_data["expert_actions"]

    # Episode length = longest OR action sequence
    ep_len = max(len(v) for v in expert_acts.values()) if expert_acts else 0
    if ep_len == 0:
        raise ValueError("expert_actions.json has empty action sequences.")

    logger.info(f"📦 Building BC dataset: {ep_len} steps × {max_trains} trains")

    obs_list    = []
    action_list = []

    obs, _ = env.reset()

    for step in range(ep_len):
        # Flatten observation (max_trains, 10) → (max_trains * 10)
        obs_list.append(obs.flatten().copy())

        # Expert action vector for this step
        action_vec = np.zeros(max_trains, dtype=np.int64)
        for idx, t_id in enumerate(train_ids):
            if idx >= max_trains:
                break
            if t_id in expert_acts and step < len(expert_acts[t_id]):
                action_vec[idx] = int(expert_acts[t_id][step])

        action_list.append(action_vec.copy())

        # Step env along expert trajectory
        obs, _, terminated, _, _ = env.step(action_vec)
        if terminated:
            obs, _ = env.reset()

    obs_tensor    = torch.tensor(np.array(obs_list),    dtype=torch.float32)
    action_tensor = torch.tensor(np.array(action_list), dtype=torch.long)

    logger.info(
        f"  obs_tensor:    {tuple(obs_tensor.shape)}  "
        f"({obs_tensor.nbytes / 1e6:.1f} MB)"
    )
    logger.info(
        f"  action_tensor: {tuple(action_tensor.shape)}"
    )
    return obs_tensor, action_tensor


# ─────────────────────────────────────────────────────────────────
# POLICY FORWARD PASS
# Extracts logits from the MaskablePPO MlpPolicy for our action heads
# ─────────────────────────────────────────────────────────────────

def get_action_logits(policy, obs_flat, max_trains):
    """
    Forwards obs_flat [B, max_trains*10] through the MlpPolicy actor head.
    Returns logits [B, max_trains, NUM_ACTIONS].
    """
    # Extract latent actor features
    features    = policy.extract_features(obs_flat)
    latent_pi   = policy.mlp_extractor.forward_actor(features)
    action_logits = policy.action_net(latent_pi)      # [B, max_trains * NUM_ACTIONS]

    B = obs_flat.shape[0]
    return action_logits.view(B, max_trains, NUM_ACTIONS)


# ─────────────────────────────────────────────────────────────────
# BEHAVIOUR CLONING TRAINING LOOP
# ─────────────────────────────────────────────────────────────────

def run_bc(
    model,
    obs_tensor,
    action_tensor,
    max_trains,
    epochs      = 15,
    batch_size  = 64,
    lr          = 3e-4,
    save_path   = None,
    device      = None,
):
    """
    Trains the MaskablePPO policy via supervised Cross-Entropy Loss.

    Parameters
    ----------
    model        : MaskablePPO
    obs_tensor   : FloatTensor [T, max_trains * 10]
    action_tensor: LongTensor  [T, max_trains]
    max_trains   : int
    epochs       : int
    batch_size   : int
    lr           : float
    save_path    : str | None — where to save the warmed-up model
    device       : str | None — 'cpu' / 'cuda' / 'mps'

    Returns
    -------
    model : MaskablePPO (with updated policy weights)
    history : list of (epoch, avg_loss, per_action_loss) dicts
    """
    if device is None:
        device = str(model.device)

    policy    = model.policy.to(device)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    obs_tensor    = obs_tensor.to(device)
    action_tensor = action_tensor.to(device)
    num_samples   = obs_tensor.shape[0]

    logger.info("=" * 64)
    logger.info("🎓 BEHAVIOUR CLONING — Training on Expert Trajectories")
    logger.info(f"   Model device : {device}")
    logger.info(f"   Dataset size : {num_samples} steps")
    logger.info(f"   Batch size   : {batch_size}")
    logger.info(f"   Epochs       : {epochs}")
    logger.info(f"   LR           : {lr}")
    logger.info("=" * 64)

    # Action name mapping for per-class loss breakdown
    action_names = ["STOP", "MAIN", "DIVERT"]

    history = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        perm        = torch.randperm(num_samples, device=device)

        epoch_loss      = 0.0
        per_action_loss = {a: 0.0 for a in action_names}
        per_action_cnt  = {a: 0   for a in action_names}
        num_batches     = 0
        correct         = 0
        total           = 0

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            idx = perm[start:end]

            obs_b    = obs_tensor[idx]        # [B, max_trains * 10]
            act_b    = action_tensor[idx]     # [B, max_trains]

            # Forward
            logits = get_action_logits(policy, obs_b, max_trains)
            # logits: [B, max_trains, 3]

            # Reshape for CE: [B*max_trains, 3] vs [B*max_trains]
            logits_flat = logits.view(-1, NUM_ACTIONS)
            acts_flat   = act_b.view(-1)

            loss = criterion(logits_flat, acts_flat)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_loss  += loss.item()
            num_batches += 1

            # Accuracy
            preds    = logits_flat.argmax(dim=1)
            correct += (preds == acts_flat).sum().item()
            total   += acts_flat.numel()

            # Per-action breakdown
            for a_idx, a_name in enumerate(action_names):
                mask     = (acts_flat == a_idx)
                if mask.any():
                    a_loss = criterion(logits_flat[mask], acts_flat[mask])
                    per_action_loss[a_name] += a_loss.item()
                    per_action_cnt[a_name]  += 1

        scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        accuracy = correct / max(total, 1) * 100
        elapsed  = time.perf_counter() - epoch_start

        # Per-action average losses
        pa_str_parts = []
        for a_name in action_names:
            cnt = per_action_cnt[a_name]
            al  = per_action_loss[a_name] / max(cnt, 1)
            pa_str_parts.append(f"{a_name}={al:.3f}")
        pa_str = "  ".join(pa_str_parts)

        # Live progress bar
        bar_len  = 20
        filled   = int(bar_len * epoch / epochs)
        bar      = "█" * filled + "░" * (bar_len - filled)

        print(
            f"  Epoch [{epoch:>3}/{epochs}] [{bar}] "
            f"Loss: {avg_loss:.4f}  Acc: {accuracy:5.1f}%  "
            f"LR: {scheduler.get_last_lr()[0]:.2e}  "
            f"({elapsed:.1f}s)\n"
            f"            Per-action: {pa_str}"
        )

        history.append({
            "epoch":           epoch,
            "avg_loss":        round(avg_loss, 5),
            "accuracy_pct":    round(accuracy, 2),
            "per_action_loss": {k: round(v / max(per_action_cnt[k], 1), 5)
                                for k, v in per_action_loss.items()},
        })

    logger.info("")
    logger.info("=" * 64)
    logger.info(f"✅ BC Training complete.")
    logger.info(f"   Initial loss : {history[0]['avg_loss']:.4f}")
    logger.info(f"   Final   loss : {history[-1]['avg_loss']:.4f}")
    logger.info(
        f"   Loss drop    : "
        f"{((history[0]['avg_loss'] - history[-1]['avg_loss']) / history[0]['avg_loss'] * 100):.1f}%"
    )
    logger.info(f"   Final accuracy: {history[-1]['accuracy_pct']:.1f}%")
    logger.info("=" * 64)

    if save_path:
        model.save(save_path)
        logger.info(f"💾 Warmed-up model saved → {save_path}")

    return model, history


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Behaviour Cloning warm-up for MaskablePPO on OR expert data",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--expert", default="expert_actions.json",
        help="Path to expert_actions.json (from generate_golden_trajectories.py)"
    )
    parser.add_argument(
        "--load", default=None,
        help="Path to an existing MaskablePPO .zip to continue from"
    )
    parser.add_argument(
        "--trains", type=int, default=5,
        help="Number of trains for environment difficulty"
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Number of BC training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate for BC Adam optimizer"
    )
    parser.add_argument(
        "--out", default=None,
        help="Save path for warmed-up model (default: models/hybrid_step1_BC_warmup.zip)"
    )
    args = parser.parse_args()

    # ── Guard: check expert file ───────────────────────────────────────────
    if not os.path.exists(args.expert):
        logger.error(
            f"expert_actions.json not found at '{args.expert}'.\n"
            "Run generate_golden_trajectories.py first:\n"
            "  python generate_golden_trajectories.py"
        )
        sys.exit(1)

    with open(args.expert) as f:
        expert_data = json.load(f)

    if "expert_actions" not in expert_data or not expert_data["expert_actions"]:
        logger.error("'expert_actions' key missing or empty in expert file.")
        sys.exit(1)

    logger.info(f"✅ Loaded expert data: {len(expert_data['expert_actions'])} trains")

    # ── Build environment ──────────────────────────────────────────────────
    from train_env import TrainDispatchEnv
    from config import MAX_TRAINS_CAPACITY

    env = TrainDispatchEnv()
    env.set_difficulty(args.trains)
    logger.info(f"🌍 Environment ready: {args.trains} trains | "
                f"MAX_TRAINS_CAPACITY={MAX_TRAINS_CAPACITY}")

    # ── Build or load model ────────────────────────────────────────────────
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.env_util import make_vec_env
    import gymnasium as gym

    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    LOGS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(MODELS_DIR, exist_ok=True)

    def mask_fn(e: gym.Env) -> np.ndarray:
        return e.get_action_mask()

    def make_env_fn():
        e = TrainDispatchEnv()
        return ActionMasker(e, mask_fn)

    vec_env  = make_vec_env(make_env_fn, n_envs=1)
    norm_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, clip_reward=10.0)

    stats_path = os.path.join(MODELS_DIR, "vec_normalize_hybrid.pkl")

    if args.load and os.path.exists(args.load):
        logger.info(f"🔄 Loading model from {args.load}")
        if os.path.exists(stats_path):
            norm_env = VecNormalize.load(stats_path, vec_env)
        model = MaskablePPO.load(args.load, env=norm_env, tensorboard_log=LOGS_DIR)
    else:
        logger.info("✨ Creating fresh MaskablePPO model...")
        model = MaskablePPO(
            "MlpPolicy", norm_env,
            verbose=0,
            tensorboard_log=LOGS_DIR,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.03,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="auto"
        )

    # ── Build dataset (rolls out env following expert) ─────────────────────
    obs_tensor, action_tensor = build_bc_dataset(expert_data, env, MAX_TRAINS_CAPACITY)

    # ── Run BC ────────────────────────────────────────────────────────────
    save_path = args.out or os.path.join(MODELS_DIR, "hybrid_step1_BC_warmup.zip")

    model, history = run_bc(
        model        = model,
        obs_tensor   = obs_tensor,
        action_tensor= action_tensor,
        max_trains   = MAX_TRAINS_CAPACITY,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        save_path    = save_path,
    )

    # Save normalisation stats alongside model
    norm_env.save(stats_path)
    logger.info(f"👓 Normalisation stats → {stats_path}")

    # ── Write training history ────────────────────────────────────────────
    hist_path = os.path.join(MODELS_DIR, "bc_loss_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=4)
    logger.info(f"📈 Loss history → {hist_path}")

    # ── Print the "what to do next" message ──────────────────────────────
    print("\n" + "═" * 64)
    print("  Behaviour Cloning complete. Your agent now understands:")
    print("  ✔ Stop at red signals (STOP action at blocked blocks)")
    print("  ✔ Yield for Vande Bharat / Rajdhani (priority precedence)")
    print("  ✔ Divert to loops at stations (DIVERT action)")
    print("═" * 64)
    print(f"\n  Next → Run Step 2 (Masked PPO Exploration):")
    print(
        f"  python hybrid_connector.py --step 2 \\\n"
        f"    --load {save_path} \\\n"
        f"    --expert {args.expert} \\\n"
        f"    --trains {args.trains} \\\n"
        f"    --steps 200000\n"
    )


if __name__ == "__main__":
    main()
