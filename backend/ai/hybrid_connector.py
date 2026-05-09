"""
hybrid_connector.py
===================
RL-OR Hybrid Integration Bridge (The Handshake)

Architecture:
  Channel A — Behavior Cloning (BC) warm-up with OR-Tools expert schedule.
  Channel B — OR-Solver feasibility check injected into get_action_mask().
  Monitoring — OR_RL_Similarity_Score logged per step for TensorBoard.

Teacher-Student Workflow:
  Step 1 (Warm-up)   → pretrain_from_expert()    : BC from expert_actions.json
  Step 2 (Explore)   → Masked PPO Training         : OR "Lawyer" blocks illegal moves
  Step 3 (Refine)    → Fine-tune without BC pressure, RL optimises reward
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Silence OR-Tools trace inside training loops
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("HybridConnector")

# ─────────────────────────────────────────────────
# CONSTANTS — must stay aligned with or_solver.py
# ─────────────────────────────────────────────────
TIME_HORIZON_MINUTES = 120          # episode length ceiling (minutes)
ACTION_STOP   = 0                   # Wait at signal / loop
ACTION_MAIN   = 1                   # Proceed on main line
ACTION_DIVERT = 2                   # Enter loop / platform
NUM_ACTIONS   = 3

# ─────────────────────────────────────────────────
# CHANNEL A — BEHAVIOUR CLONING (BC)
# ─────────────────────────────────────────────────

def load_expert_data(expert_data_path: str) -> dict:
    """Load and validate the expert_actions.json produced by or_solver.py."""
    if not os.path.exists(expert_data_path):
        raise FileNotFoundError(
            f"expert_actions.json not found at '{expert_data_path}'. "
            "Run or_solver.py first to generate the expert schedule."
        )
    with open(expert_data_path, "r") as f:
        data = json.load(f)

    if "expert_actions" not in data:
        raise KeyError("'expert_actions' key missing from expert_actions.json.")

    logger.info(
        f"✅ Loaded expert data for {len(data['expert_actions'])} trains "
        f"from '{expert_data_path}'."
    )
    return data


def _build_bc_dataset(expert_data: dict, env, max_trains: int):
    """
    Step through the environment's reset state and align per-minute expert
    actions with the 10-feature observation vectors to produce (obs, action)
    supervised pairs.

    Returns
    -------
    obs_tensor    : FloatTensor  [T, max_trains, 10]
    action_tensor : LongTensor   [T, max_trains]
    """
    train_ids   = [t["id"] for t in env.trains]
    expert_acts = expert_data["expert_actions"]

    # Find the episode length (longest expert action sequence)
    ep_len = max(len(v) for v in expert_acts.values()) if expert_acts else 0
    if ep_len == 0:
        raise ValueError("Expert data has zero-length action sequences.")

    obs_list    = []
    action_list = []

    # Reset env and collect aligned (obs, action) pairs step by step
    obs, _ = env.reset()

    for step in range(ep_len):
        obs_list.append(obs.copy())           # shape: (max_trains, 10)

        # Build the action vector for this timestep
        action_vec = np.zeros(max_trains, dtype=np.int64)
        for idx, t_id in enumerate(train_ids):
            if t_id in expert_acts and step < len(expert_acts[t_id]):
                action_vec[idx] = int(expert_acts[t_id][step])
            # else: leave as 0 (STOP) — safe default

        action_list.append(action_vec)

        # Advance environment with expert action so subsequent observations
        # reflect the expert trajectory rather than a null trajectory.
        obs, _, terminated, _, _ = env.step(action_vec)
        if terminated:
            obs, _ = env.reset()

    obs_tensor    = torch.tensor(np.array(obs_list),    dtype=torch.float32)
    action_tensor = torch.tensor(np.array(action_list), dtype=torch.long)

    logger.info(
        f"📦 BC dataset built: {len(obs_list)} steps × {max_trains} trains."
    )
    return obs_tensor, action_tensor


def pretrain_from_expert(
    model,
    env,
    expert_data_path: str = "expert_actions.json",
    max_trains: int       = None,
    bc_epochs: int        = 15,
    lr: float             = 3e-4,
    batch_size: int       = 64,
    bc_loss_weight: float = 1.0
):
    """
    Channel A: Behaviour Cloning warm-up.

    Updates the MaskablePPO *policy* weights using supervised Cross-Entropy
    Loss so the agent's initial action distribution mirrors the OR-solver's
    optimal decisions before any PPO self-exploration begins.

    Parameters
    ----------
    model           : MaskablePPO  — the RL model whose policy is to be warmed up.
    env             : TrainDispatchEnv  — unwrapped (single) environment instance.
    expert_data_path: str  — path to expert_actions.json.
    max_trains      : int  — must match MAX_TRAINS_CAPACITY from config.
    bc_epochs       : int  — number of supervised passes over the dataset.
    lr              : float — learning rate for the BC SGD step.
    batch_size      : int  — mini-batch size.
    bc_loss_weight  : float — scale factor applied to CE loss (for fine-tuning
                              phase where BC guidance is phased out, set < 1.0).
    """
    if max_trains is None:
        from ai.config import MAX_TRAINS_CAPACITY
        max_trains = MAX_TRAINS_CAPACITY

    expert_data          = load_expert_data(expert_data_path)
    obs_tensor, act_tensor = _build_bc_dataset(expert_data, env, max_trains)

    policy    = model.policy
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    num_samples = obs_tensor.shape[0]

    logger.info(f"🎓 Starting Behaviour Cloning — {bc_epochs} epochs, lr={lr}")

    for epoch in range(bc_epochs):
        perm   = torch.randperm(num_samples)
        epoch_loss  = 0.0
        num_batches = 0

        for start in range(0, num_samples, batch_size):
            end   = min(start + batch_size, num_samples)
            idx   = perm[start:end]

            # obs shape  : [B, max_trains, 10]  →  flatten to [B, max_trains*10]
            obs_b    = obs_tensor[idx].to(model.device)
            obs_flat = obs_b.view(obs_b.shape[0], -1)

            # Forward pass through the actor head
            with torch.no_grad():
                features = policy.extract_features(obs_b)
                latent_pi, _ = policy.mlp_extractor(features)

            # action_net produces logits per action head
            # MultiDiscrete has max_trains heads, each size 3
            action_logits = policy.action_net(latent_pi)  # [B, max_trains*3]
            action_logits = action_logits.view(
                action_logits.shape[0], max_trains, NUM_ACTIONS
            )  # [B, max_trains, 3]

            # Target actions
            act_b = act_tensor[idx].to(model.device)  # [B, max_trains]

            # Cross-entropy over all trains simultaneously
            # Reshape: [B*max_trains, 3]  vs  [B*max_trains]
            loss = criterion(
                action_logits.view(-1, NUM_ACTIONS),
                act_b.view(-1)
            ) * bc_loss_weight

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_loss  += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"  BC Epoch [{epoch+1}/{bc_epochs}] — avg_loss: {avg_loss:.4f}")

    logger.info("✅ Behaviour Cloning pre-training complete.")
    return model


# ─────────────────────────────────────────────────
# CHANNEL B — OR-SOLVER FEASIBILITY SHIELD
# ─────────────────────────────────────────────────

class FeasibilityShield:
    """
    Lightweight wrapper around the OR-solver's schedule to provide
    real-time feasibility checks callable from get_action_mask().

    Rather than re-running CP-SAT every step (which is too slow), the shield
    caches the pre-computed expert schedule and derives feasibility per minute
    by comparing the proposed action against what the solver recommended.
    It masks an action False only when it is *provably* unsafe:
      — a MAIN move into a block the solver marks as STOP/occupied, or
      — a move at a time the solver reserved the block for another train.

    For conflicts not covered by the cached schedule (e.g. "Chaos Monkey"
    disruptions), the existing structural checks in get_action_mask() remain
    the last safety line.
    """

    def __init__(self, expert_data: dict, train_ids: list):
        self.expert_actions = expert_data.get("expert_actions", {})
        self.schedule       = expert_data.get("schedule", {})
        self.train_ids      = train_ids
        logger.info("🛡️  FeasibilityShield initialised.")

    def check_feasibility(self, train_id: str, proposed_action: int, current_step: int) -> bool:
        """
        Returns True if the proposed action is feasible according to the
        OR-solver's pre-computed schedule, False if it is a guaranteed
        violation.

        Guaranteed violation = the solver prescribes STOP at this step, but
        the agent proposes MAIN/DIVERT into a section the solver reserved for
        another train.
        """
        if train_id not in self.expert_actions:
            return True  # No data → optimistic (let structural mask decide)

        actions = self.expert_actions[train_id]
        if current_step >= len(actions):
            return True  # Past the planned horizon → allow (RL takes over)

        expert_action = int(actions[current_step])

        # Hard block: solver said STOP and agent proposes to move into main line
        if expert_action == ACTION_STOP and proposed_action == ACTION_MAIN:
            return False

        return True

    def get_masked_actions(self, sim_time: int, current_mask: np.ndarray) -> np.ndarray:
        """
        Overlay OR-feasibility on top of the structural mask already
        computed by get_action_mask().

        Parameters
        ----------
        sim_time     : int — current environment simulation step (minutes).
        current_mask : np.ndarray [MAX_TRAINS, 3] bool — existing mask.

        Returns
        -------
        Updated mask with additional OR-solver constraints applied.
        """
        updated = current_mask.copy()

        for idx, t_id in enumerate(self.train_ids):
            for action in [ACTION_MAIN, ACTION_DIVERT]:
                if updated[idx][action]:  # Only evaluate currently-legal actions
                    if not self.check_feasibility(t_id, action, sim_time):
                        updated[idx][action] = False
                        logger.debug(
                            f"🔴 Shield blocked action={action} for {t_id} at t={sim_time}"
                        )

        return updated


# ─────────────────────────────────────────────────
# MONITORING — OR_RL Similarity Score
# ─────────────────────────────────────────────────

class ORRLMonitor:
    """
    Tracks and logs the similarity between the RL agent's chosen actions
    and the OR-solver's recommendations at each timestep.

    OR_RL_Similarity_Score = (matching actions) / (total comparable actions)
    """

    def __init__(self, expert_data: dict, train_ids: list):
        self.expert_actions  = expert_data.get("expert_actions", {})
        self.train_ids       = train_ids
        self._match_count    = 0
        self._total_count    = 0
        self._step_scores    = []

    def record_step(self, rl_actions: np.ndarray, sim_time: int):
        """
        Compare RL actions vs OR actions at this timestep.

        Parameters
        ----------
        rl_actions : np.ndarray [MAX_TRAINS] — actions chosen by the RL agent.
        sim_time   : int — current simulation step.
        """
        step_match = 0
        step_total = 0

        for idx, t_id in enumerate(self.train_ids):
            if t_id not in self.expert_actions:
                continue
            if sim_time >= len(self.expert_actions[t_id]):
                continue

            expert_act = int(self.expert_actions[t_id][sim_time])
            rl_act     = int(rl_actions[idx])

            if rl_act == expert_act:
                step_match += 1

            step_total += 1
            self._match_count += 1 if rl_act == expert_act else 0
            self._total_count += 1

        if step_total > 0:
            step_score = step_match / step_total
            self._step_scores.append(step_score)
        else:
            self._step_scores.append(None)

    @property
    def similarity_score(self) -> float:
        """Running OR_RL_Similarity_Score over the full episode."""
        if self._total_count == 0:
            return 0.0
        return self._match_count / self._total_count

    def episode_score(self) -> float:
        """Mean step similarity score for the last episode."""
        valid = [s for s in self._step_scores if s is not None]
        return float(np.mean(valid)) if valid else 0.0

    def reset(self):
        self._match_count = 0
        self._total_count = 0
        self._step_scores = []

    def log_to_tensorboard(self, writer, global_step: int):
        """Write OR_RL_Similarity_Score to a SummaryWriter if provided."""
        if writer is not None:
            writer.add_scalar(
                "OR_RL/Similarity_Score", self.similarity_score, global_step
            )
            logger.debug(
                f"[Step {global_step}] OR_RL_Similarity_Score: "
                f"{self.similarity_score:.3f}"
            )


# ─────────────────────────────────────────────────
# DATA SYNC HELPERS
# ─────────────────────────────────────────────────

def assert_temporal_alignment(env, expert_data: dict):
    """
    Validates that the OR-solver's episode horizon aligns with the RL
    environment's max episode length, preventing temporal mismatch.

    The OR solver uses TIME_HORIZON_MINUTES (120).
    The RL env terminates at last_spawn + 1500 steps (from train_env.py).
    We check that the expert action sequences don't exceed the RL horizon.
    """
    from ai.config import MAX_TRAINS_CAPACITY

    expert_acts = expert_data.get("expert_actions", {})
    max_or_len  = max((len(v) for v in expert_acts.values()), default=0)

    schedule    = env.schedule
    if schedule:
        last_spawn  = max(t["start_time"] for t in schedule.values())
        max_rl_len  = last_spawn + 1500
    else:
        max_rl_len = 1500

    logger.info(
        f"⏱  Temporal Alignment Check: OR horizon={max_or_len} steps | "
        f"RL max episode={max_rl_len} steps"
    )

    if max_or_len > max_rl_len:
        logger.warning(
            f"⚠️  OR solver horizon ({max_or_len}) exceeds RL episode length "
            f"({max_rl_len}). Steps beyond the RL horizon will be ignored."
        )
    else:
        logger.info("✅ Temporal alignment OK — OR horizon fits within RL episode.")

    obs_shape    = env.observation_space.shape
    expected_obs = (MAX_TRAINS_CAPACITY, 10)
    if obs_shape != expected_obs:
        raise ValueError(
            f"Observation space mismatch: env has {obs_shape}, "
            f"hybrid_connector expects {expected_obs}. "
            "Ensure train_env.py uses shape=(MAX_TRAINS_CAPACITY, 10)."
        )
    else:
        logger.info(f"✅ Observation vector aligned: {obs_shape}.")

    action_shape = env.action_space.nvec
    if len(action_shape) != MAX_TRAINS_CAPACITY or any(n != 3 for n in action_shape):
        raise ValueError(
            f"Action space mismatch: env has {action_shape}, "
            f"expected MultiDiscrete([3] * {MAX_TRAINS_CAPACITY})."
        )
    else:
        logger.info(f"✅ Action space aligned: MultiDiscrete([3] × {MAX_TRAINS_CAPACITY}).")


# ─────────────────────────────────────────────────
# STEP FUNCTIONS — each runs independently.
# You decide when to advance based on TensorBoard.
# ─────────────────────────────────────────────────

def _build_env_and_model(args, MODELS_DIR, LOGS_DIR):
    """
    Shared helper: build the vectorised env + load or create the MaskablePPO.
    Returns (norm_env, raw_env, model).
    """
    import gymnasium as gym
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.env_util import make_vec_env

    # Path-safe import: works when called from backend/ or backend/ai/
    try:
        from ai.train_env import TrainDispatchEnv
    except ModuleNotFoundError:
        from train_env import TrainDispatchEnv


    def mask_fn(e: gym.Env) -> np.ndarray:
        return e.get_action_mask()

    def make_env():
        e = TrainDispatchEnv()
        return ActionMasker(e, mask_fn)

    vec_env  = make_vec_env(make_env, n_envs=2)
    norm_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True,
        clip_obs=10.0, clip_reward=10.0
    )

    raw_env = TrainDispatchEnv()
    raw_env.set_difficulty(args.trains)

    load_path  = args.load
    stats_path = os.path.join(MODELS_DIR, "vec_normalize_hybrid.pkl")

    if load_path and os.path.exists(load_path):
        logger.info(f"🔄 Loading model from: {load_path}")
        if os.path.exists(stats_path):
            logger.info(f"👓 Loading normalisation stats from: {stats_path}")
            norm_env = VecNormalize.load(stats_path, vec_env)
        model = MaskablePPO.load(load_path, env=norm_env, tensorboard_log=LOGS_DIR)
    else:
        logger.info("✨ Creating fresh MaskablePPO model...")
        model = MaskablePPO(
            "MlpPolicy", norm_env,
            verbose=1,
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

    return norm_env, raw_env, model


# ────────────────────────────────────────────────────────────────────────────
# STEP 1 — Behaviour Cloning warm-up
#   Channel A: ON  ✅  |  Channel B: OFF ❌
#   Run this first. Check TensorBoard → when BC loss stabilises,
#   manually run Step 2.
# ────────────────────────────────────────────────────────────────────────────

def run_step1_bc_warmup(args, MODELS_DIR, LOGS_DIR):
    """
    Step 1: Behaviour Cloning from expert_actions.json.

    Reads the OR-solver schedule, builds a supervised dataset, and
    updates the MaskablePPO policy weights via Cross-Entropy Loss so the
    agent's initial distribution mirrors the expert before any PPO rollout.

    Output
    ------
    Saves: hybrid_step1_BC_warmup.zip  (warmed-up policy)
           vec_normalize_hybrid.pkl    (normalisation stats)

    When to move to Step 2
    ----------------------
    Open TensorBoard, watch 'train/loss'. When the BC loss flattens
    (no improvement over ~3 epochs), it's ready. Run --step 2.
    """
    from ai.config import MAX_TRAINS_CAPACITY

    logger.info("=" * 60)
    logger.info("STEP 1 ▶  Behaviour Cloning Warm-up")
    logger.info("  Channel A: ON ✅   Channel B: OFF ❌")
    logger.info("=" * 60)

    norm_env, raw_env, model = _build_env_and_model(args, MODELS_DIR, LOGS_DIR)
    expert_data = load_expert_data(args.expert)
    assert_temporal_alignment(raw_env, expert_data)

    model = pretrain_from_expert(
        model,
        raw_env,
        expert_data_path=args.expert,
        max_trains=MAX_TRAINS_CAPACITY,
        bc_epochs=args.bc_epochs,
        lr=args.lr,
        bc_loss_weight=1.0
    )

    save_path  = os.path.join(MODELS_DIR, "hybrid_step1_BC_warmup.zip")
    stats_path = os.path.join(MODELS_DIR, "vec_normalize_hybrid.pkl")
    model.save(save_path)
    norm_env.save(stats_path)

    logger.info(f"💾 Model  → {save_path}")
    logger.info(f"👓 Stats  → {stats_path}")
    logger.info("")
    logger.info("✅ Step 1 complete.")
    logger.info("   Monitor TensorBoard. When BC loss flattens → run --step 2")
    logger.info(f"   Command: python hybrid_connector.py --step 2 --load {save_path}")


# ────────────────────────────────────────────────────────────────────────────
# STEP 2 — Masked PPO + Chaos Monkey (Resilience Training)
#   Channel A: OFF ❌  |  Channel B: ON ✅  |  Chaos: ON 🐒
#   Run this after Step 1 when you're satisfied with BC loss.
#   Check TensorBoard → when ep_rew_mean stabilises, run Step 3.
# ────────────────────────────────────────────────────────────────────────────

def run_step2_masked_ppo(args, MODELS_DIR, LOGS_DIR):
    """

    Step 2: Masked PPO Exploration with OR Safety Shield + Chaos Monkey.

    Three things happen simultaneously:
      1. FeasibilityShield (Channel B) is attached — blocks provably unsafe moves.
      2. Chaos Monkey is enabled — each episode injects random disruptions:
           • 30% per-train chance of 1–10 min late start
           • One random train gets its speed reduced to 80%
      3. The agent must learn to recover from disruptions using the PPO reward signal.

    Expected TensorBoard signals:
      • rollout/ep_rew_mean  : dips initially (agent unlearning static BC paths),
                               then recovers as it finds Dynamic Precedence.
      • train/explained_variance : should stay > 0.8. A sharp drop means the
                               value network is confused by new disruptions.
      • OR_RL/Similarity_Score   : will drop below 1.0 (agent deviates from
                               golden schedule to handle chaos — this is good).

    Parameters used
    ---------------
    --steps              : total PPO timesteps for this phase
    --chaos-delay-prob   : per-train late-start probability (default 0.30)
    --chaos-delay-max    : max late-start minutes (default 10)
    --chaos-speed-factor : speed snag factor (default 0.80)

    Output
    ------
    Saves: hybrid_step2_PPO_explore.zip
           vec_normalize_hybrid.pkl  (updated stats)

    When to move to Step 3
    ----------------------
    Watch rollout/ep_rew_mean. When it stops growing AND the agent
    handles most chaos episodes without collision, run --step 3.
    """
    logger.info("=" * 60)
    logger.info("STEP 2 ▶  Masked PPO + Chaos Monkey (Resilience Training)")
    logger.info("  Channel A: OFF ❌   Channel B: ON ✅   Chaos: ON 🐒")
    logger.info("=" * 60)

    norm_env, raw_env, model = _build_env_and_model(args, MODELS_DIR, LOGS_DIR)
    expert_data = load_expert_data(args.expert)
    train_ids   = [t["id"] for t in raw_env.trains]

    # ── CHANNEL B: ON — Attach the FeasibilityShield ───────────────────
    shield = FeasibilityShield(expert_data, train_ids)
    raw_env.attach_feasibility_shield(shield)
    try:
        model.env.env_method("attach_feasibility_shield", shield)
        logger.info("🛡️  Shield attached to all VecEnv sub-environments.")
    except Exception as e:
        logger.warning(f"⚠️  VecEnv shield propagation failed: {e}")

    # ── CHAOS MONKEY: ON — Inject stochastic disruptions per episode ────
    chaos_delay_prob   = getattr(args, "chaos_delay_prob",   0.30)
    chaos_delay_max    = getattr(args, "chaos_delay_max",    10)
    chaos_speed_factor = getattr(args, "chaos_speed_factor", 0.80)

    raw_env.set_chaos_mode(
        enabled      = True,
        delay_prob   = chaos_delay_prob,
        delay_min    = 1,
        delay_max    = chaos_delay_max,
        speed_snag   = True,
        speed_factor = chaos_speed_factor,
    )
    try:
        model.env.env_method(
            "set_chaos_mode",
            True,                  # enabled
            chaos_delay_prob,
            1,                     # delay_min
            chaos_delay_max,
            True,                  # speed_snag
            chaos_speed_factor,
        )
        logger.info(
            f"🐒 Chaos Monkey enabled on all VecEnv sub-environments | "
            f"delay_prob={chaos_delay_prob:.0%} "
            f"delay_max={chaos_delay_max}m "
            f"speed_factor={chaos_speed_factor:.0%}"
        )
    except Exception as e:
        logger.warning(f"⚠️  VecEnv chaos propagation failed: {e}")

    # ── Similarity Monitor ───────────────────────────────────────────
    monitor = ORRLMonitor(expert_data, train_ids)

    logger.info(
        f"🚀 Running {args.steps:,} PPO steps — shield + chaos active..."
    )
    model.learn(
        total_timesteps=args.steps,
        reset_num_timesteps=False,
        tb_log_name="Hybrid_Step2_ChaosPPO"
    )

    save_path  = os.path.join(MODELS_DIR, "hybrid_step2_PPO_explore.zip")
    stats_path = os.path.join(MODELS_DIR, "vec_normalize_hybrid.pkl")
    model.save(save_path)
    model.env.save(stats_path)

    logger.info(f"💾 Model  → {save_path}")
    logger.info(f"👓 Stats  → {stats_path}")
    logger.info(f"📈 OR_RL_Similarity_Score: {monitor.similarity_score:.3f}")
    logger.info("")
    logger.info("✅ Step 2 complete.")
    logger.info("   Watch TensorBoard: ep_rew_mean should recover after dip.")
    logger.info("   When reward stabilises → run --step 3 for pure fine-tuning.")
    logger.info(f"   Command: python hybrid_connector.py --step 3 --load {save_path}")


# ────────────────────────────────────────────────────────────────────────────
# STEP 3 — Pure RL Fine-Tuning (Both channels OFF)
#   Channel A: OFF ❌  |  Channel B: OFF ❌  |  Chaos: OFF ✅
#   Run this after Step 2 when the agent handles nominal traffic well
#   and you want it to optimise for maximum weighted-delay reward.
# ────────────────────────────────────────────────────────────────────────────

def run_step3_pure_rl(args, MODELS_DIR, LOGS_DIR):
    """
    Step 3: Pure PPO fine-tuning — zero OR supervision, chaos OFF.

    Loads the Step 2 model. Detaches the FeasibilityShield AND disables
    Chaos Monkey so the agent fine-tunes on a stable environment.
    Only structural safety rules in get_action_mask() remain.

    Parameters used
    ---------------
    --steps : total PPO timesteps for fine-tuning

    Output
    ------
    Saves: hybrid_step3_FINAL.zip
           vec_normalize_hybrid.pkl  (final stats)
    """
    logger.info("=" * 60)
    logger.info("STEP 3 ▶  Pure RL Fine-Tuning (zero OR + chaos pressure)")
    logger.info("  Channel A: OFF ❌   Channel B: OFF ❌   Chaos: OFF ✅")
    logger.info("=" * 60)

    norm_env, raw_env, model = _build_env_and_model(args, MODELS_DIR, LOGS_DIR)

    # ── CHANNEL B: OFF + CHAOS: OFF ──────────────────────────────────
    raw_env.attach_feasibility_shield(None)
    raw_env.set_chaos_mode(enabled=False)
    try:
        model.env.env_method("attach_feasibility_shield", None)
        model.env.env_method("set_chaos_mode", False)
        logger.info("🔓 Shield detached + Chaos disabled on all VecEnv sub-environments.")
    except Exception as e:
        logger.warning(f"⚠️  VecEnv detach/disable failed: {e}")

    logger.info(f"🚀 Running {args.steps:,} fine-tuning steps...")
    model.learn(
        total_timesteps=args.steps,
        reset_num_timesteps=False,
        tb_log_name="Hybrid_Step3_FinetuneRL"
    )

    save_path  = os.path.join(MODELS_DIR, "hybrid_step3_FINAL.zip")
    stats_path = os.path.join(MODELS_DIR, "vec_normalize_hybrid.pkl")
    model.save(save_path)
    model.env.save(stats_path)

    logger.info(f"💾 Final model → {save_path}")
    logger.info(f"👓 Stats       → {stats_path}")
    logger.info("")
    logger.info("🏁 Step 3 complete. Training pipeline finished.")


# ─────────────────────────────────────────────────
# STAND-ALONE CLI ENTRY POINT
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    LOGS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    parser = argparse.ArgumentParser(
        description="RailMind Hybrid RL-OR Pipeline — manual step control",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Step progression (you control when to advance):

  Step 1 — Behaviour Cloning warm-up
    python hybrid_connector.py --step 1 --expert expert_actions.json --trains 5

  Step 2 — Masked PPO with OR Safety Shield  (run when BC loss plateaus)
    python hybrid_connector.py --step 2 --load models/hybrid_step1_BC_warmup.zip --steps 200000

  Step 3 — Pure RL fine-tuning              (run when ep_rew_mean plateaus)
    python hybrid_connector.py --step 3 --load models/hybrid_step2_PPO_explore.zip --steps 300000
"""
    )

    parser.add_argument(
        "--step", type=int, required=True, choices=[1, 2, 3],
        help="Which step to run: 1=BC Warmup, 2=Masked PPO, 3=Pure RL Finetune"
    )
    parser.add_argument(
        "--expert", default="expert_actions.json",
        help="Path to expert_actions.json (required for steps 1 & 2)"
    )
    parser.add_argument(
        "--load", default=None,
        help="Path to a MaskablePPO .zip to load (required for steps 2 & 3)"
    )
    parser.add_argument(
        "--trains", type=int, default=5,
        help="Number of trains for environment difficulty"
    )
    parser.add_argument(
        "--bc-epochs", type=int, default=15,
        help="[Step 1] Number of Behaviour Cloning epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="[Step 1] Learning rate for BC optimizer"
    )
    parser.add_argument(
        "--steps", type=int, default=200_000,
        help="[Steps 2 & 3] Number of PPO timesteps"
    )
    # ── Chaos Monkey (Step 2 only) ─────────────────────────────────────────
    parser.add_argument(
        "--chaos-delay-prob", type=float, default=0.30,
        dest="chaos_delay_prob",
        help="[Step 2] Per-train probability of late start (default 0.30)"
    )
    parser.add_argument(
        "--chaos-delay-max", type=int, default=10,
        dest="chaos_delay_max",
        help="[Step 2] Max late-start minutes (default 10)"
    )
    parser.add_argument(
        "--chaos-speed-factor", type=float, default=0.80,
        dest="chaos_speed_factor",
        help="[Step 2] Speed snag as fraction of max_speed (default 0.80)"
    )

    args = parser.parse_args()

    # ── Pre-flight checks ─────────────────────────────────────────────────
    if args.step in [2, 3] and not args.load:
        parser.error(
            f"--step {args.step} requires --load <path_to_model.zip>\n"
            f"  Example: --load models/hybrid_step{args.step - 1}_*.zip"
        )

    if args.step in [1, 2] and not os.path.exists(args.expert):
        parser.error(
            f"expert_actions.json not found at '{args.expert}'. "
            "Run or_solver.py first."
        )

    # ── Dispatch ──────────────────────────────────────────────────────────
    if args.step == 1:
        run_step1_bc_warmup(args, MODELS_DIR, LOGS_DIR)

    elif args.step == 2:
        run_step2_masked_ppo(args, MODELS_DIR, LOGS_DIR)

    elif args.step == 3:
        run_step3_pure_rl(args, MODELS_DIR, LOGS_DIR)

