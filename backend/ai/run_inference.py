import time
import sys
import os
import argparse
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import SessionLocal, init_db, TrainPosition
from ai.train_env import TrainDispatchEnv


def mask_fn(env):
    return env.get_action_mask()


def run_inference_service():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="models/ppo_phase3_L2_7Trains_final.zip",
                        help="Path to trained MaskablePPO model")
    parser.add_argument("--stats", type=str,
                        default="models/vec_normalize_L2_7Trains.pkl",
                        help="Path to VecNormalize stats")
    parser.add_argument("--trains", type=int, default=7,
                        help="Number of trains to simulate")
    args = parser.parse_args()

    # ── 1. Database ───────────────────────────────────────────────────────────
    init_db()
    db = SessionLocal()
    print("--- 🔌 Database Connected ---")

    # ── 2. Environment — MUST match training setup exactly ───────────────────
    print(f"--- 🚄 Initializing Railway Digital Twin ({args.trains} Trains) ---")

    raw = DummyVecEnv([lambda: ActionMasker(TrainDispatchEnv(), mask_fn)])

    if not os.path.exists(args.stats):
        print(f"❌ CRITICAL: VecNormalize stats not found at {args.stats}")
        print("   Make sure you point --stats to the .pkl saved during training.")
        return

    env = VecNormalize.load(args.stats, raw)
    env.training = False      # do NOT update running mean/var at inference
    env.norm_reward = False   # do NOT normalize rewards at inference
    env.env_method("set_difficulty", args.trains)

    # Inner env reference — used to read train positions / speeds directly
    inner_env = env.envs[0].env

    obs = env.reset()

    # ── 3. Model loading ─────────────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"❌ CRITICAL: Model file not found at {args.model}")
        return

    try:
        model = MaskablePPO.load(args.model, device="cpu")
        print(f"--- 🧠 AI Brain Loaded: {os.path.basename(args.model)} ---")
        print(f"--- 📅 Active Schedule: {len(inner_env.schedule)} Trains ---")
        print("-" * 50)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error loading model: {e}")
        return

    # ── 4. Simulation loop ───────────────────────────────────────────────────
    step_counter = 0

    try:
        while True:
            step_counter += 1

            # --- AI DECISION with action mask ---
            action_masks = np.array(env.env_method("get_action_mask"))
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=action_masks
            )
            safe_actions = action  # kept for forensics printout below

            # --- EXECUTE PHYSICS ---
            obs, reward, terminated, truncated = env.step(action)[:4]

            # --- LOGGING ---
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"⏱️  SIMULATION TIME: T={step_counter} min | "
                  f"Active Trains: {args.trains}")
            print(f"🧠 Current Step Reward: {float(reward):.2f}")

            # Write every train's state to the database
            for i, t in enumerate(inner_env.trains):
                train_id = t['id']
                pos   = int(t['position'])
                speed = int(t['speed'])

                db_status = "RUNNING"
                if pos == 0:                              db_status = "SCHEDULED"
                elif pos == 999:                          db_status = "FINISHED"
                elif speed == 0 and pos not in [0, 999]: db_status = "WAITING"

                db.add(TrainPosition(
                    train_id=train_id,
                    section=str(pos),
                    speed_kmh=speed,
                    status=db_status
                ))
            db.commit()

            # --- VISUAL FEEDBACK (Terminal) ---
            moving  = sum(1 for t in inner_env.trains
                          if t['speed'] > 0 and t['position'] not in [0, 999])
            # Exclude Node 1 (spawn gate) — holding there on red is NOT a deadlock
            waiting = sum(1 for t in inner_env.trains
                          if t['speed'] == 0 and t['position'] not in [0, 1, 999])
            finished = sum(1 for t in inner_env.trains if t['position'] == 999)

            print(f"🟢 Moving: {moving} | 🔴 Waiting: {waiting} | "
                  f"🏁 Finished: {finished}")

            # 🚨 GRIDLOCK DETECTION ───────────────────────────────────────────
            if moving == 0 and waiting > 0 and step_counter > 50:
                print("\n" + "=" * 40)
                print("🚨 TOTAL NETWORK GRIDLOCK DETECTED! 🚨")
                print("=" * 40)

                node_1_occupancy = sum(
                    1 for t in inner_env.trains if t['position'] == 1)
                node_1_cap = inner_env.track_map.get(1, {}).get('capacity', 2)
                print(f"🚪 Node 1 (Spawn Gate): "
                      f"{node_1_occupancy} / {node_1_cap} trains")

                print("\n🚂 --- TRAIN FORENSICS ---")
                for i, t in enumerate(inner_env.trains):
                    if t['position'] not in [0, 999]:
                        ai_intent = action[0][i]   # action is (1, N) from vecenv
                        final_act = safe_actions[0][i]

                        intent_str = ("STOP" if ai_intent == 0
                                      else ("MAIN" if ai_intent == 1 else "DIVERT"))
                        final_str  = ("STOP" if final_act == 0
                                      else ("MAIN" if final_act == 1 else "DIVERT"))

                        print(f"Train {t['id']} | Pos: {t['position']} | "
                              f"AI: {intent_str} -> Final: {final_str}")
                print("=" * 40)
                print("🔄 Auto-resetting episode...")
                time.sleep(2)
                obs = env.reset()
                step_counter = 0
                continue

            # --- END OF EPISODE ──────────────────────────────────────────────
            if terminated or truncated:
                print("\n" + "=" * 30)
                finished_count = sum(
                    1 for t in inner_env.trains if t['position'] == 999)
                total = len(inner_env.trains)

                if float(reward) <= -74.0:   # collision penalty is -75.0
                    print(f"💥 EPISODE CRASHED. "
                          f"Finished: {finished_count}/{total} | "
                          f"Reward: {float(reward):.2f}")
                elif finished_count == total:
                    print(f"🏆 DAY COMPLETE! All {total} trains finished! | "
                          f"Reward: {float(reward):.2f}")
                else:
                    print(f"⏱️ TIMEOUT. "
                          f"Finished: {finished_count}/{total} | "
                          f"Reward: {float(reward):.2f}")

                print("=" * 30 + "\n🔄 Resetting...")
                obs = env.reset()
                step_counter = 0
                time.sleep(2)

            # Simulation speed — adjust to taste
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by User.")
        db.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error: {e}")
        db.close()


if __name__ == "__main__":
    run_inference_service()