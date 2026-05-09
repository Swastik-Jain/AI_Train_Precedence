# backend/scripts/run_ai_simulation.py
import time
import sys
import os
import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import SessionLocal, init_db, TrainPosition
from ai.train_env import TrainDispatchEnv
from ai.config import TRAIN_CONFIG, SCHEDULE

# Point to backend/ai/models/ppo_train_dispatcher.zip
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "backend", "ai", "models", "ppo_train_dispatcher.zip")

def run_ai_simulation():
    init_db()
    db = SessionLocal()
    
    print("--- 🕰️ Initializing Real-Time Railway Twin ---")
    env = TrainDispatchEnv()
    obs, _ = env.reset()
    
    try:
        model = PPO.load(MODEL_PATH)
        print("--- 🧠 AI Brain Loaded ---")
        
        step_counter = 0
        while True:
            step_counter += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, _, _ = env.step(action)

            print(f"\n⏱️  SIMULATION TIME: T={step_counter} min")
            
            for i, t in enumerate(TRAIN_CONFIG):
                train_id = t['id']
                schedule = SCHEDULE[train_id]
                pos = int(obs[i*3])
                speed = int(obs[i*3+1])
                
                # Visuals
                status_icon = "🟢"
                loc_str = f"Sec {pos}"
                
                if pos == 0:
                    loc_str = "🛑 VIRTUAL YARD"
                    status_icon = "⏳"
                elif pos == 999:
                    loc_str = "✅ ARRIVED"
                    status_icon = "🏁"
                elif pos == 104:
                    loc_str = "🔄 LOOP (104)"
                    status_icon = "⚠️"
                
                punctuality = "ON TIME"
                if step_counter > schedule['deadline'] and pos != 999:
                    punctuality = f"🚨 LATE (+{step_counter - schedule['deadline']} min)"
                    status_icon = "🔻"

                print(f"{status_icon} {train_id}: {loc_str} | {punctuality}")

                # DB Update
                db_status = "RUNNING"
                if pos == 0: db_status = "SCHEDULED"
                if pos == 999: db_status = "FINISHED"
                
                db.add(TrainPosition(
                    train_id=train_id,
                    section=str(pos),
                    speed_kmh=speed,
                    status=db_status
                ))

            db.commit()
            
            if terminated:
                print(f"\n🏁 SCENARIO END. Reward: {reward}\nResetting...")
                obs, _ = env.reset()
                step_counter = 0
                time.sleep(3)
            
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        db.close()

if __name__ == "__main__":
    run_ai_simulation()