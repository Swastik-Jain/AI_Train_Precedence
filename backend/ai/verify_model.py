
import os
from stable_baselines3 import PPO
from train_env import TrainDispatchEnv

def verify_level_4_5():
    print("--- 🕵️‍♀️ VERIFYING LEVEL 4.5 MODEL ---")
    
    # 1. Load Environment
    env = TrainDispatchEnv()
    env.set_difficulty(10) # 10 Trains (Level 1.5)
    
    # 2. Load Model
    model_path = "models/ppo_phase3_L1.5_Manual.zip"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    model = PPO.load(model_path)
    print(f"✅ Loaded Model: {model_path}")
    
    # 3. Run Episode
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    print("🚀 Starting Simulation (Max 300 steps)...")
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Check specific termination conditions based on reward/state
        # We can't see internal state easily without accessing env directly, 
        # but we can infer from reward spikes.
        
        if reward < -500:
            print(f"💥 CRASH DETECTED at Step {steps} (Reward: {reward})")
            
        if reward > 1000 and reward < 3000:
             print(f"🚆 Train Finished at Step {steps}")
             
        if reward > 4000:
            print(f"🏆 ALL TRAINS FINISHED at Step {steps}")

    print("-" * 30)
    print(f"🏁 Episode Ended at Step {steps}")
    print(f"💰 Total Reward: {total_reward}")
    
    # Diagnostics
    if steps >= 300:
        print("result: TIMEOUT (Traffic Jam?)")
    elif total_reward < -500: # Heuristic
        print("result: CRASH (System Failure)")
    elif total_reward > 10000:
        print("result: SUCCESS (Likely)")
    else:
        print("result: UNKNOWN")

if __name__ == "__main__":
    verify_level_4_5()
