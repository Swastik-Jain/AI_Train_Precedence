import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_env import TrainDispatchEnv

def run_verification():
    env = TrainDispatchEnv()
    obs = env.reset()
    
    prev_state = {}
    num_trains = len(env.trains)
    
    for tick in range(150):
        action = [1] * num_trains
        try:
            result = env.step(action)
        except Exception as e:
            print("Action format error:", e)
            break
            
        trains = env.trains
        for train in trains:
            tid = train['id']
            curr_committed = train.get('committed_next_node')
            curr_pos = train.get('position')
            
            if tid in prev_state:
                prev_committed = prev_state[tid]['committed_next_node']
                prev_pos = prev_state[tid]['position']
                
                if curr_committed != prev_committed:
                    print(f"Tick {tick}, Train {tid}: committed_next_node changed {prev_committed} -> {curr_committed}")
                    if curr_pos == prev_pos:
                        print(f"  ERROR: position did NOT change! (stayed at {curr_pos})")
                    else:
                        print(f"  OK: position changed {prev_pos} -> {curr_pos}")
            
            prev_state[tid] = {
                'committed_next_node': curr_committed,
                'position': curr_pos
            }
        
if __name__ == '__main__':
    run_verification()
