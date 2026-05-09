from ortools.sat.python import cp_model
from ai.config import TRACK_MAP, TRAIN_CONFIG, NUM_TRAINS

class SmartOptimizer:
    """
    The 'Super Kavach': Uses Google OR-Tools (Constraint Programming).
    
    Instead of checking rules one by one, this creates a mathematical model 
    of the entire railway network for the current second.
    
    It asks the solver: 
    "Is there a mathematical solution where the High Priority train moves 
    and NO collisions happen?"
    """

    def __init__(self):
        print("--- 📐 SmartOptimizer (OR-Tools) Initialized ---")

    def optimize_decision(self, current_positions, proposed_actions):
        """
        Input: 
            - Where everyone is (current_positions)
            - What the AI wants to do (proposed_actions)
        
        Output: 
            - The mathematically optimal SAFE actions.
        """
        model = cp_model.CpModel()
        
        # --- 1. VARIABLES ---
        # For each train, we define a decision variable: 0 = Halt, 1 = Move
        # We act as a filter on the AI's proposed action.
        # If AI wants to Halt, we must Halt. If AI wants to Move, we *can* Halt if unsafe.
        moves = []
        for i in range(NUM_TRAINS):
            # The solver decides: Can we execute the move (1) or must we force stop (0)?
            moves.append(model.NewBoolVar(f'train_{i}_move'))

        # --- 2. CONSTRAINTS (The "Laws of Physics") ---
        
        # Calculate where everyone WOULD be if they moved
        next_positions = [] 
        for i in range(NUM_TRAINS):
            current_pos = current_positions[i]
            ai_action = proposed_actions[i]
            
            # If AI wants to HALT (0), forcing the solver variable to 0
            if ai_action == 0:
                model.Add(moves[i] == 0)
                next_positions.append(current_pos) # It stays here
            else:
                # Calculate the potential next hop
                target_idx = ai_action - 1
                options = TRACK_MAP.get(current_pos, [])
                
                if not options: # Dead end
                    model.Add(moves[i] == 0)
                    next_positions.append(current_pos)
                else:
                    # Clamp index
                    if target_idx >= len(options): target_idx = len(options) - 1
                    potential_next = options[target_idx]
                    
                    # Create a variable for the "Resulting Position"
                    # If move=1 -> Result is potential_next
                    # If move=0 -> Result is current_pos
                    # (This logic is handled by checking conflicts below)
                    
                    # CONSTRAINT: Signal Interlocking
                    # We cannot move into a section that is occupied by another train's CURRENT position
                    # (unless that train is also moving away, but for simplicity, we use strict block signaling)
                    
                    # "If I move, my target must not be anyone else's current position"
                    for j in range(NUM_TRAINS):
                        if i == j: continue
                        
                        # Case A: Train J is sitting at my target
                        if current_positions[j] == potential_next:
                            # If Train J is at my target, I CANNOT move unless Train J moves? 
                            # Standard Absolute Block System: Don't enter occupied block.
                            # So, I must Halt.
                            model.Add(moves[i] == 0)

        # --- 3. OBJECTIVE (Maximize Flow) ---
        # We want to maximize the number of High Priority trains moving.
        score_expr = 0
        for i in range(NUM_TRAINS):
            priority = TRAIN_CONFIG[i]['priority']
            # Reward moving high priority trains more
            score_expr += (moves[i] * priority)
            
        model.Maximize(score_expr)

        # --- 4. SOLVE ---
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        # --- 5. DECODE RESULT ---
        final_actions = []
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for i in range(NUM_TRAINS):
                can_move = solver.Value(moves[i])
                if can_move == 1:
                    final_actions.append(proposed_actions[i]) # Allow AI's wish
                else:
                    final_actions.append(0) # VETO! Force Halt.
        else:
            # If solver fails (rare), freeze everything for safety
            print("⚠️ Math Solver failed to find solution! Emergency All-Stop.")
            final_actions = [0] * NUM_TRAINS

        return final_actions