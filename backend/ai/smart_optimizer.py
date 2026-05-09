import numpy as np
from typing import Optional

class SmartOptimizer:
    """
    Interlocking System (Safety Layer).
    Map-aware to handle switches, loops, and anti-loitering.
    """

    def __init__(self):
        pass 

    def optimize_decision(self, trains, ai_actions, track_map):
        """
        Scans the AI's proposed actions and overrides them if they act unsafely
        or if they stall unnecessarily on the main line.
        """
        safe_actions = ai_actions.copy()
        
        # 1. Map where everyone is RIGHT NOW and count occupancy
        current_occupancy = {}
        for t in trains:
            if t['position'] not in [0, 999]:
                current_occupancy[t['position']] = current_occupancy.get(t['position'], 0) + 1
        
        # 2. Track claimed spots for this timestep (First-Come-First-Served)
        claimed_occupancy = {}

        # Sort indices by priority (highest first) so high priority trains claim capacity first
        sorted_indices = sorted(range(len(trains)), key=lambda idx: trains[idx].get('priority', 0), reverse=True)

        for i in sorted_indices:
            train = trains[i]
            current_pos = train['position']
            proposed_action = safe_actions[i]
            
            # Skip finished trains entirely
            if current_pos == 999:
                continue
                
            node_data = track_map.get(current_pos, {})
            next_opts = node_data.get('next', [])

            # --- Layer 1: Intelligent Overrides ---
            if current_pos != 0 and next_opts:
                
                # --- Anti-Loitering ---
                if proposed_action == 0:
                    target_node = next_opts[0]
                    target_cap = track_map.get(target_node, {}).get('capacity', 1)
                    occ_ahead = current_occupancy.get(target_node, 0) + claimed_occupancy.get(target_node, 0)
                    
                    # 🚨 X-RAY DEBUG: Uncomment the line below if a train deadlocks on the highway
                    # print(f"🔍 DEBUG: Train {train['id']} at Pos {current_pos}. Target {target_node} Occ: {occ_ahead}/{target_cap}")
                    
                    # If the track immediately ahead is empty...
                    if occ_ahead < target_cap:
                        
                        # 1. Highway Prod: Do not allow trains to stop on the mainline.
                        if int(current_pos) < 20000: 
                            proposed_action = 1
                            safe_actions[i] = 1
                            
                        # 2. Loop "Wake Up Call": Push trains out of loops if the exit is completely clear.
                        elif int(current_pos) >= 20000:
                            # Verify no other train is claiming the exit switch right this second
                            if claimed_occupancy.get(target_node, 0) == 0:
                                proposed_action = 1
                                safe_actions[i] = 1

                # --- Divert Fallback ---
                elif proposed_action == 2:
                    loop_found = False
                    if len(next_opts) > 1:
                        for loop_node in next_opts[1:]:
                            occ = current_occupancy.get(loop_node, 0) + claimed_occupancy.get(loop_node, 0)
                            cap = track_map.get(loop_node, {}).get('capacity', 1)
                            if occ < cap:
                                loop_found = True
                                break
                    if not loop_found:
                        # Loops are full or don't exist here. Force MAIN instead of failing!
                        proposed_action = 1
                        safe_actions[i] = 1


            # --- Layer 2: Collision Prevention ---
            
            # Ensure yard trains don't spawn into a blocked Node 1
            if current_pos == 0:
                if proposed_action > 0:
                    target_pos = 1
                else:
                    continue
            else:
                # --- Predict Future Position ---
                if proposed_action == 0:
                    target_pos = current_pos
                else:
                    if not next_opts: 
                        target_pos = 999 # End of line
                    else:
                        if proposed_action == 1:
                            target_pos = next_opts[0] # MAIN
                        elif proposed_action == 2:
                            # Try to find a free loop
                            target_pos = next_opts[0] # Default to main if no loop
                            if len(next_opts) > 1:
                                for platform_node in next_opts[1:]:
                                    occ = current_occupancy.get(platform_node, 0) + claimed_occupancy.get(platform_node, 0)
                                    cap = track_map.get(platform_node, {}).get('capacity', 1)
                                    if occ < cap:
                                        target_pos = platform_node
                                        break

            # --- Safety Checks ---
            # Rule 1: Node Capacity
            target_cap = track_map.get(target_pos, {}).get('capacity', 1) if track_map else 1
            if target_pos in [0, 999]:
                target_cap = 999 # Infinite capacity for Yard/Dest
            
            current_occ = current_occupancy.get(target_pos, 0)
            incoming_occ = claimed_occupancy.get(target_pos, 0)
            
            # If the target is where I currently am, I can always stay
            if target_pos == current_pos:
                claimed_occupancy[current_pos] = claimed_occupancy.get(current_pos, 0) + 1
            else:
                total_target_occ = current_occ + incoming_occ
                
                if total_target_occ >= target_cap:
                    # Unsafe action: override with Stop
                    safe_actions[i] = 0
                    claimed_occupancy[current_pos] = claimed_occupancy.get(current_pos, 0) + 1
                else:
                    claimed_occupancy[target_pos] = claimed_occupancy.get(target_pos, 0) + 1

        return safe_actions

    # ---------------------------------------------------------------------------
    # OR-Shield: API-level hard-constraint gate for copilot suggestions
    # ---------------------------------------------------------------------------
    def or_shield_check(
        self,
        suggestion: dict,
        train_states: dict,
        active_blocks: Optional[dict] = None,
        dynamic_constraints: Optional[dict] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Validates a high-level API suggestion dict against the live TRAIN_STATES
        snapshot before queuing it for the controller or executing a commit.

        Hard constraints checked (in priority order):
          1. Target train must exist in the live simulation.
          2. Target train must not have already reached its destination.
          3. At least one affected edge must not already be under a TOTAL_BLOCK
             maintenance window (if active_blocks dict is provided).
          4. No affected edge may be occupied by 2 or more trains simultaneously
             (Absolute Block System — one train per block at a time).
          5. Must not violate any dynamic capacity or speed limits from Sandbox.

        Args:
            suggestion          : The AISuggestion dict from _make_suggestion() / RL model.
            train_states        : The live TRAIN_STATES dict  { train_id -> state dict }.
            active_blocks       : Optional ACTIVE_BLOCKS dict { element_id -> block dict }.
            dynamic_constraints : Optional DYNAMIC_CONSTRAINTS dict.

        Returns:
            (True, None)          — safe, may forward to controller.
            (False, reason_str)   — hard constraint violated, must be dropped.
        """
        target_id    = suggestion.get("target_train_id", "")
        affected     = suggestion.get("affected_edges", [])
        active_blocks = active_blocks or {}
        dynamic_constraints = dynamic_constraints or {}

        # ── Constraint 1: Train must exist in live sim ───────────────────────
        if target_id not in train_states:
            return False, (
                f"Train '{target_id}' not found in live simulation — "
                "may have been removed or not yet spawned"
            )

        train = train_states[target_id]

        # ── Constraint 2: Train must not already be finished / blocked ───────
        if train.get("status") in ("Finished", "Arrived"):
            return False, (
                f"Train '{target_id}' has already reached its destination — "
                "action would be a no-op"
            )

        # ── Constraint 3: Affected edges must not be under TOTAL_BLOCK ───────
        for edge_id in affected:
            blk = active_blocks.get(edge_id)
            if blk and blk.get("severity") == "TOTAL_BLOCK":
                return False, (
                    f"MaintenanceBlock: edge '{edge_id}' is under an active "
                    "TOTAL_BLOCK maintenance window — routing through this "
                    "segment is forbidden"
                )

        # ── Constraint 4: Absolute block — max 1 train per edge ─────────────
        edge_occupancy: dict[str, int] = {}
        for t in train_states.values():
            eid = t.get("edge_id")
            if eid:
                edge_occupancy[eid] = edge_occupancy.get(eid, 0) + 1

        for edge_id in affected:
            occ = edge_occupancy.get(edge_id, 0)
            if occ >= 2:
                return False, (
                    f"AbsoluteBlock violation: edge '{edge_id}' already "
                    f"occupied by {occ} trains — headway constraint violated"
                )

        # ── Constraint 5: Dynamic constraints (Sandbox Sandbox) ───────────────
        for edge_id in affected:
            for c_id, c_data in dynamic_constraints.items():
                if c_data.get("edge_id") == edge_id:
                    if c_data.get("type") == "CAPACITY_LIMIT":
                        cap = c_data.get("value", 1)
                        if edge_occupancy.get(edge_id, 0) >= cap:
                            return False, (
                                f"DynamicConstraint violation: edge '{edge_id}' has a strict capacity "
                                f"limit of {cap} applied via simulation sandbox."
                            )

        # ── All checks passed ────────────────────────────────────────────────
        return True, None