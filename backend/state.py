from typing import Dict, Any, List, Tuple, Set
import asyncio
from topology import get_network_topology
from or_tools.smart_optimizer import SmartOptimizer
from config import DEFAULT_TICK_INTERVAL_S, OVERRIDE_TICKS

class SimulationState:
    def __init__(self):
        # Topology
        self.topology_data = get_network_topology()
        self.network_topology = {"nodes": self.topology_data["nodes"], "edges": self.topology_data["edges"]}
        self.raw_track_map = self.topology_data["raw"]["track_map"]
        
        # Simulation State
        self.train_states: Dict[str, Any] = {}
        self.inference_train_ids: List[str] = []
        
        # Websockets
        self.active_websockets: Set[Any] = set()
        self.copilot_websockets: Set[Any] = set()
        
        # System Overrides
        self.system_lockdown: bool = False
        self.or_shield_enabled: bool = True
        self.ai_auto_commit: bool = False
        self.consecutive_inference_errors: int = 0
        self.autopilot_mode: bool = True
        self.explain_before_act_mode: bool = False
        
        # Simulation Control
        self.sim_tick: int = 0
        self.inference_sim_time: int = 0  # RL env's episodic time (resets each episode, unlike sim_tick)
        self.tick_interval_s: float = DEFAULT_TICK_INTERVAL_S
        self.is_sim_running: bool = True
        self.last_punctuality: float = 100.0
        self.sim_lock = asyncio.Lock()
        self.shutdown_inference_flag: bool = False
        
        # Operator Loop Action Tables
        self.latest_model_proposal: Dict[str, int] = {}
        self.pending_operator_actions: Dict[str, int] = {}
        self.sticky_actions: Dict[str, Tuple[int, int]] = {}
        self.ripple_delay: int = OVERRIDE_TICKS
        self.or_shield = SmartOptimizer()
        
        # Copilot Suggestions Cache
        self.copilot_suggestions: Dict[str, Any] = {}
        
        # Maintenance Blocks
        self.active_blocks: Dict[str, Any] = {}
        self.sandbox_blocks: Dict[str, Any] = {}
        self.dynamic_constraints: Dict[str, Any] = {}
        
        # Simulation Brain (Lazy PPO)
        self.sim_model = None
        self.sim_env = None
        self.inference_active: bool = False
        self.inference_obs = None
        self.inference_actions = None
        self.inference_raw_actions = None
        self.inference_decision_meta: Dict[str, Any] = {}
        # Fleet and OR-Tools Schedule
        self.last_or_schedule: Dict[str, Any] = {}
        self.fleet_registry: Dict[str, Any] = {}
        
        self.audit_logs: List[Any] = []

_state_instance = SimulationState()

def get_state() -> SimulationState:
    return _state_instance
