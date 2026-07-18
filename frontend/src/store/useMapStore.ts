import { create } from 'zustand';
import { WS_BASE } from '../lib/api';
import { useMaintenanceStore } from './useMaintenanceStore';

export interface Node {
  id: string;
  x: number;
  y: number;
  type: string;
  km: number;
  capacity?: number;
  stId?: string;
  platform_index?: number;
  loop_index?: number;
}

export interface Edge {
  id: string;
  source: string;
  target: string;
  length: number;
  max_speed: number;
  capacity?: number;
  type: string;
}

export interface TopologyData {
  nodes: Node[];
  edges: Edge[];
}

export interface TrainState {
  train_id: string;
  edge_id: string;
  position_node: number;
  position_percentage: number;
  status: 'Finished' | 'Scheduled' | 'Banker Ops' | 'Boarding' | 'Waiting at Signal' | 'Moving' | 'Halted';
  path: string[];
  direction?: string | number;
}

export interface TrainOption {
  train_id: string;
  status: string;
}

interface MapState {
  topology: TopologyData | null;
  trainStates: TrainState[];
  allTrains: TrainOption[];
  conflicts: string[];
  trainConflicts: string[];
  tokenTrains: string[];
  /** Edges that just had an AI action committed — shown as green flash */
  committedEdges: Set<string>;
  selectedTrainId: string | null;
  selectedEdgeId: string | null;
  isConnected: boolean;
  /** Train that just had an AI commit applied — shown as highlighted ring for 6 s */
  committedTrainId: string | null;
  /** The RL action that was committed: 0=STOP, 2=DIVERT, 1=MAIN */
  committedAction: number | null;
  zoomLevel: number;
  simTick: number;
  tickIntervalS: number;
  
  setTopology: (topology: TopologyData) => void;
  updateLiveState: (trains: TrainState[], conflicts: string[], trainConflicts: string[], allTrains: TrainOption[]) => void;
  setSelectedTrain: (trainId: string | null) => void;
  setSelectedEdge: (edgeId: string | null) => void;
  setIsConnected: (status: boolean) => void;
  setZoomLevel: (zoom: number | ((prev: number) => number)) => void;
  setSimTick: (t: number) => void;
  connectWebSocket: () => void;
}


export const useMapStore = create<MapState>((set) => {
  let socket: WebSocket | null = null;
  
  return {
    topology: null,
    trainStates: [],
    allTrains: [],
    conflicts: [],
    trainConflicts: [],
    tokenTrains: [],
    committedEdges: new Set<string>(),
    selectedTrainId: null,
    selectedEdgeId: null,
    isConnected: false,
    committedTrainId: null,
    committedAction: null,
    zoomLevel: 1.2,
    simTick: 0,
    tickIntervalS: 1.0,


    setTopology: (topology) => set({ topology }),
    
    updateLiveState: (trainStates, conflicts, trainConflicts, allTrains) => set({ trainStates, conflicts, trainConflicts, allTrains }),
    
    setSelectedTrain: (selectedTrainId) => set({ selectedTrainId }),
    
    setSelectedEdge: (selectedEdgeId) => set({ selectedEdgeId }),
    
    setIsConnected: (isConnected) => set({ isConnected }),

    setZoomLevel: (zoom) => set((state) => ({ 
      zoomLevel: typeof zoom === 'function' ? zoom(state.zoomLevel) : zoom 
    })),

    setSimTick: (simTick) => set({ simTick }),

    connectWebSocket: () => {
      if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        return;
      }
      
      const wsUrl = `${WS_BASE}/ws/topology`;
      socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        set({ isConnected: true });
        console.log('Topology WebSocket Connected');
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'topology_init') {
          set({ topology: data.topology });
        } else if (data.type === 'topology_update') {
          set((state) => ({ 
            trainStates: data.trains || [],
            conflicts: data.conflicts || [],
            trainConflicts: data.train_conflicts || [],
            allTrains: data.all_trains || [],
            tokenTrains: data.token_trains || state.tokenTrains,
            simTick: data.sim_time !== undefined ? data.sim_time : state.simTick,
            tickIntervalS: data.tick_interval_s !== undefined ? data.tick_interval_s : state.tickIntervalS
          }));
          
          if (data.maintenance_blocks) {
             useMaintenanceStore.setState((state) => {
                const newMap = new Map();
                // Retain what-if blocks
                state.activeBlocks.forEach((b, k) => {
                   if (b.isWhatIf) newMap.set(k, b);
                });
                // Add backend blocks
                data.maintenance_blocks.forEach((b: any) => {
                   newMap.set(b.element_id, { ...b, blockId: b.element_id });
                });
                return { activeBlocks: newMap };
             });
          }
        } else if (data.type === 'SCHEDULE_UPDATED') {
          // Controller approved an AI action — apply to live train state immediately
          const { target_train_id, new_edge_id, affected_edges, rl_action } = data as {
            target_train_id: string;
            new_edge_id: string | null;
            affected_edges: string[];
            rl_action?: number;
          };

          if (new_edge_id) {
            // Move the train to its new edge in the live state
            set((state) => ({
              trainStates: state.trainStates.map((t) =>
                t.train_id === target_train_id
                  ? { ...t, edge_id: new_edge_id, position_percentage: 0, status: 'Moving' as const }
                  : t
              ),
            }));
          }

          // ── Highlight the committed train for 6 seconds ─────────────────
          set({
            committedTrainId: target_train_id,
            committedAction: rl_action ?? 1,
          });
          setTimeout(() => {
            set({ committedTrainId: null, committedAction: null });
          }, 6000);

          // Flash affected edges green for 4 seconds regardless of edge advance
          const edgesToFlash: string[] = [
            ...(new_edge_id ? [new_edge_id] : []),
            ...(affected_edges ?? []),
          ];
          if (edgesToFlash.length > 0) {
            set((state) => ({
              committedEdges: new Set([...state.committedEdges, ...edgesToFlash]),
            }));
            setTimeout(() => {
              set((state) => {
                const next = new Set(state.committedEdges);
                edgesToFlash.forEach((e) => next.delete(e));
                return { committedEdges: next };
              });
            }, 4000);
          }


          // Sync the maintenance block into the maintenance store
          const block = data.block;
          if (block?.element_id) {
            useMaintenanceStore.getState().applyBlock({
              blockId: block.element_id,
              element_id: block.element_id,
              type: block.type ?? 'TRACK_SEGMENT',
              severity: block.severity ?? 'TOTAL_BLOCK',
              start_time: block.start_time ?? new Date().toISOString(),
              end_time: block.end_time ?? new Date().toISOString(),
              reason: block.reason,
              applied_at: block.applied_at,
            });
          }
        } else if (data.type === 'MAINTENANCE_CLEARED') {
          if (data.element_id) {
            useMaintenanceStore.getState().removeBlock(data.element_id);
          }
        }
      };


      socket.onclose = () => {
        set({ isConnected: false });
        console.log('Topology WebSocket Disconnected. Retrying in 3s...');
        socket = null;
        setTimeout(() => {
          useMapStore.getState().connectWebSocket();
        }, 3000);
      };

      socket.onerror = (err) => {
        console.error('Topology WebSocket Error:', err);
      };
    }
  };
});
