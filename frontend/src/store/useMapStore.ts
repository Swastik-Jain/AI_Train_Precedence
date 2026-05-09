import { create } from 'zustand';
import { useMaintenanceStore } from './useMaintenanceStore';

export interface Node {
  id: string;
  x: number;
  y: number;
  type: string;
}

export interface Edge {
  id: string;
  source: string;
  target: string;
  length: number;
  max_speed: number;
}

export interface TopologyData {
  nodes: Node[];
  edges: Edge[];
}

export interface TrainState {
  train_id: string;
  edge_id: string;
  position_percentage: number;
  status: 'Moving' | 'Halted' | 'Conflict';
  path: string[];
}

interface MapState {
  topology: TopologyData | null;
  trainStates: TrainState[];
  conflicts: string[];
  /** Edges that just had an AI action committed — shown as green flash */
  committedEdges: Set<string>;
  selectedTrainId: string | null;
  selectedEdgeId: string | null;
  isConnected: boolean;
  /** Train that just had an AI commit applied — shown as highlighted ring for 6 s */
  committedTrainId: string | null;
  /** The RL action that was committed: 0=STOP, 2=DIVERT, 1=MAIN */
  committedAction: number | null;
  
  setTopology: (topology: TopologyData) => void;
  updateLiveState: (trains: TrainState[], conflicts: string[]) => void;
  setSelectedTrain: (trainId: string | null) => void;
  setSelectedEdge: (edgeId: string | null) => void;
  setIsConnected: (status: boolean) => void;
  connectWebSocket: () => void;
}


export const useMapStore = create<MapState>((set) => {
  let socket: WebSocket | null = null;
  
  return {
    topology: null,
    trainStates: [],
    conflicts: [],
    committedEdges: new Set<string>(),
    selectedTrainId: null,
    selectedEdgeId: null,
    isConnected: false,
    committedTrainId: null,
    committedAction: null,


    setTopology: (topology) => set({ topology }),
    
    updateLiveState: (trainStates, conflicts) => set({ trainStates, conflicts }),
    
    setSelectedTrain: (selectedTrainId) => set({ selectedTrainId }),
    
    setSelectedEdge: (selectedEdgeId) => set({ selectedEdgeId }),
    
    setIsConnected: (isConnected) => set({ isConnected }),

    connectWebSocket: () => {
      if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        return;
      }
      
      const wsUrl = `ws://localhost:8000/ws/topology`;
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
          set({ 
            trainStates: data.trains || [],
            conflicts: data.conflicts || []
          });
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
