import { create } from 'zustand';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface AISuggestion {
  recommendation_id: string;
  type: 'AI_DECISION';
  priority_level: 1 | 2 | 3 | 4 | 5;
  target_train_id: string;
  decided_action: string;
  /** Minutes: positive = time saved, negative = time lost */
  impact_analysis: number;
  /** 0.0 – 1.0 */
  confidence_score: number;
  reasoning: string;
  affected_edges: string[];
  timestamp: string; // ISO-8601
  status: 'executed' | 'overridden' | 'expired' | 'rejected';
  override_state: 'none' | 'overridden';
  urgency: 'CRITICAL' | 'ADVISORY' | 'INFO';
  decided_at_edge: string;
  decided_at_tick: number;
  /** Ticks remaining until backend expires this suggestion (populated by server). */
  expires_in_ticks?: number;
}


export interface ScheduleEntry {
  train_id: string;
  /** SVG path points for Marey chart */
  path: { x: number; y: number }[];
  type: 'actual' | 'ghost';
}

export interface ToastNotification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  timestamp: number;
}

export const STATION_DISTANCES: Record<string, number> = {
  "CSMT": 0,
  "DADAR": 9,
  "KALYAN": 54,
  "AMBERNATH": 63,
  "TITWALA": 80,
  "ATGAON": 98,
  "KASARA": 121,
  "IGATPURI": 136,
  "DEVLALI": 182,
  "NASHIK": 187,
  "MANMAD": 261
};

export const getSchematicY = (km: number, totalH: number, padTop: number): number => {
  const stations = Object.entries(STATION_DISTANCES).sort((a,b) => a[1]-b[1]);
  const index = Math.max(0, stations.findIndex(s => s[1] === km));
  const maxIndex = Math.max(1, stations.length - 1);
  const maxKm = 261;
  
  const physicalPct = km / maxKm;
  const rankPct = index / maxIndex;
  
  // Blend 75% physical distance with 25% even spacing to ensure lines don't overlap
  const blendedPct = physicalPct * 0.75 + rankPct * 0.25;
  return padTop + totalH * blendedPct;
};

// ---------------------------------------------------------------------------
// Store Shape
// ---------------------------------------------------------------------------
interface CopilotState {
  activeSuggestions: AISuggestion[];
  /** The suggestion currently being previewed (hovered) → ghost projection */
  previewState: AISuggestion | null;
  /** Committed + live train schedule lines rendered by MareyTimeline */
  globalSchedule: ScheduleEntry[];
  scheduleMaxTime: number;
  toasts: ToastNotification[];
  isConnected: boolean;

  // Actions
  addSuggestion: (s: AISuggestion) => void;
  removeSuggestion: (id: string) => void;
  updateSuggestionStatus: (id: string, status: AISuggestion['status']) => void;
  setPreviewState: (s: AISuggestion | null) => void;
  commitSuggestion: (id: string) => void;
  rejectSuggestion: (id: string) => void;

  addToast: (type: ToastNotification['type'], message: string) => void;
  dismissToast: (id: string) => void;

  setIsConnected: (v: boolean) => void;
  connectCopilotWS: () => void;
  fetchBaseSchedule: () => Promise<void>;
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------
export const useCopilotStore = create<CopilotState>((set, get) => {
  let socket: WebSocket | null = null;


  return {
    activeSuggestions: [],
    previewState: null,
    globalSchedule: [],
    scheduleMaxTime: 120,
    toasts: [],
    isConnected: false,

    // -----------------------------------------------------------------------
    // Base Schedule Fetching
    // -----------------------------------------------------------------------
    fetchBaseSchedule: async () => {
      try {
        const res = await fetch('http://localhost:8000/api/v1/fleet/generate-schedule', { method: 'POST' });
        if (!res.ok) return;
        const data = await res.json();
        if (data.schedule) {
          const W = 800;
          const PAD_LEFT = 140;
          const PAD_RIGHT = 20;
          const totalW = W - PAD_LEFT - PAD_RIGHT;
          const H = 400;
          const PAD_TOP = 20;
          const PAD_BTM = 28;
          const totalH = H - PAD_TOP - PAD_BTM;
          
          const newEntries: ScheduleEntry[] = [];
          let maxTime = 120;
          
          // First pass: find max time
          Object.values(data.schedule).forEach((nodesObj) => {
              Object.values(nodesObj as Record<string, {arrival: number, departure: number}>).forEach(times => {
                  if (times.arrival > maxTime) maxTime = times.arrival;
                  if (times.departure > maxTime) maxTime = times.departure;
              });
          });

          // Second pass: map to coords
          Object.entries(data.schedule).forEach(([trainId, nodesObj]) => {
             const nodes = Object.entries(nodesObj as Record<string, {arrival: number, departure: number}>);
             if (nodes.length > 0) {
                 // Sort nodes chronologically so the SVG path draws forwards in time
                 nodes.sort((a, b) => a[1].arrival - b[1].arrival);

                 const path = nodes.map(([nodeName, times]) => {
                     // x maps to time (0 - maxTime)
                     const timeVal = times.arrival || times.departure;
                     const timePct = Math.min(Math.max(timeVal / maxTime, 0), 1);
                     const x = PAD_LEFT + (totalW * timePct);
                     // y maps to absolute distance but blended to prevent overlap
                     const km = STATION_DISTANCES[nodeName] || 0;
                     const y = getSchematicY(km, totalH, PAD_TOP);
                     return { x, y };
                 });
                 newEntries.push({
                     train_id: trainId,
                     type: 'actual',
                     path
                 });
             }
          });
          set({ globalSchedule: newEntries, scheduleMaxTime: maxTime });
        }
      } catch (err) {
        console.warn('Failed to fetch base schedule', err);
      }
    },

    // -----------------------------------------------------------------------
    // Suggestion management
    // -----------------------------------------------------------------------
    addSuggestion: (s) =>
      set((state) => ({
        activeSuggestions: [
          s,
          // Keep only up to 5 pending suggestions in the queue
          ...state.activeSuggestions.filter((x) => x.recommendation_id !== s.recommendation_id).slice(0, 4),
        ],
      })),

    removeSuggestion: (id) =>
      set((state) => ({
        activeSuggestions: state.activeSuggestions.filter((s) => s.recommendation_id !== id),
      })),

    updateSuggestionStatus: (id, status) =>
      set((state) => ({
        activeSuggestions: state.activeSuggestions.map((s) =>
          s.recommendation_id === id ? { ...s, status } : s
        ),
      })),

    // -----------------------------------------------------------------------
    // Preview (Ghost Projection)
    // -----------------------------------------------------------------------
    setPreviewState: (s) => set({ previewState: s }),

    // -----------------------------------------------------------------------
    // Commit → handled by SCHEDULE_UPDATED
    // -----------------------------------------------------------------------
    commitSuggestion: (id) => {
      set((state) => ({
        activeSuggestions: state.activeSuggestions.filter((s) => s.recommendation_id !== id),
        previewState: null,
      }));
    },

    // -----------------------------------------------------------------------
    // Reject
    // -----------------------------------------------------------------------
    rejectSuggestion: (id) =>
      set((state) => ({
        activeSuggestions: state.activeSuggestions.filter((s) => s.recommendation_id !== id),
        previewState: state.previewState?.recommendation_id === id ? null : state.previewState,
      })),

    // -----------------------------------------------------------------------
    // Toasts
    // -----------------------------------------------------------------------
    addToast: (type, message) =>
      set((state) => ({
        toasts: [
          ...state.toasts,
          { id: crypto.randomUUID(), type, message, timestamp: Date.now() },
        ].slice(-5), // max 5 toasts
      })),

    dismissToast: (id) =>
      set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),

    // -----------------------------------------------------------------------
    // Connection
    // -----------------------------------------------------------------------
    setIsConnected: (v) => set({ isConnected: v }),

    connectCopilotWS: () => {
      if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        return; // already connected / connecting
      }

      socket = new WebSocket('ws://localhost:8000/ws/copilot');

      socket.onopen = () => {
        set({ isConnected: true });
        console.log('[ORBIT] Co-pilot WebSocket connected');
      };

      socket.onmessage = (event) => {
        try {
          // DEBUG TRACE
          fetch('/api/v1/telemetry', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ debug_ws: event.data.substring(0, 100) }) }).catch(() => {});

          const data = JSON.parse(event.data);

          if (data.type === 'AI_DECISION') {
            get().addSuggestion(data as AISuggestion);
          } else if (data.type === 'SCHEDULE_UPDATED') {
            // Backend confirmed commit — update UI accordingly
            const rid         = data.recommendation_id as string;
            const trainId     = data.target_train_id    as string;
            const action      = data.decided_action    as string;
            const newEdge     = data.new_edge_id        as string | null;

            // 1. Update card status in active suggestions queue
            get().updateSuggestionStatus(rid, 'overridden');

            // 2. Toast with action text + new edge (if advanced)
            get().addToast(
              'warning',
              `⚠️ Overridden: "${action}" → ${trainId}` +
              (newEdge ? ` (→ ${newEdge})` : '')
            );

            // 3. Fetch real updated schedule from the backend to redraw the graph
            get().fetchBaseSchedule();

            console.log('[ORBIT] Schedule updated:', rid);
          } else if (data.type === 'AUTO_INTERVENTION') {
            get().addToast('warning', data.message);
          }
        } catch {
          console.warn('[ORBIT] Failed to parse WS message', event.data);
        }
      };

      socket.onclose = () => {
        set({ isConnected: false });
        socket = null;
        console.log('[ORBIT] Co-pilot WS closed — retrying in 4s...');
        setTimeout(() => {
          useCopilotStore.getState().connectCopilotWS();
        }, 4000);
      };

      socket.onerror = (err) => {
        console.error('[ORBIT] Co-pilot WS error:', err);
      };
    },
  };
});
