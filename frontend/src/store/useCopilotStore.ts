import { create } from 'zustand';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface AISuggestion {
  recommendation_id: string;
  type: 'AI_RECOMMENDATION';
  priority_level: 1 | 2 | 3 | 4 | 5;
  target_train_id: string;
  proposed_action: string;
  /** Seconds: positive = time saved, negative = time lost */
  impact_analysis: number;
  /** 0.0 – 1.0 */
  confidence_score: number;
  reasoning: string;
  affected_edges: string[];
  timestamp: string; // ISO-8601
  status: 'pending' | 'committed' | 'rejected';
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

// ---------------------------------------------------------------------------
// Store Shape
// ---------------------------------------------------------------------------
interface CopilotState {
  activeSuggestions: AISuggestion[];
  /** The suggestion currently being previewed (hovered) → ghost projection */
  previewState: AISuggestion | null;
  /** Committed + live train schedule lines rendered by MareyTimeline */
  globalSchedule: ScheduleEntry[];
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

  /** Seed schedule with some dummy Marey lines that persist across sessions */
  const seedSchedule: ScheduleEntry[] = [
    { train_id: 'TRN100', type: 'actual', path: [{ x: 50, y: 40 }, { x: 300, y: 120 }, { x: 550, y: 200 }] },
    { train_id: 'TRN101', type: 'actual', path: [{ x: 100, y: 200 }, { x: 400, y: 120 }, { x: 700, y: 40 }] },
  ];

  return {
    activeSuggestions: [],
    previewState: null,
    globalSchedule: [],
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
          const PAD_LEFT = 90;
          const PAD_RIGHT = 20;
          const totalW = W - PAD_LEFT - PAD_RIGHT;
          const H = 260;
          const PAD_TOP = 20;
          const PAD_BTM = 28;
          const totalH = H - PAD_TOP - PAD_BTM;
          
          const newEntries: ScheduleEntry[] = [];
          Object.entries(data.schedule).forEach(([trainId, nodesObj]) => {
             const nodes = Object.entries(nodesObj as Record<string, {arrival: number, departure: number}>);
             if (nodes.length > 0) {
                 const path = nodes.map(([, times], idx) => {
                     // x maps to time (0-120 minutes)
                     const timePct = Math.min(Math.max((times.arrival || times.departure) / 120, 0), 1);
                     const x = PAD_LEFT + (totalW * timePct);
                     // y maps to node index (roughly station)
                     const y = PAD_TOP + (totalH / (nodes.length - 1)) * idx;
                     return { x, y: y || PAD_TOP };
                 });
                 newEntries.push({
                     train_id: trainId,
                     type: 'actual',
                     path
                 });
             }
          });
          set({ globalSchedule: newEntries });
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
    // Commit → transition ghost path into actual schedule
    // -----------------------------------------------------------------------
    commitSuggestion: (id) => {
      const suggestion = get().activeSuggestions.find((s) => s.recommendation_id === id);
      if (!suggestion) return;

      // Build a ghost Marey path for the committed action
      // (In production this would come from the backend schedule update)
      const newEntry: ScheduleEntry = {
        train_id: suggestion.target_train_id,
        type: 'actual',
        path: [
          { x: 80,  y: 180 },
          { x: 320, y: 100 },
          { x: 620, y: 50  },
        ],
      };

      set((state) => ({
        activeSuggestions: state.activeSuggestions.filter((s) => s.recommendation_id !== id),
        globalSchedule: [...state.globalSchedule, newEntry],
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
          const data = JSON.parse(event.data);

          if (data.type === 'AI_RECOMMENDATION') {
            get().addSuggestion(data as AISuggestion);
          } else if (data.type === 'SCHEDULE_UPDATED') {
            // Backend confirmed commit — update UI accordingly
            const rid         = data.recommendation_id as string;
            const trainId     = data.target_train_id    as string;
            const action      = data.proposed_action    as string;
            const newEdge     = data.new_edge_id        as string | null;
            const affEdges    = (data.affected_edges    as string[]) ?? [];

            // 1. Remove card from active suggestions queue
            get().commitSuggestion(rid);

            // 2. Toast with action text + new edge (if advanced)
            get().addToast(
              'success',
              `✅ Applied: "${action}" → ${trainId}` +
              (newEdge ? ` (→ ${newEdge})` : '')
            );

            // 3. Add a Marey ghost-line built from affected_edges as x-axis proxy
            //    In production replace with real timetable arrival/departure times.
            if (affEdges.length > 0) {
              const newEntry: ScheduleEntry = {
                train_id: trainId,
                type: 'actual',
                path: affEdges.map((_, i) => ({
                  x: 80 + i * 120,
                  y: 180 - i * 40,
                })),
              };
              set((state) => ({
                globalSchedule: [...state.globalSchedule, newEntry],
              }));
            }

            console.log('[ORBIT] Schedule updated:', rid);
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
