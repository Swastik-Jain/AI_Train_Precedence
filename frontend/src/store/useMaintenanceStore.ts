import { apiUrl, wsUrl } from '../lib/api';
import { create } from 'zustand';
import { useCopilotStore } from './useCopilotStore';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export type BlockType     = 'PLATFORM' | 'TRACK_SEGMENT';
export type BlockSeverity = 'TOTAL_BLOCK' | 'SPEED_RESTRICTION';
export type RerouteStrategy =
  | 'TEMPORAL_SHIFT'
  | 'SPATIAL_SHIFT'
  | 'MANUAL_INTERVENTION';

export interface BlockRecord {
  blockId: string;         // UUID (frontend)
  element_id: string;
  type: BlockType;
  start_time: string;      // ISO-8601
  end_time: string;        // ISO-8601
  severity: BlockSeverity;
  reason?: string;
  speed_limit?: number;    // km/h — only relevant for SPEED_RESTRICTION
  applied_at?: string;
  isWhatIf?: boolean;      // sandbox-only, NOT sent to backend
}

export interface ImpactReport {
  affected_trains: number;
  affected_train_ids: string[];
  strategy: RerouteStrategy;
  detail: string;
  element_id: string;
  timestamp: string;
}

export interface ConsolidatedImpact {
  status: 'clear' | 'blocks_active';
  message: string;
  total_affected_trains: number;
  affected_train_ids: string[];
  primary_strategy?: RerouteStrategy;
  block_reports: ImpactReport[];
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Store shape
// ---------------------------------------------------------------------------
interface MaintenanceState {
  activeBlocks: Map<string, BlockRecord>;
  selectedEdgeForBlock: string | null;
  impactReport: ConsolidatedImpact | null;
  isDrawerOpen: boolean;

  // Actions
  applyBlock: (block: BlockRecord) => void;
  removeBlock: (elementId: string) => void;
  setSelectedEdgeForBlock: (id: string | null) => void;
  openDrawer: (edgeId?: string) => void;
  closeDrawer: () => void;
  setImpactReport: (r: ConsolidatedImpact | null) => void;

  // REST helpers
  applyBlockRemote: (block: Omit<BlockRecord, 'blockId'>) => Promise<void>;
  removeBlockRemote: (elementId: string) => Promise<void>;
  fetchImpactAnalysis: () => Promise<void>;
  fetchActiveBlocks: () => Promise<void>;
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------
export const useMaintenanceStore = create<MaintenanceState>((set, get) => ({
  activeBlocks: new Map(),
  selectedEdgeForBlock: null,
  impactReport: null,
  isDrawerOpen: false,

  // -------------------------------------------------------------------------
  applyBlock: (block) =>
    set((state) => {
      const next = new Map(state.activeBlocks);
      next.set(block.element_id, block);
      return { activeBlocks: next };
    }),

  removeBlock: (elementId) =>
    set((state) => {
      const next = new Map(state.activeBlocks);
      next.delete(elementId);
      return { activeBlocks: next };
    }),

  setSelectedEdgeForBlock: (id) => set({ selectedEdgeForBlock: id }),

  openDrawer: (edgeId) =>
    set((state) => ({
      isDrawerOpen: true,
      selectedEdgeForBlock: edgeId ?? state.selectedEdgeForBlock,
    })),

  closeDrawer: () => set({ isDrawerOpen: false }),

  setImpactReport: (r) => set({ impactReport: r }),

  // -------------------------------------------------------------------------
  // REST — apply on backend then update local state + trigger impact toast
  // -------------------------------------------------------------------------
  applyBlockRemote: async (block) => {
    // Route to the correct endpoint based on whether this is a sandbox what-if
    // block or a real maintenance block.
    // What-if → /api/v1/sandbox/blocks   (never touches ACTIVE_BLOCKS or RL env)
    // Real    → /api/v1/maintenance/blocks (syncs to RL env immediately)
    const endpoint = block.isWhatIf
      ? '/api/v1/sandbox/blocks'
      : '/api/v1/maintenance/blocks';

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          element_id: block.element_id,
          type: block.type,
          start_time: block.start_time,
          end_time: block.end_time,
          severity: block.severity,
          reason: block.reason ?? (block.isWhatIf ? 'What-if scenario' : 'Scheduled maintenance'),
          speed_limit: block.speed_limit ?? 30,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        useCopilotStore.getState().addToast('error', data.detail ?? 'Failed to apply block.');
        return;
      }

      // Store locally with the isWhatIf flag preserved
      const record: BlockRecord = {
        ...block,
        blockId: crypto.randomUUID(),
        applied_at: data.block?.applied_at ?? data.timestamp,
        isWhatIf: block.isWhatIf ?? false,
      };
      get().applyBlock(record);

      // Only trigger impact analysis for real blocks — sandbox blocks don't
      // affect the live network so there's no ripple effect to report.
      if (!block.isWhatIf) {
        await get().fetchImpactAnalysis();
      } else {
        useCopilotStore.getState().addToast('info', `🧪 What-if block added on ${block.element_id} (sandbox only).`);
      }

    } catch {
      useCopilotStore.getState().addToast('error', 'Network error — could not apply block.');
    }
  },

  // -------------------------------------------------------------------------
  // REST — remove from backend
  // -------------------------------------------------------------------------
  removeBlockRemote: async (elementId) => {
    const isWhatIf = get().activeBlocks.get(elementId)?.isWhatIf;

    // Route DELETE to the matching endpoint so sandbox blocks never hit the
    // maintenance DELETE route (which would 404 since they aren't in ACTIVE_BLOCKS).
    const endpoint = isWhatIf
      ? `/api/v1/sandbox/blocks/${encodeURIComponent(elementId)}`
      : `/api/v1/maintenance/blocks/${encodeURIComponent(elementId)}`;

    try {
      const res = await fetch(endpoint, { method: 'DELETE' });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        useCopilotStore.getState().addToast('error', data.detail ?? 'Failed to remove block.');
        return;
      }

      get().removeBlock(elementId);
      useCopilotStore.getState().addToast(
        'info',
        isWhatIf ? `🧪 What-if block removed: ${elementId}.` : `🔓 Maintenance cleared on ${elementId}.`
      );
    } catch {
      useCopilotStore.getState().addToast('error', 'Network error — could not remove block.');
    }
  },

  // -------------------------------------------------------------------------
  // GET /api/v1/impact-analysis — trigger the "Ripple Effect" toast
  // -------------------------------------------------------------------------
  fetchImpactAnalysis: async () => {
    try {
      const res = await fetch(apiUrl('/api/v1/impact-analysis'));
      const data: ConsolidatedImpact = await res.json();
      get().setImpactReport(data);

      if (data.status === 'blocks_active') {
        const strat = data.primary_strategy?.replace(/_/g, ' ') ?? 'Rerouting';
        useCopilotStore.getState().addToast(
          'warning',
          `🔧 ${data.message} (${strat})`
        );
      }
    } catch {
      console.warn('[MMS] Impact analysis fetch failed');
    }
  },

  // -------------------------------------------------------------------------
  // Sync active blocks on page load
  // -------------------------------------------------------------------------
  fetchActiveBlocks: async () => {
    try {
      // Fetch real maintenance blocks AND sandbox what-if blocks separately,
      // then merge them into the same local Map with correct isWhatIf tags.
      const [realRes, sandboxRes] = await Promise.all([
        fetch(apiUrl('/api/v1/maintenance/blocks')),
        fetch(apiUrl('/api/v1/sandbox/blocks')),
      ]);
      const realData = realRes.ok ? await realRes.json() : { blocks: [] };
      const sandboxData = sandboxRes.ok ? await sandboxRes.json() : { blocks: [] };

      set((state) => {
        const next = new Map(state.activeBlocks);
        (realData.blocks ?? []).forEach((b: BlockRecord) => {
          next.set(b.element_id, { ...b, blockId: b.element_id, isWhatIf: false });
        });
        (sandboxData.blocks ?? []).forEach((b: BlockRecord) => {
          next.set(b.element_id, { ...b, blockId: b.element_id, isWhatIf: true });
        });
        return { activeBlocks: next };
      });
    } catch {
      console.warn('[MMS] Could not fetch active blocks from backend');
    }
  },
}));
