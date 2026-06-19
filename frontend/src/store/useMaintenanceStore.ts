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
    try {
      const res = await fetch('/api/v1/maintenance/blocks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          element_id: block.element_id,
          type: block.type,
          start_time: block.start_time,
          end_time: block.end_time,
          severity: block.severity,
          reason: block.reason ?? 'Scheduled maintenance',
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        useCopilotStore.getState().addToast('error', data.detail ?? 'Failed to apply block.');
        return;
      }

      // Store locally
      const record: BlockRecord = {
        ...block,
        blockId: crypto.randomUUID(),
        applied_at: data.timestamp,
      };
      get().applyBlock(record);

      // Trigger impact analysis + toast
      await get().fetchImpactAnalysis();

    } catch {
      useCopilotStore.getState().addToast('error', 'Network error — could not apply maintenance block.');
    }
  },

  // -------------------------------------------------------------------------
  // REST — remove from backend
  // -------------------------------------------------------------------------
  removeBlockRemote: async (elementId) => {
    const isWhatIf = get().activeBlocks.get(elementId)?.isWhatIf;

    // What-if blocks are local only
    if (isWhatIf) {
      get().removeBlock(elementId);
      return;
    }

    try {
      const res = await fetch(
        `/api/v1/maintenance/blocks/${encodeURIComponent(elementId)}`,
        { method: 'DELETE' }
      );

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        useCopilotStore.getState().addToast('error', data.detail ?? 'Failed to remove block.');
        return;
      }

      get().removeBlock(elementId);
      useCopilotStore.getState().addToast('info', `🔓 Maintenance cleared on ${elementId}.`);
    } catch {
      useCopilotStore.getState().addToast('error', 'Network error — could not remove block.');
    }
  },

  // -------------------------------------------------------------------------
  // GET /api/v1/impact-analysis — trigger the "Ripple Effect" toast
  // -------------------------------------------------------------------------
  fetchImpactAnalysis: async () => {
    try {
      const res = await fetch('/api/v1/impact-analysis');
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
      const res = await fetch('/api/v1/maintenance/blocks');
      const data = await res.json();
      set((state) => {
        const next = new Map(state.activeBlocks);
        (data.blocks ?? []).forEach((b: BlockRecord) => {
          next.set(b.element_id, { ...b, blockId: b.element_id });
        });
        return { activeBlocks: next };
      });
    } catch {
      console.warn('[MMS] Could not fetch active blocks from backend');
    }
  },
}));
