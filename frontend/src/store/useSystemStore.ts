import { create } from 'zustand';

interface SystemState {
  isLockdown: boolean;
  isSafetyShield: boolean;
  isAutoCommit: boolean;
  networkFluidity: 'Nominal' | 'Warning' | 'Degraded' | '-';
  haltedPct: number;
  activeTrains: number;
  fetchStatus: () => Promise<void>;
  setLockdown: (enabled: boolean) => Promise<void>;
  setSafetyShield: (enabled: boolean) => Promise<void>;
  setAutoCommit: (enabled: boolean) => Promise<void>;
}

export const useSystemStore = create<SystemState>((set, get) => ({
  isLockdown: false,
  isSafetyShield: true,
  isAutoCommit: false,
  networkFluidity: '-',
  haltedPct: 0,
  activeTrains: 0,

  fetchStatus: async () => {
    try {
      const [statusRes, telemetryRes] = await Promise.all([
        fetch('http://localhost:8000/api/v1/system/inference-status'),
        fetch('http://localhost:8000/api/v1/telemetry'),
      ]);
      
      if (statusRes.ok && telemetryRes.ok) {
        const statusData = await statusRes.json();
        const telemetryData = await telemetryRes.json();
        set({
          isLockdown: statusData.lockdown || false,
          isSafetyShield: statusData.safety_shield !== false, // default true
          isAutoCommit: statusData.auto_commit || false,
          networkFluidity: telemetryData.network_fluidity || '-',
          haltedPct: telemetryData.halted_pct || 0,
          activeTrains: telemetryData.active_trains || 0,
        });
      }
    } catch (e) {
      console.error("Failed to fetch system status or telemetry:", e);
    }
  },

  setLockdown: async (enabled) => {
    const previous = get().isLockdown;
    set({ isLockdown: enabled }); // optimistic
    try {
      const res = await fetch('http://localhost:8000/api/v1/system/lockdown', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      if (!res.ok) throw new Error("Failed to set lockdown");
    } catch (e) {
      console.error(e);
      set({ isLockdown: previous }); // rollback
    }
  },

  setSafetyShield: async (enabled) => {
    const previous = get().isSafetyShield;
    set({ isSafetyShield: enabled }); // optimistic
    try {
      const res = await fetch('http://localhost:8000/api/v1/system/safety-shield', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      if (!res.ok) throw new Error("Failed to set safety shield");
    } catch (e) {
      console.error(e);
      set({ isSafetyShield: previous }); // rollback
    }
  },

  setAutoCommit: async (enabled) => {
    const previous = get().isAutoCommit;
    set({ isAutoCommit: enabled }); // optimistic
    try {
      const res = await fetch('http://localhost:8000/api/v1/system/auto-commit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      if (!res.ok) throw new Error("Failed to set auto commit");
    } catch (e) {
      console.error(e);
      set({ isAutoCommit: previous }); // rollback
    }
  },
}));
