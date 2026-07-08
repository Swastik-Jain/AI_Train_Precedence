import { create } from 'zustand';
import { apiUrl, wsUrl } from '../lib/api';

interface SystemState {
  isLockdown: boolean;
  isSafetyShield: boolean;
  networkFluidity: 'Nominal' | 'Warning' | 'Degraded' | '-';
  haltedPct: number;
  activeTrains: number;
  isBackendReachable: boolean;
  fetchStatus: () => Promise<void>;
  setLockdown: (enabled: boolean) => Promise<void>;
  setSafetyShield: (enabled: boolean) => Promise<void>;
}

export const useSystemStore = create<SystemState>((set, get) => ({
  isLockdown: false,
  isSafetyShield: true,
  networkFluidity: '-',
  haltedPct: 0,
  activeTrains: 0,

  isBackendReachable: true,

  fetchStatus: async () => {
    try {
      const [statusRes, telemetryRes] = await Promise.all([
        fetch(apiUrl('/api/v1/system/inference-status')),
        fetch(apiUrl('/api/v1/telemetry')),
      ]);
      
      if (statusRes.ok && telemetryRes.ok) {
        const statusData = await statusRes.json();
        const telemetryData = await telemetryRes.json();
        set({
          isLockdown: statusData.lockdown || false,
          isSafetyShield: statusData.safety_shield !== false, // default true
          networkFluidity: telemetryData.network_fluidity || '-',
          haltedPct: telemetryData.halted_pct || 0,
          activeTrains: telemetryData.active_trains || 0,
          isBackendReachable: true,
        });
      } else {
        set({ isBackendReachable: false });
      }
    } catch (e) {
      console.error("Failed to fetch system status or telemetry:", e);
      set({ isBackendReachable: false });
    }
  },

  setLockdown: async (enabled) => {
    const previous = get().isLockdown;
    set({ isLockdown: enabled }); // optimistic
    try {
      const res = await fetch(apiUrl('/api/v1/system/lockdown'), {
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
      const res = await fetch(apiUrl('/api/v1/system/safety-shield'), {
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
}));
