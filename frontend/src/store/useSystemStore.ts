import { create } from 'zustand';

interface SystemState {
  isLockdown: boolean;
  isSafetyShield: boolean;
  isAutoCommit: boolean;
  fetchStatus: () => Promise<void>;
  setLockdown: (enabled: boolean) => Promise<void>;
  setSafetyShield: (enabled: boolean) => Promise<void>;
  setAutoCommit: (enabled: boolean) => Promise<void>;
}

export const useSystemStore = create<SystemState>((set) => ({
  isLockdown: false,
  isSafetyShield: true,
  isAutoCommit: false,

  fetchStatus: async () => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/system/inference-status');
      if (res.ok) {
        const data = await res.json();
        set({
          isLockdown: data.lockdown || false,
          isSafetyShield: data.safety_shield !== false, // default true
          isAutoCommit: data.auto_commit || false,
        });
      }
    } catch (e) {
      console.error("Failed to fetch system status:", e);
    }
  },

  setLockdown: async (enabled) => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/system/lockdown', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      if (res.ok) {
        set({ isLockdown: enabled });
      }
    } catch (e) {
      console.error("Failed to set lockdown:", e);
    }
  },

  setSafetyShield: async (enabled) => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/system/safety-shield', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      if (res.ok) {
        set({ isSafetyShield: enabled });
      }
    } catch (e) {
      console.error("Failed to set safety shield:", e);
    }
  },

  setAutoCommit: async (enabled) => {
    try {
      const res = await fetch('http://localhost:8000/api/v1/system/auto-commit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      if (res.ok) {
        set({ isAutoCommit: enabled });
      }
    } catch (e) {
      console.error("Failed to set auto commit:", e);
    }
  },
}));
