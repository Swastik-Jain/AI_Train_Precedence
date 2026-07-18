/**
 * usePresentationStore.ts
 * 
 * This module is responsible for owning the presentation (render) state for trains.
 * It is structured as a Zustand store to match the existing conventions in the 
 * codebase (e.g., useMapStore, useCopilotStore).
 * 
 * By separating presentation state from raw simulation state (useMapStore), we 
 * create a clear boundary that allows us to trigger cosmetic tweens for 
 * intra-station moves while preserving physics interpolation for inter-station moves.
 */

import { create } from 'zustand';

export type AnimationMode = 'initial' | 'physics' | 'cosmetic' | 'dwell';

export const INTRA_STATION_TWEEN_DURATION_S = 0.8;
export const EDGE_DEBOUNCE_TICKS = 2; // How many consecutive ticks a new display-default edge_id must be observed before we accept it


export interface TrainPresentationState {
  train_id: string;
  lastConfirmedEdge: string | null;
  lastConfirmedNode?: number | null;
  
  // The logical target positions for framer-motion to animate towards.
  targetX: number;
  targetY: number;
  
  // Transition configuration for framer-motion
  animationMode: AnimationMode;
  durationS: number;
  ease: string;

  // Debounce bookkeeping for display-default noise
  candidateEdge?: string;
  candidateCount?: number;
}

interface PresentationStore {
  trains: Record<string, TrainPresentationState>;
  updateTrainPresentation: (train_id: string, updates: Partial<Omit<TrainPresentationState, 'train_id'>>) => void;
  initializeTrainPresentation: (train_id: string, initialState: Omit<TrainPresentationState, 'train_id'>) => void;
}

export const usePresentationStore = create<PresentationStore>((set) => ({
  trains: {},
  
  updateTrainPresentation: (train_id, updates) => set((state) => {
    const existing = state.trains[train_id];
    if (!existing) return state; // Must initialize first
    
    return {
      trains: {
        ...state.trains,
        [train_id]: { ...existing, ...updates },
      },
    };
  }),
  
  initializeTrainPresentation: (train_id, initialState) => set((state) => {
    if (state.trains[train_id]) return state; // Already initialized
    
    return {
      trains: {
        ...state.trains,
        [train_id]: {
          train_id,
          ...initialState,
        },
      },
    };
  }),
}));
