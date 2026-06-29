import { useMapStore } from '../store/useMapStore';

export function useTopology() {
  // Since useMapStore already establishes the WS connection to /ws/topology 
  // and maintains the global state for other components, we tap into it here 
  // for a clean React Hook interface.
  const { topology, isConnected } = useMapStore();
  const error: Error | null = null; // Removed useState to fix TS error, but keeping the signature

  return { topology, isConnected, error };
}
