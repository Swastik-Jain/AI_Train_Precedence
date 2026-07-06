export type SegZone     = { type:'SEG'; x1:number; x2:number; cap:number; isGhat?:boolean; speed?:number; km?:number; startKm?:number; endKm?:number };
export type SwitchZone  = { type:'SW';  x1:number; x2:number; fromCap:number; toCap:number };
export type StationZone = { type:'ST';  x1:number; x2:number; cap:number; stId:string; isLeft?:boolean; isRight?:boolean };
export type Zone        = SegZone | SwitchZone | StationZone;

/** Convert backend topology graph to KineticMap zones array */
export function topologyToZones(topology: {
  nodes: Array<{id: string; type: string; capacity?: number; km?: number; speed?: number}>;
  edges: Array<{id: string; source?: string; target?: string; from?: string; to?: string; length_px?: number; length?: number; capacity?: number; max_speed?: number}>;
}): Zone[] {
  const zones: Zone[] = [];
  
  if (!topology || !topology.nodes || !topology.edges) return zones;

  const stationTypes = ['TERMINUS', 'MAJOR_JUNCTION', 'CROSSING_LOOP', 'STATION', 'PLATFORM', 'LOOP'];
  const rawStations = topology.nodes
    .filter(n => stationTypes.includes(n.type))
    .sort((a, b) => (a.km || 0) - (b.km || 0));

  if (rawStations.length === 0) return zones;

  // Group multiple platform nodes at the same location into a single macro-station
  const stations: typeof rawStations = [];
  for (const n of rawStations) {
    // @ts-ignore
    const stId = n.stId;
    const existing = stations.find(s => 
      // @ts-ignore
      (stId && (s as any).stId === stId) || Math.abs((s.km || 0) - (n.km || 0)) < 0.1
    );
    if (existing) {
      if (n.type !== 'LOOP') {
        existing.capacity = (existing.capacity || 0) + (n.capacity || 1);
      }
    } else {
      stations.push({ ...n, capacity: n.type !== 'LOOP' ? (n.capacity || 1) : 0 });
    }
  }

  let currentX = 50;

  for (let i = 0; i < stations.length; i++) {
    const st = stations[i];
    
    // Determine station capacity from connected edges
    const connectedEdges = topology.edges.filter(e => 
      e.source === st.id || e.target === st.id || e.from === st.id || e.to === st.id
    );
    const maxCap = connectedEdges.reduce((max, e) => Math.max(max, e.capacity || 2), 2);
    const stCap = st.capacity || maxCap;

    const stWidth = 140;
    zones.push({
      type: 'ST',
      x1: currentX,
      x2: currentX + stWidth,
      cap: stCap,
      stId: (st as any).stId || st.id,
      isLeft: i === 0,
      isRight: i === stations.length - 1,
    });
    
    currentX += stWidth;

    if (i < stations.length - 1) {
      const nextSt = stations[i + 1];
      const distKm = (nextSt.km || 0) - (st.km || 0);
      
      const segWidth = 100 + (distKm * 10); 
      
      // Find intermediate blocks to determine segment capacity & speed
      const intermediateBlocks = topology.nodes.filter(n => 
        (n.km || 0) > (st.km || 0) && (n.km || 0) < (nextSt.km || 0) &&
        (n.type === 'MAIN_BLOCK' || n.type === 'GHAT_BLOCK')
      );
      
      let segCap = 2;
      let segSpeed = 100;
      let isGhat = false;

      if (intermediateBlocks.length > 0) {
        segCap = intermediateBlocks[0].capacity || 2;
        segSpeed = Math.min(...intermediateBlocks.map(b => b.speed || 100));
        if (intermediateBlocks.some(b => b.type === 'GHAT_BLOCK')) {
          isGhat = true;
        }
      } else {
        segCap = stCap;
      }

      // Switch before segment
      if (stCap !== segCap && stCap > 1 && segCap > 0) {
        const swWidth = 44;
        zones.push({ type: 'SW', x1: currentX, x2: currentX + swWidth, fromCap: stCap, toCap: segCap });
        currentX += swWidth;
      }

      // Segment
      zones.push({
        type: 'SEG',
        x1: currentX,
        x2: currentX + segWidth,
        cap: segCap,
        speed: segSpeed,
        km: distKm,
        isGhat,
        startKm: st.km,
        endKm: nextSt.km,
      });
      currentX += segWidth;
      
      const nextStCap = nextSt.capacity || 2;
      
      if (segCap !== nextStCap && nextStCap > 1 && segCap > 0) {
        const swWidth = 44;
        zones.push({ type: 'SW', x1: currentX, x2: currentX + swWidth, fromCap: segCap, toCap: nextStCap });
        currentX += swWidth;
      }
    }
  }

  return zones;
}
