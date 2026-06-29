export type SegZone     = { type:'SEG'; x1:number; x2:number; cap:number; isGhat?:boolean; speed?:number; km?:number };
export type SwitchZone  = { type:'SW';  x1:number; x2:number; fromCap:number; toCap:number };
export type StationZone = { type:'ST';  x1:number; x2:number; cap:number; stId:string; isLeft?:boolean; isRight?:boolean };
export type Zone        = SegZone | SwitchZone | StationZone;

/** Convert backend topology graph to KineticMap zones array */
export function topologyToZones(topology: {
  nodes: Array<{id: string; type: string; capacity?: number; km?: number; speed?: number}>;
  edges: Array<{id: string; source?: string; target?: string; from?: string; to?: string; length_px?: number; length?: number}>;
}): Zone[] {
  const zones: Zone[] = [];
  let currentX = 50;
  
  // Build chronologically from CSMT to Manmad
  const orderedNodes = ['CSMT', 'DADAR', 'KALYAN', 'KASARA', 'IGATPURI', 'JALNA', 'MANMAD'];
  
  for (let i = 0; i < orderedNodes.length; i++) {
    const node = topology.nodes.find(n => n.id === orderedNodes[i]);
    if (!node) continue;
    
    if (node.type === 'station' || node.type === 'TERMINUS' || node.type === 'MAJOR_JUNCTION') {
      zones.push({
        type: 'ST',
        x1: currentX,
        x2: currentX + 90,
        cap: node.capacity || 4,
        stId: node.id,
        isLeft: i === 0,
        isRight: i === orderedNodes.length - 1,
      });
      currentX += 90;
    } else if (node.type === 'section' && i < orderedNodes.length - 1) {
      zones.push({
        type: 'SEG',
        x1: currentX,
        x2: currentX + 400,
        cap: node.capacity || 4,
        speed: node.speed || 130,
        km: node.km || 30,
      });
      currentX += 400;
    }
  }
  
  return zones;
}
