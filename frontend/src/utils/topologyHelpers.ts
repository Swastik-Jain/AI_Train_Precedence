import type { TopologyData } from '../store/useMapStore';

export const getNodeStId = (node: any) => node?.station || node?.stId || node?.id;

export function isIntraStationMove(topology: TopologyData, nodeIdA: string, nodeIdB: string): boolean {
  const nodeA = topology.nodes.find(n => n.id === nodeIdA);
  const nodeB = topology.nodes.find(n => n.id === nodeIdB);
  
  if (!nodeA || !nodeB) return false;
  
  const stIdA = getNodeStId(nodeA);
  const stIdB = getNodeStId(nodeB);
  
  return stIdA === stIdB && stIdA != null;
}
