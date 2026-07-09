import { apiUrl, wsUrl } from '../../lib/api';
import { motion } from 'framer-motion';
import React, { useEffect, useMemo, useState } from 'react';
import { useMapStore } from '../../store/useMapStore';
import { useCopilotStore } from '../../store/useCopilotStore';
import { useMaintenanceStore } from '../../store/useMaintenanceStore';
import type { TrainState, TopologyData, Node } from '../../store/useMapStore';
import { topologyToZones } from '../../utils/topologyToZones';
import type { Zone, SegZone, StationZone, SwitchZone } from '../../utils/topologyToZones';
import './KineticMap.css';

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
const SVG_H     = 400;
const MAIN_Y    = 200;
const TRACK_GAP = 22;   // px between parallel track centres
const LOOP_OFF  = TRACK_GAP * 2;  // 44 px — how far into segment the loop arch extends
const CP_OFF    = LOOP_OFF  / 2;  // 22 px — bezier S-curve control point
const PF_W      = 28;   // platform marker width
const PF_H      = 10;   // platform marker height

/** Y coordinate of track index i in an N-track bundle (centred on MAIN_Y) */
const trackY = (i: number, n: number): number =>
  MAIN_Y + (i - (n - 1) / 2) * TRACK_GAP;

/**
 * Map a from-track index to the nearest to-track index during a switch.
 *  fromCap → toCap : j = round(i × (toCap-1) / (fromCap-1))
 */
const mapIdx = (srcCap: number, dstCap: number, srcIdx: number): number => {
  if (srcCap <= 1 || dstCap <= 1) return 0;
  return Math.round(srcIdx * (dstCap - 1) / (srcCap - 1));
};

// ─────────────────────────────────────────────────────────────────────────────
// ZONE DEFINITIONS  (pre-computed layout along the x-axis)
// ─────────────────────────────────────────────────────────────────────────────
// Types and Zones are imported from topologyToZones

// ─────────────────────────────────────────────────────────────────────────────
// STATION METADATA
// ─────────────────────────────────────────────────────────────────────────────
type LoopSide = 'segment' | 'inside' | 'bumper';
interface StMeta {
  label: string; km: number; loops: number; passing: boolean;
  /** 'segment' = bezier extends into adjacent segment
   *  'inside'  = bezier stays within station box (for capacity-change junctions)
   *  'bumper'  = loop terminates with a bumper bar */
  loopLeft: LoopSide; loopRight: LoopSide;
}
const STATION_META: Record<string, StMeta> = {
  //                                                               loopLeft     loopRight
  CSMT:           { label:'CSMT',        km:0,   loops:2, passing:false, loopLeft:'bumper',   loopRight:'segment' },
  DADAR:          { label:'DADAR',       km:9,   loops:1, passing:false, loopLeft:'segment',  loopRight:'segment' },
  KALYAN:         { label:'KALYAN JN',   km:54,  loops:3, passing:false, loopLeft:'segment',  loopRight:'inside'  },
  AMBERNATH:      { label:'AMBERNATH',   km:63,  loops:2, passing:true,  loopLeft:'segment',  loopRight:'segment' },
  TITWALA:        { label:'TITWALA',     km:80,  loops:2, passing:true,  loopLeft:'segment',  loopRight:'segment' },
  ATGAON:         { label:'ATGAON',      km:98,  loops:2, passing:true,  loopLeft:'segment',  loopRight:'segment' },
  KASARA:         { label:'KASARA',      km:121, loops:3, passing:true,  loopLeft:'segment',  loopRight:'inside'  },
  IGATPURI:       { label:'IGATPURI',    km:136, loops:4, passing:true,  loopLeft:'inside',   loopRight:'segment' },
  DEVLALI:        { label:'DEVLALI',     km:182, loops:2, passing:false, loopLeft:'segment',  loopRight:'segment' },
  NASHIK:         { label:'NASHIK ROAD', km:187, loops:3, passing:false, loopLeft:'segment',  loopRight:'segment' },
  LOOP_NANDGAON:  { label:'NANDGAON',    km:210, loops:2, passing:true,  loopLeft:'segment',  loopRight:'segment' },
  LOOP_LASALGAON: { label:'LASALGAON',   km:235, loops:2, passing:true,  loopLeft:'segment',  loopRight:'segment' },
  MANMAD:         { label:'MANMAD JN',   km:261, loops:2, passing:false, loopLeft:'segment',  loopRight:'bumper'  },
};

// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENT
// ─────────────────────────────────────────────────────────────────────────────
const TrainBadge = ({ train, getPos, isSel, isCommit, isHover, isConflict, isHalted, isAI, tickIntervalS, actionLabel, setHoveredTrain, setSelectedTrain }: any) => {
  const pos = getPos(train);
  if (!pos) return null;

  const fill   = isConflict ? '#ef4444' : isHalted ? '#f59e0b' : isAI ? '#38bdf8' : '#22c55e';
  const bW     = 50;
  const bH     = 14;
  const isUp = train.direction === "UP" || train.direction === -1 || train.direction === 1;
  const trainX = isUp ? pos.x : pos.x - bW;

  return (
    <motion.g
      onClick={(e: any) => { e.stopPropagation(); setSelectedTrain(train.train_id); }}
      onMouseEnter={() => setHoveredTrain(train.train_id)}
      onMouseLeave={() => setHoveredTrain(null)}
      initial={false}
      animate={{ x: trainX, y: pos.y }}
      transition={{ 
        x: { type: "tween", duration: tickIntervalS, ease: "linear" },
        y: { type: "spring", stiffness: 90, damping: 15 } 
      }}
      style={{ cursor: 'pointer' }}
    >
      {isConflict && <rect x={-4} y={-bH / 2 - 4} width={bW + 8} height={bH + 8} fill="none" stroke="#ef4444" strokeWidth={1} strokeDasharray="3 2" rx={2} className="sch-conflict-anim" />}
      {isCommit && <rect x={-5} y={-bH / 2 - 5} width={bW + 10} height={bH + 10} fill="none" stroke="#22c55e" strokeWidth={1.5} rx={3} className="sch-commit-anim" />}
      <rect x={0} y={-bH / 2} width={bW} height={bH} fill={`${fill}22`} stroke={fill} strokeWidth={isSel ? 1.5 : 1} rx={2} />
      <text x={bW / 2} y={4} textAnchor="middle" fill={fill} className="sch-train-id" opacity={isHover || isSel || isCommit ? 1 : 0.85}>{train.train_id}</text>
      {isCommit && (
        <>
          <rect x={bW / 2 - 22} y={-bH / 2 - 15} width={44} height={12} fill="#22c55e" rx={2} />
          <text x={bW / 2} y={-bH / 2 - 5} textAnchor="middle" className="sch-commit-tag">{actionLabel}</text>
        </>
      )}
    </motion.g>
  );
};

export const KineticMap: React.FC = () => {
  const {
    topology, trainStates, conflicts,
    connectWebSocket, setSelectedTrain,
    selectedTrainId, committedTrainId, committedAction, zoomLevel,
    tickIntervalS,
  } = useMapStore();

  const dynamicZones = useMemo(() => {
    if (!topology || !topology.nodes || !topology.edges) return [];
    return topologyToZones(topology);
  }, [topology]);

  /**
   * Map a km value to a schematic X coordinate.
   * Uses the dynamicZones station zones as anchors.
   * Station node types in topology are 'SWITCH', 'CROSSING_LOOP', 'TERMINUS',
   * 'MAJOR_JUNCTION', 'STATION', 'ORIGIN', 'DESTINATION'.
   */
  const getSxForKm = (km: number): number => {
    if (!dynamicZones.length || !topology) return 20;

    // Build a sorted list of unique km→schematic-x anchors from station zones.
    // Each ST zone has a known km (from STATION_META or the station node itself).
    const stZones = dynamicZones.filter(z => z.type === 'ST') as StationZone[];
    if (stZones.length === 0) return 20;

    // Build km→x1 anchors for each unique station zone
    // (prefer the STATION_META km if available, else use topology node km)
    const stationTypes = [
      'SWITCH', 'CROSSING_LOOP', 'TERMINUS', 'MAJOR_JUNCTION',
      'STATION', 'ORIGIN', 'DESTINATION', 'PLATFORM', 'LOOP',
    ];
    const stationNodes = topology.nodes
      .filter(n => stationTypes.includes(n.type) && (n.km !== undefined && n.km !== null))
      .sort((a, b) => (a.km || 0) - (b.km || 0));

    // Build sorted anchor list: { km, sx }
    // For each ST zone, resolve the km from STATION_META or topology
    const anchors: { km: number; sx: number }[] = [];
    for (const z of stZones) {
      const meta = STATION_META[z.stId];
      const zKm = meta ? meta.km : (stationNodes.find(n => {
        const nStId = (n as any).stId || n.id;
        return nStId === z.stId;
      })?.km ?? null);
      if (zKm !== null && zKm !== undefined) {
        anchors.push({ km: zKm, sx: z.x1 });
      }
    }
    anchors.sort((a, b) => a.km - b.km);

    if (anchors.length === 0) return 20;
    if (km <= anchors[0].km) return anchors[0].sx;
    if (km >= anchors[anchors.length - 1].km) return anchors[anchors.length - 1].sx;

    // Interpolate between bracketing anchors
    for (let i = 0; i < anchors.length - 1; i++) {
      if (km >= anchors[i].km && km <= anchors[i + 1].km) {
        const span = anchors[i + 1].km - anchors[i].km;
        if (span === 0) return anchors[i].sx;
        const t = (km - anchors[i].km) / span;
        return anchors[i].sx + t * (anchors[i + 1].sx - anchors[i].sx);
      }
    }
    return 20;
  };

  const nodeSx = (node: { km: number; id: string; type: string }): number => {
    const km = node.km || 0;
    if (!dynamicZones.length || !topology) return getSxForKm(km);

    if (node.type === 'MAIN_BLOCK' || node.type === 'GHAT_BLOCK') {
      const seg = dynamicZones.find(z => z.type === 'SEG' && z.startKm !== undefined && z.endKm !== undefined && km >= z.startKm && km <= z.endKm) as SegZone | undefined;
      if (seg && seg.startKm !== undefined && seg.endKm !== undefined) {
        const span = seg.endKm - seg.startKm;
        if (span === 0) return seg.x1;
        return seg.x1 + ((km - seg.startKm) / span) * (seg.x2 - seg.x1);
      }
    }

    const stationTypes = ['SWITCH', 'CROSSING_LOOP', 'TERMINUS', 'MAJOR_JUNCTION', 'STATION', 'ORIGIN', 'DESTINATION', 'PLATFORM', 'LOOP'];
    if (stationTypes.includes(node.type)) {
      const stZones = dynamicZones.filter(z => z.type === 'ST') as StationZone[];
      const stZone = stZones.find(z => {
         const meta = STATION_META[z.stId];
         const zKm = meta ? meta.km : (topology.nodes.find(n => ((n as any).stId || n.id) === z.stId)?.km ?? null);
         return zKm === km;
      });

      if (stZone) {
         if (node.type === 'SWITCH') {
            const connectedEdges = topology.edges.filter(e => e.source === node.id || e.target === node.id);
            let connectsLeft = false;
            let connectsRight = false;
            const thisStId = (node as any).stId;
            
            for (const e of connectedEdges) {
              const otherId = e.source === node.id ? e.target : e.source;
              const otherNode = topology.nodes.find(n => n.id === otherId);
              if (!otherNode) continue;
              if ((otherNode as any).stId !== thisStId) {
                if (e.target === node.id) connectsLeft = true;
                if (e.source === node.id) connectsRight = true;
              }
            }
            if (connectsLeft && !connectsRight) return stZone.x1;
            if (connectsRight && !connectsLeft) return stZone.x2;
         }
         return (stZone.x1 + stZone.x2) / 2;
      }
    }
    return getSxForKm(km);
  };

  const previewState    = useCopilotStore(s => s.previewState);
  const aiAffectedEdges = useMemo(() => new Set(previewState?.affected_edges ?? []), [previewState]);
  const activeBlocks    = useMaintenanceStore(s => s.activeBlocks);

  const [hoveredTrain, setHoveredTrain] = useState<string | null>(null);
  const [selectedZone, setSelectedZone] = useState<Zone | null>(null);

  const selectedTrain = useMemo(() => 
    trainStates.find(t => t.train_id === selectedTrainId),
  [trainStates, selectedTrainId]);

  useEffect(() => { connectWebSocket(); }, [connectWebSocket]);

  const handleForceAction = async (trainId: string, action: number) => {
    try {
      const resp = await fetch(apiUrl('/api/v1/dispatch/force-action'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ train_id: trainId, action, duration_ticks: 50 })
      });
      if (!resp.ok) {
        console.error('Failed to force action');
      }
    } catch (err) {
      console.error(err);
    }
  };

  // ── Node → schematic position ──────────────────────────────────────────────
  // Uses km-based mapping (nodeSx) instead of the BFS layout x coordinate,
  // because BFS x values are arbitrary and don't correspond to schematic zones.
  const nodePos = useMemo(() => {
    const map = new Map<string, { x: number; y: number }>();
    if (!topology || !dynamicZones.length) return map;
    topology.nodes.forEach(n => {
      map.set(n.id, {
        x: nodeSx(n),
        y: MAIN_Y,
      });
    });
    return map;
  }, [topology, dynamicZones]);

  // ── Train schematic position ───────────────────────────────────────────────
  /**
   * Resolve the track-capacity used for the Y-coordinate of a train.
   *
   * Priority:
   *  1. MAIN_BLOCK / GHAT_BLOCK source nodes carry the exact segment cap.
   *  2. PLATFORM / LOOP edges: cap = 1 (siding track).
   *  3. SWITCH–SWITCH edges (intra-station main): use the adjacent SEG zone cap
   *     so the train sits on the same line as the through-running tracks.
   *  4. Fallback: nearest SEG zone at position x, or 2.
   *
   * We deliberately avoid using SW zone cap (= max of both sides) because that
   * places the badge above/below the actual drawn track lines.
   */
  const resolveEdgeCap = (edgeId: string, posX: number): number => {
    const edge = topology!.edges.find(e => e.id === edgeId);
    if (!edge) return 2;

    const srcNode = topology!.nodes.find(n => n.id === edge.source);
    const tgtNode = topology!.nodes.find(n => n.id === edge.target);

    // MAIN_BLOCK / GHAT_BLOCK — source node holds the correct segment capacity
    if (srcNode && (srcNode.type === 'MAIN_BLOCK' || srcNode.type === 'GHAT_BLOCK')) {
      return srcNode.capacity || 2;
    }
    if (tgtNode && (tgtNode.type === 'MAIN_BLOCK' || tgtNode.type === 'GHAT_BLOCK')) {
      return tgtNode.capacity || 2;
    }

    // PLATFORM / LOOP edges — siding, always single-track
    if (
      (srcNode && (srcNode.type === 'PLATFORM' || srcNode.type === 'LOOP')) ||
      (tgtNode && (tgtNode.type === 'PLATFORM' || tgtNode.type === 'LOOP'))
    ) {
      return 1;
    }

    // SWITCH–SWITCH (intra-station) or ORIGIN/DESTINATION edges:
    // Use the nearest SEG zone's cap so the train rides on the mainline tracks.
    const nearestSeg = dynamicZones
      .filter(z => z.type === 'SEG')
      .reduce<{ zone: SegZone | null; dist: number }>(
        (best, z) => {
          const seg = z as SegZone;
          const cx = (seg.x1 + seg.x2) / 2;
          const dist = Math.abs(cx - posX);
          return dist < best.dist ? { zone: seg, dist } : best;
        },
        { zone: null, dist: Infinity }
      );
    if (nearestSeg.zone) return nearestSeg.zone.cap;

    return 2;
  };

  const getPos = (train: TrainState): { x: number; y: number } | null => {
    if (!topology) return null;
    const edge = topology.edges.find(e => e.id === train.edge_id);
    if (!edge) return null;
    const srcNode = topology.nodes.find(n => n.id === edge.source);
    const tgtNode = topology.nodes.find(n => n.id === edge.target);
    const src = nodePos.get(edge.source);
    const tgt = nodePos.get(edge.target);
    if (!src || !tgt || !srcNode || !tgtNode) return null;

    const x = src.x + (tgt.x - src.x) * train.position_percentage;
    let currentY = MAIN_Y;

    if (srcNode.type === 'PLATFORM' && srcNode.platform_index !== undefined) {
        const stZone = dynamicZones.find(z => z.type === 'ST' && (z as StationZone).stId === (srcNode as any).stId) as StationZone | undefined;
        currentY = trackY(srcNode.platform_index, stZone?.cap || 1);
    } else if ((srcNode.type === 'LOOP' || srcNode.type === 'CROSSING_LOOP') && srcNode.loop_index !== undefined) {
        const stZone = dynamicZones.find(z => z.type === 'ST' && (z as StationZone).stId === (srcNode as any).stId) as StationZone | undefined;
        currentY = trackY(0, stZone?.cap || 1) - (srcNode.loop_index + 1) * TRACK_GAP;
    } else if (tgtNode.type === 'PLATFORM' && tgtNode.platform_index !== undefined) {
        const stZone = dynamicZones.find(z => z.type === 'ST' && (z as StationZone).stId === (tgtNode as any).stId) as StationZone | undefined;
        currentY = trackY(tgtNode.platform_index, stZone?.cap || 1);
    } else if ((tgtNode.type === 'LOOP' || tgtNode.type === 'CROSSING_LOOP') && tgtNode.loop_index !== undefined) {
        const stZone = dynamicZones.find(z => z.type === 'ST' && (z as StationZone).stId === (tgtNode as any).stId) as StationZone | undefined;
        currentY = trackY(0, stZone?.cap || 1) - (tgtNode.loop_index + 1) * TRACK_GAP;
    } else {
        const cap = resolveEdgeCap(train.edge_id, x);
        const trackIdx = trainTrackAt(train, cap);
        currentY = trackY(trackIdx, cap);
    }

    return { x, y: currentY };
  };

  // ── Determine which track a train occupies (UP=top, DOWN=bottom) ───────────
  const trainTrackAt = (train: TrainState, cap: number): number => {
    if (cap <= 1) return 0;
    const isUp = train.direction === "UP" || train.direction === 1 || train.direction === -1; 
    
    const peers = trainStates
      .filter(t => t.edge_id === train.edge_id && t.direction === train.direction)
      .sort((a, b) => a.train_id.localeCompare(b.train_id));
      
    const myIndex = peers.findIndex(t => t.train_id === train.train_id);
    const order = Math.max(0, myIndex);
    
    if (isUp) {
      const available = Math.ceil(cap / 2);
      return order % available;
    } else {
      const available = Math.floor(cap / 2);
      return (cap - available) + (order % available);
    }
  };

  // ── Block tick marks: actual topology block boundaries ──────────────────────
  const blockTicks = useMemo(() => {
    if (!topology || !dynamicZones.length) return [] as { x: number; cap: number }[];
    const result: { x: number; cap: number }[] = [];
    topology.nodes.forEach(n => {
      if (n.type !== 'MAIN_BLOCK' && n.type !== 'GHAT_BLOCK') return;
      const sx = nodeSx(n); // use km-based lookup instead of BFS layout x
      // Find the SEG zone this node falls inside
      const seg = dynamicZones.find(
        z => z.type === 'SEG' && sx > (z as SegZone).x1 && sx < (z as SegZone).x2
      ) as SegZone | undefined;
      if (seg) result.push({ x: sx, cap: seg.cap });
    });
    return result;
  }, [topology]);

  // ── Render: SEG ─────────────────────────────────────────────────────────────
  const renderSeg = (z: SegZone, key: number, blockXs: number[] = []) => {
    const elems: React.ReactNode[] = [];
    
    // Check if adjacent stations encroach on this segment with their loops
    const prevZ = dynamicZones[key - 1];
    const nextZ = dynamicZones[key + 1];
    
    let drawX1 = z.x1;
    let drawX2 = z.x2;
    
    if (prevZ && prevZ.type === 'ST') {
      const meta = STATION_META[(prevZ as StationZone).stId];
      if (meta && meta.loopRight === 'segment') drawX1 += LOOP_OFF;
    }
    if (nextZ && nextZ.type === 'ST') {
      const meta = STATION_META[(nextZ as StationZone).stId];
      if (meta && meta.loopLeft === 'segment') drawX2 -= LOOP_OFF;
    }

    const mx  = (drawX1 + drawX2) / 2;
    const w   = drawX2 - drawX1;
    const GAP = 3;  // visible gap at each block boundary (px)

    // Draw each block section as a separate line with a small gap at boundaries
    const validBlocks = blockXs.filter(x => x > drawX1 && x < drawX2).sort((a, b) => a - b);
    const xs = [drawX1, ...validBlocks, drawX2];
    
    for (let bi = 0; bi < xs.length - 1; bi++) {
      const bx1 = xs[bi]     + (bi > 0           ? GAP : 0);
      const bx2 = xs[bi + 1] - (bi < xs.length-2 ? GAP : 0);
      for (let i = 0; i < z.cap; i++) {
        const y = trackY(i, z.cap);
        elems.push(
          <line key={`t${i}-b${bi}`}
            x1={bx1} y1={y} x2={bx2} y2={y}
            stroke={z.isGhat ? '#3a7090' : '#484848'}
            strokeWidth={z.isGhat ? 3 : 2}
            strokeLinecap="square"
          />
        );
      }
    }

    // Segment speed/distance label (only when wide enough)
    if (w >= 120 && !z.isGhat) {
      elems.push(
        <text key="lbl" x={mx} y={trackY(0, z.cap) - 9}
          textAnchor="middle" className="sch-seg-lbl">
          {z.speed}km/h · {z.km}km
        </text>
      );
    }

    // Ghat label and token block outline box
    if (z.isGhat) {
      elems.push(
        <text key="ghat" x={mx} y={trackY(0, 1) - 12}
          textAnchor="middle" className="sch-ghat-lbl">TOKEN BLOCK</text>,
        <rect key="ghbox"
          x={drawX1} y={trackY(0, 1) - 6} width={drawX2 - drawX1} height={12}
          fill="rgba(58,112,144,0.06)" stroke="#3a7090" strokeWidth={1}
          strokeDasharray="6 3" />,
      );
    }

    const yMin = trackY(0, z.cap) - 30;
    const yMax = trackY(z.cap - 1, z.cap) + 30;

    return (
      <g 
        key={`seg-${key}`}
        style={{ cursor: 'pointer' }}
        onClick={(e) => { e.stopPropagation(); setSelectedZone(z); setSelectedTrain(null); }}
      >
        <rect x={z.x1} y={yMin} width={Math.max(1, z.x2 - z.x1)} height={yMax - yMin} fill="transparent" />
        {elems}
      </g>
    );
  };

  // ── Render: SWITCH (Bezier Throat) ──────────────────────────────────────────
  const renderSwitch = (z: SwitchZone, key: number) => {
    const { fromCap, toCap, x1, x2 } = z;
    const elems: React.ReactNode[] = [];
    const cx = (x1 + x2) / 2;

    // Helper: Connects outer tracks gracefully to the nearest incoming track without crossing lines
    const getClosestSource = (destIdx: number, destTotal: number, srcTotal: number) => {
        return Math.min(Math.max(0, destIdx - Math.floor((destTotal - srcTotal) / 2)), srcTotal - 1);
    };

    if (fromCap >= toCap) {
      // Convergence (e.g., 4 tracks narrowing to 3)
      for (let i = 0; i < fromCap; i++) {
        const j = getClosestSource(i, fromCap, toCap);
        elems.push(
          <path key={`conv-${i}`}
            d={`M ${x1} ${trackY(i, fromCap)} C ${cx} ${trackY(i, fromCap)}, ${cx} ${trackY(j, toCap)}, ${x2} ${trackY(j, toCap)}`}
            fill="none" stroke="#484848" strokeWidth={2} strokeLinecap="round" />
        );
      }
      
      // Draw merge markers
      const seen = new Set<number>();
      for (let i = 0; i < fromCap; i++) {
        const j = getClosestSource(i, fromCap, toCap);
        const mergeCount = Array.from({ length: fromCap }, (_, k) => k).filter(k => getClosestSource(k, fromCap, toCap) === j).length;
        if (mergeCount > 1 && !seen.has(j)) {
          seen.add(j);
          elems.push(<rect key={`swm${j}`} x={x2-4} y={trackY(j, toCap)-4} width={8} height={8} fill="#5a5a5a" stroke="#777" strokeWidth={1} rx={1} />);
        }
      }
      
    } else {
      // Divergence (e.g., 2 tracks expanding to 4)
      for (let j = 0; j < toCap; j++) {
        const i = getClosestSource(j, toCap, fromCap);
        elems.push(
          <path key={`div-${j}`}
            d={`M ${x1} ${trackY(i, fromCap)} C ${cx} ${trackY(i, fromCap)}, ${cx} ${trackY(j, toCap)}, ${x2} ${trackY(j, toCap)}`}
            fill="none" stroke="#484848" strokeWidth={2} strokeLinecap="round" />
        );
      }
      
      // Draw diverge markers
      const seen = new Set<number>();
      for (let j = 0; j < toCap; j++) {
        const i = getClosestSource(j, toCap, fromCap);
        const divCount = Array.from({ length: toCap }, (_, k) => k).filter(k => getClosestSource(k, toCap, fromCap) === i).length;
        if (divCount > 1 && !seen.has(i)) {
          seen.add(i);
          elems.push(<rect key={`swd${i}`} x={x1-4} y={trackY(i, fromCap)-4} width={8} height={8} fill="#5a5a5a" stroke="#777" strokeWidth={1} rx={1} />);
        }
      }
    }

    return (
      <g key={`sw-${key}`} style={{ cursor: 'pointer' }} onClick={(e) => { e.stopPropagation(); setSelectedZone(z); setSelectedTrain(null); }}>
        {/* Invisible hit-box for clicking the zone */}
        <rect x={z.x1} y={MAIN_Y - 40} width={Math.max(1, z.x2 - z.x1)} height={80} fill="transparent" />
        {elems}
      </g>
    );
  };


  // ── Render: STATION ─────────────────────────────────────────────────────────
  const renderStation = (z: StationZone, zKey: number, allZones: Zone[]) => {
    const meta        = STATION_META[z.stId] || { label: z.stId, km: (z as any).km || 0, loops: 0, passing: false, loopLeft: 'inside', loopRight: 'inside' };
    const { x1, x2, cap } = z;
    const isTerminus  = !!(z.isLeft || z.isRight);
    const topTrackY   = trackY(0, cap);
    const botTrackY   = trackY(cap - 1, cap);
    
    // Dynamically retract loops into the station if there is a switch zone immediately adjacent
    const prevZone = allZones[zKey - 1];
    const nextZone = allZones[zKey + 1];
    const actualLoopLeft = prevZone?.type === 'SW' ? 'inside' : meta.loopLeft;
    const actualLoopRight = nextZone?.type === 'SW' ? 'inside' : meta.loopRight;

    // Visually expand the station boundaries to cover the loops if they branch in the segment
    const visualX1 = actualLoopLeft === 'segment' ? x1 - LOOP_OFF : x1;
    const visualX2 = actualLoopRight === 'segment' ? x2 + LOOP_OFF : x2;
    const cx       = (visualX1 + visualX2) / 2;

    // Which trains are at this station? (track occupancy)
    const stTrains = trainStates.filter(t => {
      const p = getPos(t);
      return p !== null && p.x >= x1 && p.x <= x2;
    });
    const trackOccupancy = new Map<number, TrainState>();
    stTrains.forEach(t => {
      const ti = trainTrackAt(t, cap);
      if (!trackOccupancy.has(ti)) trackOccupancy.set(ti, t);
    });

    const elems: React.ReactNode[] = [];

    // 1) Station box — covers BOTH main platform tracks AND loop sidings above
    const loopsTop = meta.loops > 0 ? topTrackY - meta.loops * TRACK_GAP - 7 : topTrackY - 7;
    const boxTop = loopsTop;
    const boxBot = botTrackY + 7;
    elems.push(
      <rect key="box"
        x={visualX1} y={boxTop} width={visualX2 - visualX1} height={boxBot - boxTop}
        fill="rgba(40,44,52,0.4)"
        stroke={isTerminus ? '#555' : '#333'}
        strokeWidth={isTerminus ? 1.5 : 1}
        rx={4}
      />
    );

    // 2) Main track lines
    for (let i = 0; i < cap; i++) {
      const y = trackY(i, cap);
      elems.push(
        <line key={`mt${i}`} x1={visualX1} y1={y} x2={visualX2} y2={y}
          stroke="#484848" strokeWidth={2} strokeLinecap="square" />
      );
    }

    // 3) Loop / siding tracks — smooth bezier S-curves above main tracks.
    //    We retain the straight middle section and use bezier entries/exits.
    for (let l = 0; l < meta.loops; l++) {
      const sidY  = topTrackY - (l + 1) * TRACK_GAP;
      const parts: string[] = [];

      // ── LEFT side entry ──────────────────────────────────────
      if (actualLoopLeft === 'segment') {
        // Smooth S-arch entering from the left segment
        parts.push(
          `M ${x1 - LOOP_OFF} ${topTrackY}`,
          `C ${x1 - CP_OFF} ${topTrackY}, ${x1 - CP_OFF} ${sidY}, ${x1} ${sidY}`
        );
      } else if (actualLoopLeft === 'inside') {
        // Diverges from main track just inside the station left boundary
        parts.push(
          `M ${x1} ${topTrackY}`,
          `C ${x1 + CP_OFF} ${topTrackY}, ${x1 + CP_OFF} ${sidY}, ${x1 + LOOP_OFF} ${sidY}`
        );
      } else {
        // 'bumper': starts at station left edge at siding level
        parts.push(`M ${x1} ${sidY}`);
      }

      // ── RIGHT side exit ──────────────────────────────────────
      if (actualLoopRight === 'segment') {
        // Extends then arches back to main track in the right segment
        parts.push(
          `L ${x2} ${sidY}`,
          `C ${x2 + CP_OFF} ${sidY}, ${x2 + CP_OFF} ${topTrackY}, ${x2 + LOOP_OFF} ${topTrackY}`
        );
      } else if (actualLoopRight === 'inside') {
        // Rejoins main track just before the station right boundary
        parts.push(
          `L ${x2 - LOOP_OFF} ${sidY}`,
          `C ${x2 - CP_OFF} ${sidY}, ${x2 - CP_OFF} ${topTrackY}, ${x2} ${topTrackY}`
        );
      } else {
        // 'bumper': ends at station right edge
        parts.push(`L ${x2} ${sidY}`);
      }

      elems.push(
        <path key={`sid${l}`} d={parts.join(' ')} fill="none"
          stroke="#404040" strokeWidth={1.5}
          strokeLinecap="round" strokeLinejoin="round"
          strokeDasharray="4 4" />
      );
    }

    // Junction dots — where loop branches off or rejoins the main track
    if (meta.loops > 0) {
      const r = 2.5;
      if (actualLoopLeft  === 'segment') elems.push(<circle key="cl-s" cx={x1 - LOOP_OFF} cy={topTrackY} r={r} fill="#505050" />);
      if (actualLoopLeft  === 'inside')  elems.push(<circle key="cl-i" cx={x1}            cy={topTrackY} r={r} fill="#505050" />);
      if (actualLoopRight === 'segment') elems.push(<circle key="cr-s" cx={x2 + LOOP_OFF} cy={topTrackY} r={r} fill="#505050" />);
      if (actualLoopRight === 'inside')  elems.push(<circle key="cr-i" cx={x2}            cy={topTrackY} r={r} fill="#505050" />);
    }

    // 4) Individual platform markers (small rects above each main track)
    for (let i = 0; i < cap; i++) {
      const y = trackY(i, cap);
      const occ = trackOccupancy.get(i);
      const occTrain = occ;
      const isConflict = occTrain && conflicts.includes(occTrain.edge_id);
      const isHalted   = occTrain?.status === 'Halted';
      const pfFill   = !occTrain ? 'rgba(35,38,45,0.6)' : isConflict ? 'rgba(60,15,15,0.9)' : isHalted ? 'rgba(60,40,10,0.9)' : 'rgba(10,35,20,0.9)';
      const pfStroke = !occTrain ? '#333' : isConflict ? '#ef4444' : isHalted ? '#f59e0b' : '#22c55e';

      elems.push(
        <g key={`pf${i}`}>
          <rect
            x={cx - PF_W / 2} y={y - PF_H - 2}
            width={PF_W} height={PF_H}
            fill={pfFill} stroke={pfStroke} strokeWidth={1} rx={1}
          />
          <text x={cx} y={y - PF_H / 2 + 2}
            textAnchor="middle" className="sch-pf-label"
            fill={!occTrain ? '#555' : pfStroke}>
            PF{i + 1}
          </text>
        </g>
      );
    }

    // 5) Loop siding platform markers — positioned above each loop track line
    const lpCx = cx; // Perfectly align with PF markers in the visually expanded box
    for (let l = 0; l < meta.loops; l++) {
      const sidY = topTrackY - (l + 1) * TRACK_GAP;
      elems.push(
        <g key={`lpf${l}`}>
          <rect x={lpCx - PF_W / 2} y={sidY - PF_H - 2}
            width={PF_W} height={PF_H}
            fill="rgba(35,38,45,0.6)" stroke="#333" strokeWidth={1} rx={1} />
          <text x={lpCx} y={sidY - PF_H / 2 + 2}
            textAnchor="middle" className="sch-pf-label" fill="#555">
            LP{l + 1}
          </text>
        </g>
      );
    }

    // 6) Track number labels (left edge of box)
    for (let i = 0; i < cap; i++) {
      const y = trackY(i, cap);
      elems.push(
        <text key={`tn${i}`} x={visualX1 + 5} y={y + 4}
          className="sch-track-num" textAnchor="start">{i + 1}</text>
      );
    }

    // 7) Station name (below box)
    elems.push(
      <text key="name" x={cx} y={boxBot + 16}
        textAnchor="middle"
        className={`sch-st-name${isTerminus ? ' sch-st-name--term' : ''}`}>
        {meta.label}
      </text>
    );

    // 8) Km
    elems.push(
      <text key="km" x={cx} y={boxBot + 28}
        textAnchor="middle" className="sch-st-km">
        {meta.km} km
      </text>
    );

    // 9) PASSING badge
    if (meta.passing) {
      elems.push(
        <text key="pass" x={cx} y={boxBot + 42}
          textAnchor="middle" className="sch-passing">▶ PASSING</text>
      );
    }

    // 10) Terminus bumpers — main tracks only
    const bumperX = z.isLeft ? x1 : z.isRight ? x2 : null;
    if (bumperX !== null) {
      for (let i = 0; i < cap; i++) {
        const y = trackY(i, cap);
        elems.push(
          <line key={`bm${i}`}
            x1={bumperX} y1={y - 5} x2={bumperX} y2={y + 5}
            stroke="#888" strokeWidth={3} strokeLinecap="round" />
        );
      }
    }

    // 11) Loop-end bumpers — only for 'bumper' sides
    if (meta.loops > 0) {
      if (actualLoopLeft === 'bumper') {
        for (let l = 0; l < meta.loops; l++) {
          const sidY = topTrackY - (l + 1) * TRACK_GAP;
          elems.push(<line key={`bll${l}`} x1={visualX1} y1={sidY-4} x2={visualX1} y2={sidY+4} stroke="#666" strokeWidth={2.5} strokeLinecap="round" />);
        }
      }
      if (actualLoopRight === 'bumper') {
        for (let l = 0; l < meta.loops; l++) {
          const sidY = topTrackY - (l + 1) * TRACK_GAP;
          elems.push(<line key={`blr${l}`} x1={visualX2} y1={sidY-4} x2={visualX2} y2={sidY+4} stroke="#666" strokeWidth={2.5} strokeLinecap="round" />);
        }
      }
    }

    return (
      <g 
        key={`st-${zKey}`}
        style={{ cursor: 'pointer' }}
        onClick={(e) => { e.stopPropagation(); setSelectedZone(z); setSelectedTrain(null); }}
      >
        {/* The station background box is already enough for clicking, but a transparent rect ensures it */}
        <rect x={visualX1} y={boxTop - 10} width={Math.max(1, visualX2 - visualX1)} height={boxBot - boxTop + 20} fill="transparent" />
        {elems}
      </g>
    );
  };

  // ── Zone elements (computed once per topology state) ───────────────────────
  const svgW = dynamicZones.length > 0 ? dynamicZones[dynamicZones.length - 1].x2 + 200 : 4200;

  const zoneElems = dynamicZones.map((z, i) => {
    if (z.type === 'SEG') {
      const segsBlocks = blockTicks
        .filter(b => b.x > (z as SegZone).x1 && b.x < (z as SegZone).x2)
        .map(b => b.x);
      return renderSeg(z as SegZone, i, segsBlocks);
    }
    if (z.type === 'SW')  return renderSwitch(z as SwitchZone, i);
    if (z.type === 'ST')  return renderStation(z as StationZone, i, dynamicZones);
    return null;
  });

  // ── Action label for committed train ──────────────────────────────────────
  const actionLabel =
    committedAction === 0 ? '■ STOP'
    : committedAction === 2 ? '⇌ DIVERT'
    : '✓ OK';

  return (
    <div className="kinetic-map-container">

      {/* ── HEADER ──────────────────────────────────────────────────────── */}
      <div className="sch-header">
        <span className="sch-live">
          <span className="sch-live-dot" />LIVE
        </span>
        <span className="sch-route">CSMT → MANMAD  /  261 km  /  {trainStates.length} trains</span>
        <div style={{ flex: 1 }} />
      </div>

      {/* ── SCROLLABLE MAP ──────────────────────────────────────────────── */}
      <div className="sch-scroll">
        <svg
          style={{
            width:     `${svgW * zoomLevel}px`,
            height:    `${SVG_H * zoomLevel}px`,
            minHeight: '100%',
            display:   'block',
          }}
          viewBox={`0 0 ${svgW} ${SVG_H}`}
          preserveAspectRatio="xMinYMid meet"
          overflow="visible"
          onClick={() => { setSelectedTrain(null); setSelectedZone(null); }}
        >
          <defs>
            <pattern id="maint-hatch-red" patternUnits="userSpaceOnUse" width="8" height="8" patternTransform="rotate(45)">
              <rect width="4" height="8" fill="#ef4444" opacity="0.3" />
            </pattern>
            <pattern id="maint-hatch-amber" patternUnits="userSpaceOnUse" width="8" height="8" patternTransform="rotate(45)">
              <rect width="4" height="8" fill="#f59e0b" opacity="0.3" />
            </pattern>
          </defs>

          {/* Zones (segments → switches → stations) */}
          {zoneElems}

          {/* ── TRAINS ────────────────────────────────────────────────── */}
          {trainStates.map(train => {
            const isSel      = selectedTrainId  === train.train_id;
            const isCommit   = committedTrainId === train.train_id;
            const isHover    = hoveredTrain     === train.train_id;
            const isConflict = conflicts.includes(train.edge_id);
            const isHalted   = train.status === 'Halted';
            const isAI       = topology?.edges.some(
              e => e.id === train.edge_id && aiAffectedEdges.has(e.id)
            ) ?? false;

            return (
              <TrainBadge
                key={train.train_id}
                train={train}
                getPos={getPos}
                isSel={isSel}
                isCommit={isCommit}
                isHover={isHover}
                isConflict={isConflict}
                isHalted={isHalted}
                isAI={isAI}
                tickIntervalS={tickIntervalS}
                actionLabel={actionLabel}
                setHoveredTrain={setHoveredTrain}
                setSelectedTrain={setSelectedTrain}
              />
            );
          })}

          {/* ── MAINTENANCE BLOCKS HIGHLIGHT ────────────────────────────── */}
          {Array.from(activeBlocks.values()).filter(blk => {
            if (!blk.start_time || !blk.end_time) return true;
            try {
              const now = new Date();
              const start = new Date(blk.start_time);
              const end = new Date(blk.end_time);
              return start <= now && now <= end;
            } catch (e) {
              return true;
            }
          }).map(blk => {
            if (!topology) return null;
            const edge = topology.edges.find(e => e.id === blk.element_id);
            if (!edge) return null;
            const src = nodePos.get(edge.source);
            const tgt = nodePos.get(edge.target);
            if (!src || !tgt) return null;

            const isTotal = blk.severity === 'TOTAL_BLOCK';
            const color = isTotal ? '#ef4444' : '#f59e0b';
            const bgFill = isTotal ? 'url(#maint-hatch-red)' : 'url(#maint-hatch-amber)';
            
            // Determine cap for the zone this block is in to set height
            const midX = (src.x + tgt.x) / 2;
            let cap = 2;
            const zone = dynamicZones.find(z => midX >= z.x1 && midX <= z.x2);
            if (zone && (zone.type === 'SEG' || zone.type === 'ST')) cap = zone.cap;
            
            const boxH = cap * TRACK_GAP + 10;
            const boxY = MAIN_Y - boxH / 2;
            
            return (
              <g key={`maint-${blk.blockId}`}>
                <rect
                  x={src.x} y={boxY}
                  width={tgt.x - src.x} height={boxH}
                  fill={bgFill}
                  stroke={color} strokeWidth={1}
                  rx={2}
                />
              </g>
            );
          })}
        </svg>
      </div>

      {/* ── LEGEND ──────────────────────────────────────────────────────── */}
      <div className="sch-legend">
        <div className="sch-legend-item">
          <span className="sch-leg-line" style={{ background:'#484848' }} />
          <span>Main line</span>
        </div>
        <div className="sch-legend-item">
          <span className="sch-leg-line" style={{ background:'#3a7090' }} />
          <span>Token block (ghat)</span>
        </div>
        <div className="sch-legend-item">
          <span className="sch-leg-line" style={{ borderTop:'2px dashed #3a3a3a', background:'transparent' }} />
          <span>Siding / loop</span>
        </div>
        <div className="sch-legend-sep" />
        <div className="sch-legend-item">
          <span className="sch-leg-dot" style={{ background:'#22c55e' }} />
          <span>Moving</span>
        </div>
        <div className="sch-legend-item">
          <span className="sch-leg-dot" style={{ background:'#f59e0b' }} />
          <span>Halted</span>
        </div>
        <div className="sch-legend-item">
          <span className="sch-leg-dot" style={{ background:'#ef4444' }} />
          <span>Conflict</span>
        </div>
        <div className="sch-legend-item">
          <span className="sch-leg-dot" style={{ background:'#38bdf8' }} />
          <span>AI target</span>
        </div>
      </div>

      {/* ── COMMITTED-ACTION BANNER ─────────────────────────────────────── */}
      {committedTrainId && (
        <div className="sch-committed-banner">
          <span className="sch-committed-icon">
            {committedAction === 0 ? '■' : committedAction === 2 ? '⇌' : '✓'}
          </span>
          <div>
            <div className="sch-committed-action">
              {committedAction === 0 ? 'STOP' : committedAction === 2 ? 'DIVERT' : 'PROCEED'}
            </div>
            <div className="sch-committed-train">{committedTrainId}</div>
          </div>
        </div>
      )}

      {/* ── ZONE DETAIL PANEL ────────────────────────────────────────── */}
      {selectedZone && (
        <div className="sch-train-detail-panel" style={{ bottom: '20px', left: '20px', right: 'auto', top: 'auto', width: '280px' }}>
          <div className="sch-td-header">
            <span className="sch-td-title">
              {selectedZone.type === 'ST' ? `STATION: ${(selectedZone as StationZone).stId}` : 
               selectedZone.type === 'SEG' ? 'TRACK SEGMENT' : 'SWITCH'}
            </span>
            <button className="sch-td-close" onClick={() => setSelectedZone(null)}>×</button>
          </div>
          <div className="sch-td-content">
            <div className="sch-td-row">
              <span className="sch-td-label">Type</span>
              <span className="sch-td-value">{selectedZone.type === 'ST' ? 'Station' : selectedZone.type === 'SEG' ? 'Mainline Segment' : 'Crossover / Switch'}</span>
            </div>
            <div className="sch-td-row">
              <span className="sch-td-label">Tracks</span>
              <span className="sch-td-value">
                {selectedZone.type === 'SW' ? `${(selectedZone as SwitchZone).fromCap} ➔ ${(selectedZone as SwitchZone).toCap}` : (selectedZone as SegZone | StationZone).cap}
              </span>
            </div>
            {selectedZone.type === 'SEG' && (selectedZone as SegZone).km && (
              <div className="sch-td-row">
                <span className="sch-td-label">Distance</span>
                <span className="sch-td-value">{(selectedZone as SegZone).km} km</span>
              </div>
            )}
            {selectedZone.type === 'SEG' && (selectedZone as SegZone).speed && (
              <div className="sch-td-row">
                <span className="sch-td-label">Max Speed</span>
                <span className="sch-td-value">{(selectedZone as SegZone).speed} km/h</span>
              </div>
            )}
            {selectedZone.type === 'SEG' && (selectedZone as SegZone).isGhat && (
              <div className="sch-td-row">
                <span className="sch-td-label">Notes</span>
                <span className="sch-td-value" style={{ color: '#f59e0b' }}>Ghat Section (Token Block)</span>
              </div>
            )}
            {selectedZone.type === 'ST' && STATION_META[(selectedZone as StationZone).stId] && (
              <>
                <div className="sch-td-row">
                  <span className="sch-td-label">Km Mark</span>
                  <span className="sch-td-value">{STATION_META[(selectedZone as StationZone).stId].km} km</span>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* ── TRAIN DETAIL PANEL ────────────────────────────────────────── */}
      {selectedTrain && (
        <div className="sch-train-detail-panel" style={{ bottom: '20px', left: selectedZone ? '320px' : '20px', right: 'auto', top: 'auto', width: '280px' }}>
          <div className="sch-td-header">
            <span className="sch-td-title">TRAIN: {selectedTrain.train_id}</span>
            <button className="sch-td-close" onClick={() => setSelectedTrain(null)}>×</button>
          </div>
          <div className="sch-td-content">
            <div className="sch-td-row">
              <span className="sch-td-label">Status</span>
              <span className="sch-td-value">{selectedTrain.status}</span>
            </div>
            <div className="sch-td-row">
              <span className="sch-td-label">Direction</span>
              <span className="sch-td-value">{selectedTrain.direction}</span>
            </div>
            <div className="sch-td-row">
              <span className="sch-td-label">Edge ID</span>
              <span className="sch-td-value">{selectedTrain.edge_id}</span>
            </div>
            <div className="sch-td-row">
              <span className="sch-td-label">Position</span>
              <span className="sch-td-value">{(selectedTrain.position_percentage * 100).toFixed(1)}%</span>
            </div>
          </div>
          <div style={{ display: 'flex', gap: '8px', padding: '12px', borderTop: '1px solid #333' }}>
            <button 
              style={{ flex: 1, padding: '6px', background: '#ef4444', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '12px', fontWeight: 'bold' }}
              onClick={() => handleForceAction(selectedTrain.train_id, 0)}
            >
              Stop
            </button>
            <button 
              style={{ flex: 1, padding: '6px', background: '#22c55e', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '12px', fontWeight: 'bold' }}
              onClick={() => handleForceAction(selectedTrain.train_id, 1)}
            >
              Proceed
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
