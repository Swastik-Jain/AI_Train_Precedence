import React, { useEffect, useMemo, useState } from 'react';
import { useMapStore } from '../../store/useMapStore';
import { useCopilotStore } from '../../store/useCopilotStore';
import type { TrainState } from '../../store/useMapStore';
import './KineticMap.css';

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
const SVG_W     = 3400;
const SVG_H     = 440;
const MAIN_Y    = 220;
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
type SegZone     = { type:'SEG'; x1:number; x2:number; cap:number; isGhat?:boolean; speed?:number; km?:number };
type SwitchZone  = { type:'SW';  x1:number; x2:number; fromCap:number; toCap:number };
type StationZone = { type:'ST';  x1:number; x2:number; cap:number; stId:string; isLeft?:boolean; isRight?:boolean };
type Zone        = SegZone | SwitchZone | StationZone;

const ZONES: Zone[] = [
  // CSMT terminus (4 tracks, compact 80 px box)
  { type:'ST',  x1:50,   x2:130,  cap:4, stId:'CSMT',     isLeft:true  },
  // CSMT → DADAR (9 km — 200 px)
  { type:'SEG', x1:130,  x2:330,  cap:4, speed:110, km:9  },
  // DADAR station (90 px)
  { type:'ST',  x1:330,  x2:420,  cap:4, stId:'DADAR'                   },
  // DADAR → KALYAN (45 km — 450 px)
  { type:'SEG', x1:420,  x2:870,  cap:4, speed:130, km:45 },
  // KALYAN JN (100 px; exits via 4→2 switch)
  { type:'ST',  x1:870,  x2:970,  cap:4, stId:'KALYAN'                  },
  { type:'SW',  x1:970,  x2:1050, fromCap:4, toCap:2                    },
  // KALYAN → KASARA (67 km — 470 px)
  { type:'SEG', x1:1050, x2:1520, cap:2, speed:100, km:67 },
  // KASARA station (80 px; exits via 2→1 for ghat)
  { type:'ST',  x1:1520, x2:1600, cap:2, stId:'KASARA'                  },
  { type:'SW',  x1:1600, x2:1644, fromCap:2, toCap:1                    },
  // Ghat — TOKEN BLOCK single-track (120 px = 15 km)
  { type:'SEG', x1:1644, x2:1764, cap:1, isGhat:true, speed:50, km:15  },
  { type:'SW',  x1:1764, x2:1808, fromCap:1, toCap:2                    },
  // IGATPURI station (80 px)
  { type:'ST',  x1:1808, x2:1888, cap:2, stId:'IGATPURI'               },
  // IGATPURI → DEVLALI (46 km — 400 px)
  { type:'SEG', x1:1888, x2:2288, cap:2, speed:110, km:46 },
  // DEVLALI station (70 px)
  { type:'ST',  x1:2288, x2:2358, cap:2, stId:'DEVLALI'                },
  // DEVLALI → NASHIK (5 km — 150 px)
  { type:'SEG', x1:2358, x2:2508, cap:2, speed:75,  km:5  },
  // NASHIK station (90 px)
  { type:'ST',  x1:2508, x2:2598, cap:2, stId:'NASHIK'                 },
  // NASHIK → MANMAD (74 km — 600 px, longest section)
  { type:'SEG', x1:2598, x2:3198, cap:2, speed:130, km:74 },
  // MANMAD terminus (2 tracks, 80 px)
  { type:'ST',  x1:3198, x2:3278, cap:2, stId:'MANMAD', isRight:true  },
];

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
  CSMT:     { label:'CSMT',        km:0,   loops:2, passing:false, loopLeft:'bumper',   loopRight:'segment' },
  DADAR:    { label:'DADAR',       km:9,   loops:1, passing:false, loopLeft:'segment',  loopRight:'segment' },
  KALYAN:   { label:'KALYAN JN',   km:54,  loops:2, passing:false, loopLeft:'segment',  loopRight:'inside'  },
  KASARA:   { label:'KASARA',      km:121, loops:1, passing:true,  loopLeft:'segment',  loopRight:'inside'  },
  IGATPURI: { label:'IGATPURI',    km:136, loops:1, passing:true,  loopLeft:'inside',   loopRight:'segment' },
  DEVLALI:  { label:'DEVLALI',     km:182, loops:1, passing:false, loopLeft:'segment',  loopRight:'segment' },
  NASHIK:   { label:'NASHIK ROAD', km:187, loops:2, passing:false, loopLeft:'segment',  loopRight:'segment' },
  MANMAD:   { label:'MANMAD JN',   km:261, loops:1, passing:false, loopLeft:'segment',  loopRight:'bumper'  },
};

// ─────────────────────────────────────────────────────────────────────────────
// BACKEND X → SCHEMATIC X  (piecewise-linear, topology is deterministic)
// ─────────────────────────────────────────────────────────────────────────────
const B2S: [number,number,number,number][] = [
  [100,   250,  20,  50 ],  // ORIGIN → CSMT sw_in
  [250,   430,  50,  130],  // CSMT station zone
  [430,   880,  130, 330],  // CSMT → DADAR segment
  [880,   1060, 330, 420],  // DADAR station zone
  [1060,  2560, 420, 870],  // DADAR → KALYAN segment
  [2560,  2740, 870, 970],  // KALYAN station zone
  [2740,  4840, 970, 1520], // KALYAN switch + KALYAN→KASARA segment
  [4840,  5020, 1520,1600], // KASARA station zone
  [5020,  5620, 1600,1808], // KASARA switch + ghat + IGATPURI switch
  [5620,  5800, 1808,1888], // IGATPURI station zone
  [5800,  7300, 1888,2288], // IGATPURI → DEVLALI segment
  [7300,  7480, 2288,2358], // DEVLALI station zone
  [7480,  7780, 2358,2508], // DEVLALI → NASHIK segment
  [7780,  7960, 2508,2598], // NASHIK station zone
  [7960,  10360,2598,3198], // NASHIK → MANMAD segment
  [10360, 10540,3198,3278], // MANMAD station zone
  [10540, 10690,3278,3360], // MANMAD → DEST
];

const bx2sx = (bx: number): number => {
  for (const [b1, b2, s1, s2] of B2S) {
    if (bx >= b1 && bx <= b2) {
      return s1 + ((bx - b1) / (b2 - b1)) * (s2 - s1);
    }
  }
  return bx < 100 ? 20 : 3400;
};

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENT
// ─────────────────────────────────────────────────────────────────────────────
export const KineticMap: React.FC = () => {
  const {
    topology, trainStates, conflicts,
    connectWebSocket, setSelectedTrain,
    selectedTrainId, committedTrainId, committedAction, zoomLevel,
  } = useMapStore();

  const previewState    = useCopilotStore(s => s.previewState);
  const aiAffectedEdges = useMemo(() => new Set(previewState?.affected_edges ?? []), [previewState]);

  const [hoveredTrain, setHoveredTrain] = useState<string | null>(null);

  useEffect(() => { connectWebSocket(); }, [connectWebSocket]);

  // ── Node → schematic position ──────────────────────────────────────────────
  const nodePos = useMemo(() => {
    const map = new Map<string, { x: number; y: number }>();
    if (!topology) return map;
    topology.nodes.forEach(n => {
      map.set(n.id, {
        x: bx2sx(n.x),
        y: MAIN_Y,
      });
    });
    return map;
  }, [topology]);

  // ── Train schematic position ───────────────────────────────────────────────
  const getPos = (train: TrainState): { x: number; y: number } | null => {
    if (!topology) return null;
    const edge = topology.edges.find(e => e.id === train.edge_id);
    if (!edge) return null;
    const src = nodePos.get(edge.source);
    const tgt = nodePos.get(edge.target);
    if (!src || !tgt) return null;
    return {
      x: src.x + (tgt.x - src.x) * train.position_percentage,
      y: MAIN_Y,
    };
  };

  // ── Determine which track a train occupies at a station (hash-based) ───────
  const trainTrackAt = (train: TrainState, cap: number): number => {
    let hash = 0;
    for (let i = 0; i < train.train_id.length; i++) hash += train.train_id.charCodeAt(i);
    return hash % cap;
  };

  // ── Block tick marks: actual topology block boundaries ──────────────────────
  const blockTicks = useMemo(() => {
    if (!topology) return [] as { x: number; cap: number }[];
    const result: { x: number; cap: number }[] = [];
    topology.nodes.forEach(n => {
      if (n.type !== 'MAIN_BLOCK' && n.type !== 'GHAT_BLOCK') return;
      const sx = bx2sx(n.x);
      // Find the SEG zone this node falls inside
      const seg = ZONES.find(
        z => z.type === 'SEG' && sx > (z as SegZone).x1 && sx < (z as SegZone).x2
      ) as SegZone | undefined;
      if (seg) result.push({ x: sx, cap: seg.cap });
    });
    return result;
  }, [topology]);

  // ── Render: SEG ─────────────────────────────────────────────────────────────
  const renderSeg = (z: SegZone, key: number, blockXs: number[] = []) => {
    const elems: React.ReactNode[] = [];
    const mx  = (z.x1 + z.x2) / 2;
    const w   = z.x2 - z.x1;
    const GAP = 3;  // visible gap at each block boundary (px)

    // Draw each block section as a separate line with a small gap at boundaries
    const xs = [z.x1, ...blockXs.slice().sort((a, b) => a - b), z.x2];
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
          x={z.x1} y={trackY(0, 1) - 6} width={z.x2 - z.x1} height={12}
          fill="rgba(58,112,144,0.06)" stroke="#3a7090" strokeWidth={1}
          strokeDasharray="6 3" />,
      );
    }

    return <g key={`seg-${key}`}>{elems}</g>;
  };

  // ── Render: SWITCH ──────────────────────────────────────────────────────────
  const renderSwitch = (z: SwitchZone, key: number) => {
    const { fromCap, toCap, x1, x2 } = z;
    const elems: React.ReactNode[] = [];
    const converge = fromCap >= toCap;

    if (converge) {
      // Convergence: iterate from-tracks
      for (let i = 0; i < fromCap; i++) {
        const j = mapIdx(fromCap, toCap, i);
        elems.push(
          <line key={i}
            x1={x1} y1={trackY(i, fromCap)}
            x2={x2} y2={trackY(j, toCap)}
            stroke="#484848" strokeWidth={2} strokeLinecap="round"
          />
        );
      }
      // Merge markers at x2
      const seen = new Set<number>();
      for (let i = 0; i < fromCap; i++) {
        const j = mapIdx(fromCap, toCap, i);
        const mergeCount = Array.from({ length: fromCap }, (_, k) => k)
          .filter(k => mapIdx(fromCap, toCap, k) === j).length;
        if (mergeCount > 1 && !seen.has(j)) {
          seen.add(j);
          const sy = trackY(j, toCap);
          elems.push(
            <rect key={`swm${j}`} x={x2-4} y={sy-4} width={8} height={8}
              fill="#5a5a5a" stroke="#777" strokeWidth={1} rx={1} />
          );
        }
      }
    } else {
      // Divergence: iterate to-tracks
      for (let j = 0; j < toCap; j++) {
        const i = mapIdx(toCap, fromCap, j);
        elems.push(
          <line key={j}
            x1={x1} y1={trackY(i, fromCap)}
            x2={x2} y2={trackY(j, toCap)}
            stroke="#484848" strokeWidth={2} strokeLinecap="round"
          />
        );
      }
      // Diverge markers at x1
      const seen = new Set<number>();
      for (let j = 0; j < toCap; j++) {
        const i = mapIdx(toCap, fromCap, j);
        const divCount = Array.from({ length: toCap }, (_, k) => k)
          .filter(k => mapIdx(toCap, fromCap, k) === i).length;
        if (divCount > 1 && !seen.has(i)) {
          seen.add(i);
          const sy = trackY(i, fromCap);
          elems.push(
            <rect key={`swd${i}`} x={x1-4} y={sy-4} width={8} height={8}
              fill="#5a5a5a" stroke="#777" strokeWidth={1} rx={1} />
          );
        }
      }
    }

    return <g key={`sw-${key}`}>{elems}</g>;
  };

  // ── Render: STATION ─────────────────────────────────────────────────────────
  const renderStation = (z: StationZone, zKey: number) => {
    const meta        = STATION_META[z.stId];
    const { x1, x2, cap } = z;
    const isTerminus  = !!(z.isLeft || z.isRight);
    const topTrackY   = trackY(0, cap);
    const botTrackY   = trackY(cap - 1, cap);
    const cx          = (x1 + x2) / 2;

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

    // 1) Station box — covers ONLY main platform tracks (not loop sidings)
    const boxTop = topTrackY - 7;
    const boxBot = botTrackY + 7;
    elems.push(
      <rect key="box"
        x={x1} y={boxTop} width={x2 - x1} height={boxBot - boxTop}
        fill="rgba(26,26,26,0.80)"
        stroke={isTerminus ? '#666' : '#3a3a3a'}
        strokeWidth={isTerminus ? 1.5 : 1}
        rx={2}
      />
    );

    // 2) Main track lines
    for (let i = 0; i < cap; i++) {
      const y = trackY(i, cap);
      elems.push(
        <line key={`mt${i}`} x1={x1} y1={y} x2={x2} y2={y}
          stroke="#484848" strokeWidth={2} strokeLinecap="square" />
      );
    }

    // 3) Loop / siding tracks — smooth bezier S-curves, branching outside the station box.
    //    loopLeft/loopRight  'segment' | 'inside' | 'bumper'
    //      'segment' → bezier arch extends LOOP_OFF px into the adjacent segment
    //      'inside'  → bezier stays within station (for capacity-change junctions)
    //      'bumper'  → loop terminates; bumper bar drawn below

    for (let l = 0; l < meta.loops; l++) {
      const sidY  = topTrackY - (l + 1) * TRACK_GAP;
      const parts: string[] = [];

      // ── LEFT side entry ──────────────────────────────────────
      if (meta.loopLeft === 'segment') {
        // Smooth S-arch entering from the left segment
        parts.push(
          `M ${x1 - LOOP_OFF} ${topTrackY}`,
          `C ${x1 - CP_OFF} ${topTrackY}, ${x1 - CP_OFF} ${sidY}, ${x1} ${sidY}`
        );
      } else if (meta.loopLeft === 'inside') {
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
      if (meta.loopRight === 'segment') {
        // Extends then arches back to main track in the right segment
        parts.push(
          `L ${x2} ${sidY}`,
          `C ${x2 + CP_OFF} ${sidY}, ${x2 + CP_OFF} ${topTrackY}, ${x2 + LOOP_OFF} ${topTrackY}`
        );
      } else if (meta.loopRight === 'inside') {
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
          stroke="#3c3c3c" strokeWidth={1.5}
          strokeLinecap="round" strokeLinejoin="round" />
      );
    }

    // Junction dots — where loop branches off or rejoins the main track
    if (meta.loops > 0) {
      const r = 2.5;
      if (meta.loopLeft  === 'segment') elems.push(<circle key="cl-s" cx={x1 - LOOP_OFF} cy={topTrackY} r={r} fill="#505050" />);
      if (meta.loopLeft  === 'inside')  elems.push(<circle key="cl-i" cx={x1}            cy={topTrackY} r={r} fill="#505050" />);
      if (meta.loopRight === 'segment') elems.push(<circle key="cr-s" cx={x2 + LOOP_OFF} cy={topTrackY} r={r} fill="#505050" />);
      if (meta.loopRight === 'inside')  elems.push(<circle key="cr-i" cx={x2}            cy={topTrackY} r={r} fill="#505050" />);
    }

    // 4) Individual platform markers (small rects above each main track)
    for (let i = 0; i < cap; i++) {
      const y = trackY(i, cap);
      const occ = trackOccupancy.get(i);
      const occTrain = occ;
      const isConflict = occTrain && conflicts.includes(occTrain.edge_id);
      const isHalted   = occTrain?.status === 'Halted';
      const pfFill   = !occTrain ? 'rgba(30,30,30,0.9)' : isConflict ? 'rgba(60,15,15,0.9)' : isHalted ? 'rgba(60,40,10,0.9)' : 'rgba(10,35,20,0.9)';
      const pfStroke = !occTrain ? '#3a3a3a' : isConflict ? '#ef4444' : isHalted ? '#f59e0b' : '#22c55e';

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

    // 5) Loop siding platform markers — centred on the horizontal section of the siding
    const lpHorizL = meta.loopLeft  === 'inside' ? x1 + LOOP_OFF : x1;
    const lpHorizR = meta.loopRight === 'inside' ? x2 - LOOP_OFF : x2;
    const lpCx = (lpHorizL + lpHorizR) / 2;
    for (let l = 0; l < meta.loops; l++) {
      const sidY = topTrackY - (l + 1) * TRACK_GAP;
      elems.push(
        <g key={`lpf${l}`}>
          <rect x={lpCx - PF_W / 2} y={sidY - PF_H - 2}
            width={PF_W} height={PF_H}
            fill="rgba(24,24,24,0.9)" stroke="#333" strokeWidth={1} rx={1} />
          <text x={lpCx} y={sidY - PF_H / 2 + 2}
            textAnchor="middle" className="sch-pf-label" fill="#444">
            LP{l + 1}
          </text>
        </g>
      );
    }

    // 6) Track number labels (left edge of box)
    for (let i = 0; i < cap; i++) {
      const y = trackY(i, cap);
      elems.push(
        <text key={`tn${i}`} x={x1 + 5} y={y + 4}
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
      if (meta.loopLeft === 'bumper') {
        for (let l = 0; l < meta.loops; l++) {
          const sidY = topTrackY - (l + 1) * TRACK_GAP;
          elems.push(<line key={`bll${l}`} x1={x1} y1={sidY-4} x2={x1} y2={sidY+4} stroke="#666" strokeWidth={2.5} strokeLinecap="round" />);
        }
      }
      if (meta.loopRight === 'bumper') {
        for (let l = 0; l < meta.loops; l++) {
          const sidY = topTrackY - (l + 1) * TRACK_GAP;
          elems.push(<line key={`blr${l}`} x1={x2} y1={sidY-4} x2={x2} y2={sidY+4} stroke="#666" strokeWidth={2.5} strokeLinecap="round" />);
        }
      }
    }

    return <g key={`st-${zKey}`}>{elems}</g>;
  };

  // ── Zone elements (computed once per topology state) ───────────────────────
  const zoneElems = ZONES.map((z, i) => {
    if (z.type === 'SEG') {
      const segsBlocks = blockTicks
        .filter(b => b.x > (z as SegZone).x1 && b.x < (z as SegZone).x2)
        .map(b => b.x);
      return renderSeg(z as SegZone, i, segsBlocks);
    }
    if (z.type === 'SW')  return renderSwitch(z as SwitchZone, i);
    if (z.type === 'ST')  return renderStation(z as StationZone, i);
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
        <span className="sch-hint">scroll · +/− zoom</span>
      </div>

      {/* ── SCROLLABLE MAP ──────────────────────────────────────────────── */}
      <div className="sch-scroll">
        <svg
          style={{
            width:     `${SVG_W * zoomLevel}px`,
            height:    `${SVG_H * zoomLevel}px`,
            minHeight: '100%',
            display:   'block',
          }}
          viewBox={`0 0 ${SVG_W} ${SVG_H}`}
          preserveAspectRatio="xMinYMid meet"
          overflow="visible"
        >
          {/* Zones (segments → switches → stations) */}
          {zoneElems}

          {/* Block section boundaries are shown as 3 px gaps within each SEG zone */}

          {/* ── TRAINS ────────────────────────────────────────────────── */}
          {trainStates.map(train => {
            const pos = getPos(train);
            if (!pos) return null;

            const isSel      = selectedTrainId  === train.train_id;
            const isCommit   = committedTrainId === train.train_id;
            const isHover    = hoveredTrain     === train.train_id;
            const isConflict = conflicts.includes(train.edge_id);
            const isHalted   = train.status === 'Halted';
            const isAI       = topology?.edges.some(
              e => e.id === train.edge_id && aiAffectedEdges.has(e.id)
            ) ?? false;

            const fill   = isConflict ? '#ef4444' : isHalted ? '#f59e0b' : isAI ? '#38bdf8' : '#22c55e';
            const txtCol = '#e2e8f0';
            const bW     = 32;   // train badge width
            const bH     = 14;   // train badge height

            return (
              <g
                key={train.train_id}
                onClick={() => setSelectedTrain(train.train_id)}
                onMouseEnter={() => setHoveredTrain(train.train_id)}
                onMouseLeave={() => setHoveredTrain(null)}
                style={{ cursor: 'pointer' }}
              >
                {/* Conflict flash ring */}
                {isConflict && (
                  <rect
                    x={pos.x - bW / 2 - 4} y={pos.y - bH / 2 - 4}
                    width={bW + 8} height={bH + 8}
                    fill="none" stroke="#ef4444" strokeWidth={1}
                    strokeDasharray="3 2" rx={2}
                    className="sch-conflict-anim"
                  />
                )}

                {/* Committed ring */}
                {isCommit && (
                  <rect
                    x={pos.x - bW / 2 - 5} y={pos.y - bH / 2 - 5}
                    width={bW + 10} height={bH + 10}
                    fill="none" stroke="#22c55e" strokeWidth={1.5} rx={3}
                    className="sch-commit-anim"
                  />
                )}

                {/* Train badge */}
                <rect
                  x={pos.x - bW / 2} y={pos.y - bH / 2}
                  width={bW} height={bH}
                  fill={`${fill}22`}
                  stroke={fill}
                  strokeWidth={isSel ? 1.5 : 1}
                  rx={2}
                />

                {/* Train ID */}
                <text x={pos.x} y={pos.y + 4}
                  textAnchor="middle"
                  fill={fill}
                  className="sch-train-id"
                  opacity={isHover || isSel || isCommit ? 1 : 0.85}>
                  {train.train_id}
                </text>

                {/* Committed action tag */}
                {isCommit && (
                  <>
                    <rect
                      x={pos.x - 22} y={pos.y - bH / 2 - 15}
                      width={44} height={12}
                      fill="#22c55e" rx={2} />
                    <text x={pos.x} y={pos.y - bH / 2 - 5}
                      textAnchor="middle" className="sch-commit-tag">
                      {actionLabel}
                    </text>
                  </>
                )}
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
    </div>
  );
};
