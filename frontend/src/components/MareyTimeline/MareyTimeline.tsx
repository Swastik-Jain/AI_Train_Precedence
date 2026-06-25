import React, { useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useCopilotStore, STATION_DISTANCES, getSchematicY } from '../../store/useCopilotStore';
import { useMaintenanceStore } from '../../store/useMaintenanceStore';
import { useMapStore } from '../../store/useMapStore';
import type { ScheduleEntry } from '../../store/useCopilotStore';
import './MareyTimeline.css';

// ---------------------------------------------------------------------------
// SVG Canvas constants (viewBox 0 0 800 260)
// ---------------------------------------------------------------------------
const W = 800;
const H = 400;
const PAD_LEFT = 140;
const PAD_RIGHT = 20;
const PAD_TOP = 20;
const PAD_BTM = 28;

/** Build an SVG path `d` string from an array of {x,y} points */
function pointsToPath(pts: { x: number; y: number }[]): string {
  if (pts.length === 0) return '';
  return pts
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`)
    .join(' ');
}

/** Assign a color to a train line based on its ID */
function getTrainColor(trainId: string): string {
  if (trainId.includes('UP')) return '#3b82f6'; // blue-500
  if (trainId.includes('DN') || trainId.includes('DOWN')) return '#f59e0b'; // amber-500
  if (trainId.includes('EXP')) return '#10b981'; // emerald-500
  if (trainId.includes('FRT')) return '#64748b'; // slate-500
  
  const palette = ['#8B5CF6', '#ec4899', '#14b8a6', '#f43f5e', '#84cc16'];
  let hash = 0;
  for (let i = 0; i < trainId.length; i++) hash = trainId.charCodeAt(i) + ((hash << 5) - hash);
  return palette[Math.abs(hash) % palette.length];
}


interface MareyTimelineProps {
  scenarios?: any[];
  hideHeader?: boolean;
  hideTelemetry?: boolean;
}

// Component
// ---------------------------------------------------------------------------
export const MareyTimeline: React.FC<MareyTimelineProps> = ({ scenarios = [], hideHeader = false, hideTelemetry = false }) => {
  const { fetchBaseSchedule, globalSchedule, scheduleMaxTime } = useCopilotStore();
  const { activeBlocks } = useMaintenanceStore();
  const { trainStates, conflicts } = useMapStore();

  const currentSimTime = useMapStore(s => s.simTime) || 0;

  const totalW = W - PAD_LEFT - PAD_RIGHT;
  const timePct = scheduleMaxTime > 0 ? Math.min(Math.max(currentSimTime / scheduleMaxTime, 0), 1) : 0;
  const nowX = PAD_LEFT + (totalW * timePct);

  // Initial fetch of the schedule from OR-Tools backend
  React.useEffect(() => {
    fetchBaseSchedule();
  }, [fetchBaseSchedule]);

  // Combine globalSchedule with projected schedules from scenarios
  const combinedSchedule = useMemo(() => {
    const base = [...globalSchedule];
    if (scenarios.length > 0) {
      const latest = scenarios[scenarios.length - 1];
      if (latest.projected_schedule) {
        const totalH = H - PAD_TOP - PAD_BTM;
        
        Object.entries(latest.projected_schedule).forEach(([trainId, nodesObj]) => {
          const nodes = Object.entries(nodesObj as Record<string, {arrival: number, departure: number}>);
          if (nodes.length > 0) {
            nodes.sort((a, b) => a[1].arrival - b[1].arrival);
            const path = nodes.map(([nodeName, times]) => {
               const timeVal = times.arrival || times.departure;
               const timePct = scheduleMaxTime > 0 ? Math.min(Math.max(timeVal / scheduleMaxTime, 0), 1) : 0;
               const x = PAD_LEFT + (totalW * timePct);
               const km = STATION_DISTANCES[nodeName] || 0;
               const y = getSchematicY(km, totalH, PAD_TOP);
               return { x, y };
            });
            base.push({
              train_id: trainId,
              type: 'projected',
              path
            });
          }
        });
      }
    }
    return base;
  }, [globalSchedule, scenarios, scheduleMaxTime, totalW]);



  // MMS hatch bands: map block index to an X position across the timeline
  const maintenanceBands = useMemo(() => {
    const blocks = Array.from(activeBlocks.values());
    const totalW = W - PAD_LEFT - PAD_RIGHT;
    return blocks.map((blk, i) => ({
      blk,
      x: PAD_LEFT + (totalW / (blocks.length + 1)) * (i + 1) - 20,
      w: 40,
    }));
  }, [activeBlocks]);

  return (
    <div className="marey-container">
      {/* Header */}
      {!hideHeader && (
        <div className="marey-header">
          <div>
            <h2 className="marey-title">Marey Schedule Timeline</h2>
            <p className="marey-subtitle">Spatio-temporal visualization of active corridors</p>
          </div>
          <div className="marey-legend">
            <div className="marey-legend-item">
              <div className="marey-legend-dot solid" />
              <span>ACTIVE</span>
            </div>

            <div className="marey-legend-item">
              <div className="marey-legend-dot maintenance" />
              <span>MAINTENANCE</span>
            </div>
          </div>
        </div>
      )}

      {/* Canvas */}
      <div className="marey-canvas-wrapper">
        {/* Dot-grid background */}
        <div className="marey-grid-bg" />

        <svg
          className="marey-svg"
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="xMidYMid meet"
        >
          {/* SVG Pattern Defs — MMS hatch for maintenance bands */}
          <defs>
            <pattern id="marey-hatch" patternUnits="userSpaceOnUse" width="8" height="8"
                     patternTransform="rotate(45)">
              <rect width="4" height="8" fill="#EAB308" opacity="0.6" />
            </pattern>
          </defs>

          {/* Station horizontal guide lines */}
          {Object.entries(STATION_DISTANCES).map(([name, km]) => {
            const y = getSchematicY(km, H - PAD_TOP - PAD_BTM, PAD_TOP);
            return (
              <g key={name}>
                <line
                  x1={PAD_LEFT}
                  x2={W - PAD_RIGHT}
                  y1={y}
                  y2={y}
                  stroke="#e2e8f0"
                  strokeWidth="1"
                  strokeDasharray="4 4"
                />
                <text
                  x={PAD_LEFT - 8}
                  y={y + 3}
                  textAnchor="end"
                  fontSize="9"
                  fontWeight="600"
                  fill="#94a3b8"
                  fontFamily="Inter, sans-serif"
                >
                  {name}
                </text>
              </g>
            );
          })}

          {/* Active schedule paths */}
          {combinedSchedule.map((entry: ScheduleEntry, i: number) => {
            const color = getTrainColor(entry.train_id);
            return (
              <g key={`${entry.train_id}-${entry.type}-${i}`}>
                <motion.path
                  d={pointsToPath(entry.path)}
                  fill="none"
                  stroke={color}
                  strokeWidth={entry.type === 'actual' ? 3.5 : 2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeDasharray={entry.type === 'projected' ? "6 6" : "none"}
                  opacity={entry.type === 'actual' ? 1 : 0.9}
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: entry.type === 'actual' ? 1 : 0.9 }}
                  transition={{ duration: 1.2, ease: 'easeOut' }}
                />
              </g>
            );
          })}


          <AnimatePresence>
            {maintenanceBands.map(({ blk, x, w }) => (
              <motion.g
                key={`maint-${blk.element_id}`}
                initial={{ opacity: 0, scaleY: 0 }}
                animate={{ opacity: 1, scaleY: 1 }}
                exit={{ opacity: 0, scaleY: 0 }}
                transition={{ duration: 0.35 }}
                style={{ transformOrigin: `${x + w/2}px ${H/2}px` }}
              >
                {/* Hatch-filled vertical band */}
                <rect
                  x={x} y={PAD_TOP}
                  width={w} height={H - PAD_TOP - PAD_BTM}
                  fill="url(#marey-hatch)"
                  rx="4"
                  opacity="0.55"
                />
                {/* Amber border */}
                <rect
                  x={x} y={PAD_TOP}
                  width={w} height={H - PAD_TOP - PAD_BTM}
                  fill="none"
                  stroke={blk.severity === 'TOTAL_BLOCK' ? '#EAB308' : '#F97316'}
                  strokeWidth="1.5"
                  rx="4"
                  opacity="0.8"
                />
                {/* Label */}
                <text
                  x={x + w / 2} y={PAD_TOP + 14}
                  textAnchor="middle" fontSize="7" fontWeight="800"
                  fill={blk.severity === 'TOTAL_BLOCK' ? '#92400e' : '#7c2d12'}
                  fontFamily="Inter, sans-serif"
                >
                  {blk.element_id.length > 10
                    ? blk.element_id.slice(0, 10) + '…'
                    : blk.element_id}
                </text>
                <text
                  x={x + w / 2} y={PAD_TOP + 24}
                  textAnchor="middle" fontSize="6" fontWeight="700"
                  fill="#b45309" fontFamily="Inter, sans-serif"
                >
                  {blk.severity === 'TOTAL_BLOCK' ? 'BLOCKED' : 'RESTRICTED'}
                </text>
              </motion.g>
            ))}
          </AnimatePresence>

          {/* "NOW" time marker */}
          <line x1={nowX} x2={nowX} y1={PAD_TOP - 4} y2={H - PAD_BTM + 4}
            stroke="#94a3b8" strokeWidth="1" strokeDasharray="3 3" />
          <rect x={nowX - 16} y={PAD_TOP - 16} width="32" height="14" rx="7" fill="#475569" />
          <text x={nowX} y={PAD_TOP - 5} textAnchor="middle" fontSize="8"
            fontWeight="700" fill="#fff" fontFamily="Inter, sans-serif">
            NOW
          </text>
        </svg>

        {/* Train ID Legend (Bottom Right) */}
        {globalSchedule.length > 0 && (
          <div className="absolute bottom-3 right-4 bg-white/90 backdrop-blur-md border border-slate-200 shadow-lg rounded-lg p-3 flex flex-col z-10 min-w-[240px]">
            <p className="text-[9px] font-bold text-slate-500 uppercase tracking-widest px-1 mb-2 border-b border-slate-100 pb-1.5">Active Trains</p>
            <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 max-h-[220px] overflow-y-auto pr-2">
              {Array.from(new Set(globalSchedule.map(s => s.train_id))).map(tid => (
                <div key={tid} className="flex items-center gap-2 px-1 hover:bg-slate-50 rounded transition-colors py-0.5 cursor-default min-w-0">
                  <div className="w-2.5 h-2.5 rounded-full shadow-sm shrink-0" style={{ backgroundColor: getTrainColor(tid) }} />
                  <span className="text-[10px] font-mono font-bold text-slate-700 truncate">{tid}</span>
                </div>
              ))}
            </div>
          </div>
        )}


      </div>

      {/* Summary chips */}
      {!hideTelemetry && (
        <div className="marey-chips">
          <div className="marey-chip marey-chip-purple">
            <p className="marey-chip-label">Active Trains</p>
            <p className="marey-chip-value">
              {trainStates.length > 0 ? `${trainStates.length} ON NETWORK` : 'NO TRAINS'}
            </p>
          </div>
          <div className="marey-chip marey-chip-green">
            <p className="marey-chip-label">Fleet Status</p>
            <p className="marey-chip-value">
              {(() => {
                const moving = trainStates.filter(t => t.status === 'Moving').length;
                return `${moving} MOVING · ${trainStates.length - moving} HALTED`;
              })()}
            </p>
          </div>
          <div className="marey-chip marey-chip-slate">
            <p className="marey-chip-label">Risk Factor</p>
            <p className="marey-chip-value">
              {(() => {
                const riskScore = 0.02 + (conflicts.length * 0.25) + (activeBlocks.size * 0.1);
                const label = riskScore >= 0.6 ? 'HIGH' : riskScore >= 0.2 ? 'MEDIUM' : 'LOW';
                return `${label} (${riskScore.toFixed(2)})`;
              })()}
            </p>
          </div>
          <div className="marey-chip marey-chip-slate">
            <p className="marey-chip-label">Energy Mode</p>
            <p className="marey-chip-value">
              {(() => {
                const moving = trainStates.filter(t => t.status === 'Moving').length;
                return moving > 15 ? 'MAX DRAW' : moving > 7 ? 'DYNAMIC' : 'ECONOMY+';
              })()}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
