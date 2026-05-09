import React, { useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useCopilotStore } from '../../store/useCopilotStore';
import { useMaintenanceStore } from '../../store/useMaintenanceStore';
import type { ScheduleEntry } from '../../store/useCopilotStore';
import './MareyTimeline.css';

// ---------------------------------------------------------------------------
// SVG Canvas constants (viewBox 0 0 800 260)
// ---------------------------------------------------------------------------
const W = 800;
const H = 260;
const PAD_LEFT = 90;
const PAD_RIGHT = 20;
const PAD_TOP = 20;
const PAD_BTM = 28;

const STATIONS = ['TERMINAL A', 'JUNCTION 4', 'HUB NORTH', 'PLATFORM 2'];

/** Build an SVG path `d` string from an array of {x,y} points */
function pointsToPath(pts: { x: number; y: number }[]): string {
  if (pts.length === 0) return '';
  return pts
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`)
    .join(' ');
}

/** Map a ghost path from the suggestion's affected_edges into Marey SVG coords */
function ghostPathFromSuggestion(edges: string[]): { x: number; y: number }[] {
  // Simple deterministic mapping from edge index to Marey xy position
  // In production this would come from the backend schedule delta.
  const bandW = (W - PAD_LEFT - PAD_RIGHT) / Math.max(edges.length, 1);
  return edges.map((_, i) => ({
    x: PAD_LEFT + bandW * i + bandW * 0.5,
    y: PAD_TOP + ((H - PAD_TOP - PAD_BTM) / (edges.length + 1)) * (i + 1) + Math.sin(i * 1.3) * 20,
  }));
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export const MareyTimeline: React.FC = () => {
  const { globalSchedule, previewState, fetchBaseSchedule } = useCopilotStore();
  const { activeBlocks } = useMaintenanceStore();

  React.useEffect(() => {
    fetchBaseSchedule();
  }, [fetchBaseSchedule]);

  // Build ghost path from the hovered suggestion
  const ghostPath = useMemo<{ x: number; y: number }[] | null>(() => {
    if (!previewState) return null;
    return ghostPathFromSuggestion(previewState.affected_edges);
  }, [previewState]);

  const ghostD = ghostPath ? pointsToPath(ghostPath) : '';

  // Station Y positions evenly spaced
  const stationYs = STATIONS.map(
    (_, i) => PAD_TOP + ((H - PAD_TOP - PAD_BTM) / (STATIONS.length - 1)) * i
  );

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
            <div className="marey-legend-dot ghost" />
            <span>AI PROPOSED</span>
          </div>
          <div className="marey-legend-item">
            <div className="marey-legend-dot projected" />
            <span>PROJECTED</span>
          </div>
          <div className="marey-legend-item">
            <div className="marey-legend-dot maintenance" />
            <span>MAINTENANCE</span>
          </div>
        </div>
      </div>

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
          {STATIONS.map((name, i) => (
            <g key={name}>
              <line
                x1={PAD_LEFT}
                x2={W - PAD_RIGHT}
                y1={stationYs[i]}
                y2={stationYs[i]}
                stroke="#e2e8f0"
                strokeWidth="1"
                strokeDasharray="4 4"
              />
              <text
                x={PAD_LEFT - 8}
                y={stationYs[i] + 4}
                textAnchor="end"
                fontSize="9"
                fontWeight="600"
                fill="#94a3b8"
                fontFamily="Inter, sans-serif"
              >
                {name}
              </text>
            </g>
          ))}

          {/* Active schedule paths */}
          {globalSchedule.map((entry: ScheduleEntry, i: number) => (
            <motion.path
              key={`${entry.train_id}-${entry.type}-${i}`}
              d={pointsToPath(entry.path)}
              fill="none"
              stroke={entry.type === 'actual' ? '#8B5CF6' : '#8B5CF6'}
              strokeWidth={entry.type === 'actual' ? 3.5 : 2}
              strokeLinecap="round"
              strokeLinejoin="round"
              opacity={entry.type === 'actual' ? 1 : 0.9}
              initial={{ pathLength: 0, opacity: 0 }}
              animate={{ pathLength: 1, opacity: entry.type === 'actual' ? 1 : 0.9 }}
              transition={{ duration: 1.2, ease: 'easeOut' }}
            />
          ))}

          {/* Ghost / AI Proposed Path */}
          <AnimatePresence>
            {previewState && ghostD && (
              <motion.path
                key={`ghost-${previewState.recommendation_id}`}
                d={ghostD}
                fill="none"
                stroke="#8B5CF6"
                strokeWidth="2.5"
                strokeDasharray="8 5"
                strokeLinecap="round"
                opacity={0.65}
                initial={{ opacity: 0, pathLength: 0 }}
                animate={{ opacity: 0.65, pathLength: 1 }}
                exit={{ opacity: 0, pathLength: 0 }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
              />
            )}
          </AnimatePresence>

          {/* Conflict ripple (static demo point) */}
          <circle className="marey-conflict-dot" cx="342" cy="110" r="4" fill="#8B5CF6" />
          <circle
            cx="342"
            cy="110"
            r="10"
            fill="none"
            stroke="#8B5CF6"
            strokeWidth="2"
            opacity="0.5"
            className="marey-ping"
          />

          {/* MMS Maintenance Hatch Bands */}
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
          <line x1="400" x2="400" y1={PAD_TOP - 4} y2={H - PAD_BTM + 4}
            stroke="#94a3b8" strokeWidth="1" strokeDasharray="3 3" />
          <rect x="384" y={PAD_TOP - 16} width="32" height="14" rx="7" fill="#475569" />
          <text x="400" y={PAD_TOP - 5} textAnchor="middle" fontSize="8"
            fontWeight="700" fill="#fff" fontFamily="Inter, sans-serif">
            NOW
          </text>
        </svg>

        {/* Ghost label overlay */}
        <AnimatePresence>
          {previewState && (
            <motion.div
              className="marey-ghost-label"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 8 }}
              transition={{ duration: 0.25 }}
            >
              <span className="marey-ghost-badge">AI PROPOSED PATH</span>
              <span className="marey-ghost-action">{previewState.proposed_action}</span>
              <span className="marey-ghost-train">{previewState.target_train_id}</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Summary chips */}
      <div className="marey-chips">
        <div className="marey-chip marey-chip-purple">
          <p className="marey-chip-label">Next Departure</p>
          <p className="marey-chip-value">14:20 · PLATFORM 2</p>
        </div>
        <div className="marey-chip marey-chip-green">
          <p className="marey-chip-label">Arrival Sync</p>
          <p className="marey-chip-value">IN 12 MINS</p>
        </div>
        <div className="marey-chip marey-chip-slate">
          <p className="marey-chip-label">Risk Factor</p>
          <p className="marey-chip-value">LOW (0.02)</p>
        </div>
        <div className="marey-chip marey-chip-slate">
          <p className="marey-chip-label">Energy Mode</p>
          <p className="marey-chip-value">ECONOMY+</p>
        </div>
      </div>
    </div>
  );
};
