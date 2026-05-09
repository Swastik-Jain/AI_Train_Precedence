import React, { useEffect, useState, useMemo } from 'react';
import { useMapStore } from '../../store/useMapStore';
import { useCopilotStore } from '../../store/useCopilotStore';
import { useMaintenanceStore } from '../../store/useMaintenanceStore';
import type { Node, Edge, TrainState } from '../../store/useMapStore';
import './KineticMap.css';

export const KineticMap: React.FC = () => {
  const { 
    topology, 
    trainStates, 
    conflicts, 
    connectWebSocket, 
    isConnected,
    setSelectedTrain,
    selectedTrainId,
    committedTrainId,
    committedAction,
  } = useMapStore();

  // AI Co-pilot ghost projection — which edges to pulse
  const previewState = useCopilotStore((s) => s.previewState);
  const aiAffectedEdges = useMemo(
    () => new Set(previewState?.affected_edges ?? []),
    [previewState]
  );

  // Committed edges — green flash after controller approval
  const committedEdges = useMapStore((s) => s.committedEdges);


  // MMS — maintenance block overlay
  const { activeBlocks, setSelectedEdgeForBlock, openDrawer } = useMaintenanceStore();

  const [hoveredTrain, setHoveredTrain] = useState<string | null>(null);
  const [selectedTrack, setSelectedTrack] = useState<Edge | null>(null);

  useEffect(() => {
    connectWebSocket();
  }, [connectWebSocket]);

  // ViewBox bounds calculation
  const viewBox = useMemo(() => {
    if (!topology || topology.nodes.length === 0) return '0 0 1000 500';
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    topology.nodes.forEach(n => {
      if (n.x < minX) minX = n.x;
      if (n.y < minY) minY = n.y;
      if (n.x > maxX) maxX = n.x;
      if (n.y > maxY) maxY = n.y;
    });
    // Add padding
    return `${minX - 100} ${minY - 100} ${maxX - minX + 200} ${maxY - minY + 200}`;
  }, [topology]);

  // Derived edge status mapping
  const activeEdges = useMemo(() => {
    const map = new Map<string, 'reserved' | 'conflict' | 'free'>();
    if (!topology) return map;
    
    // Default all to free
    topology.edges.forEach(e => map.set(e.id, 'free'));
    
    // Mark reserved
    trainStates.forEach(t => {
      map.set(t.edge_id, 'reserved');
    });

    // Mark conflicts
    conflicts.forEach(cid => {
      map.set(cid, 'conflict');
    });

    return map;
  }, [topology, trainStates, conflicts]);

  const getNode = (id: string) => topology?.nodes.find(n => n.id === id);

  const getInterpolatedPosition = (train: TrainState) => {
    if (!topology) return { x: 0, y: 0 };
    const edge = topology.edges.find(e => e.id === train.edge_id);
    if (!edge) return { x: 0, y: 0 };
    
    const sourceNode = getNode(edge.source);
    const targetNode = getNode(edge.target);
    
    if (!sourceNode || !targetNode) return { x: 0, y: 0 };
    
    const dx = targetNode.x - sourceNode.x;
    const dy = targetNode.y - sourceNode.y;
    
    return {
      x: sourceNode.x + dx * train.position_percentage,
      y: sourceNode.y + dy * train.position_percentage
    };
  };

  if (!topology) {
    return (
      <div className="kinetic-map-container flex items-center justify-center">
        <div className="text-slate-400">Loading Configuration Topology...</div>
      </div>
    );
  }

  return (
    <div className="kinetic-map-container">
      {!isConnected && (
        <div className="absolute top-2 left-2 px-2 py-1 bg-red-900/80 text-white text-xs rounded border border-red-500 z-10 hidden">
          Live Feed Disconnected
        </div>
      )}

      {selectedTrack && (
        <div className="track-health-modal">
          <div className="flex justify-between mb-2">
            <h3 className="modal-title">Track Monitor</h3>
            <button className="text-slate-400 hover:text-white" onClick={() => setSelectedTrack(null)}>✕</button>
          </div>
          <div className="modal-row"><span>Segment ID:</span> <strong>{selectedTrack.id}</strong></div>
          <div className="modal-row"><span>Max Speed:</span> <strong>{selectedTrack.max_speed} km/h</strong></div>
          <div className="modal-row"><span>Status:</span> 
            <strong className={activeEdges.get(selectedTrack.id) === 'conflict' ? 'text-red-600' : activeEdges.get(selectedTrack.id) === 'reserved' ? 'text-[#8B5CF6]' : 'text-slate-700'}>
              {activeEdges.get(selectedTrack.id)?.toUpperCase()}
            </strong>
          </div>
        </div>
      )}

      <svg className="kinetic-map-svg" viewBox={viewBox} preserveAspectRatio="xMidYMid meet"
           onContextMenu={(e) => e.preventDefault()}>

        {/* SVG Pattern Defs — MMS caution overlays */}
        <defs>
          {/* TOTAL_BLOCK: diagonal amber/black caution stripe at 45° */}
          <pattern id="caution-stripe" patternUnits="userSpaceOnUse" width="10" height="10"
                   patternTransform="rotate(45)">
            <rect width="5" height="10" fill="#EAB308" opacity="0.85" />
            <rect x="5" width="5" height="10" fill="#1e293b" opacity="0.75" />
          </pattern>

          {/* SPEED_RESTRICTION: amber dashed hatch */}
          <pattern id="speed-restriction" patternUnits="userSpaceOnUse" width="8" height="8"
                   patternTransform="rotate(45)">
            <rect width="4" height="8" fill="#F97316" opacity="0.55" />
          </pattern>
        </defs>
        {/* Draw Projected Paths for Hovered Train */}
        {hoveredTrain && (() => {
          const train = trainStates.find(t => t.train_id === hoveredTrain);
          if (!train || !train.path) return null;
          
          return train.path.map((edgeId, idx) => {
            const e = topology.edges.find(x => x.id === edgeId);
            if (!e) return null;
            const src = getNode(e.source);
            const tgt = getNode(e.target);
            if (!src || !tgt) return null;
            return (
              <line
                key={`proj-${idx}`}
                x1={src.x} y1={src.y}
                x2={tgt.x} y2={tgt.y}
                className="projected-path"
              />
            );
          });
        })()}

        {/* Draw Edges */}
        {topology.edges.map(edge => {
          const source = getNode(edge.source);
          const target = getNode(edge.target);
          if (!source || !target) return null;

          const status      = activeEdges.get(edge.id);
          const isAiAffected = aiAffectedEdges.has(edge.id);
          const isCommitted  = committedEdges.has(edge.id);
          const block       = activeBlocks.get(edge.id);

          const isBlocked = !!block;
          const isTotal   = block?.severity === 'TOTAL_BLOCK';

          // Midpoint for warning icon
          const mx = (source.x + target.x) / 2;
          const my = (source.y + target.y) / 2;

          return (
            <g
              key={edge.id}
              onContextMenu={(e) => {
                e.preventDefault();
                setSelectedEdgeForBlock(edge.id);
                openDrawer(edge.id);
              }}
            >
              {/* AI hover ghost pulse overlay */}
              {isAiAffected && (
                <line
                  x1={source.x} y1={source.y}
                  x2={target.x} y2={target.y}
                  className="map-edge ai-pulse"
                  strokeWidth={6}
                />
              )}

              {/* Committed-decision flash — green pulse for 4 s after approval */}
              {isCommitted && (
                <line
                  x1={source.x} y1={source.y}
                  x2={target.x} y2={target.y}
                  className="map-edge committed-flash"
                  strokeWidth={7}
                  style={{ pointerEvents: 'none' }}
                />
              )}

              {/* MMS Caution block overlay — fat striped rect behind edge */}
              {isBlocked && (() => {
                const dx = target.x - source.x;
                const dy = target.y - source.y;
                const len = Math.sqrt(dx*dx + dy*dy) || 1;
                const ux = dx/len, uy = dy/len;
                const nx = -uy * 8, ny = ux * 8;
                const pts = [
                  `${source.x + nx},${source.y + ny}`,
                  `${target.x + nx},${target.y + ny}`,
                  `${target.x - nx},${target.y - ny}`,
                  `${source.x - nx},${source.y - ny}`,
                ].join(' ');
                return (
                  <polygon
                    points={pts}
                    fill={isTotal ? 'url(#caution-stripe)' : 'url(#speed-restriction)'}
                    className="mms-edge-overlay"
                    onClick={() => setSelectedTrack(edge)}
                  />
                );
              })()}

              {/* Main edge line */}
              <line
                x1={source.x} y1={source.y}
                x2={target.x} y2={target.y}
                className={`map-edge ${status}${isBlocked ? isTotal ? ' map-edge--blocked' : ' map-edge--restricted' : ''}`}
                onClick={() => setSelectedTrack(edge)}
              />

              {/* Conflict label */}
              {status === 'conflict' && !isBlocked && (
                <text x={mx} y={my - 10} fill="#ff3d00" fontSize="12" fontWeight="bold" textAnchor="middle">
                  CONFLICT EVT
                </text>
              )}

              {/* MMS Warning icon at midpoint */}
              {isBlocked && (
                <g className="mms-warning-icon" onClick={() => { setSelectedEdgeForBlock(edge.id); openDrawer(edge.id); }}>
                  <circle cx={mx} cy={my} r={9}
                    fill={isTotal ? '#EAB308' : '#F97316'} stroke="#fff" strokeWidth="2" />
                  <text x={mx} y={my + 4} textAnchor="middle" fontSize="10" fontWeight="900"
                    fill="#fff" style={{ pointerEvents: 'none' }}>
                    !
                  </text>
                </g>
              )}
            </g>
          );
        })}

        {/* Draw Nodes */}
        {topology.nodes.map(node => {
          const isLargeHub = node.type === 'YARD' || node.type === 'DESTINATION';
          const isPlatform = node.type === 'PLATFORM';
          const isPSR = node.type === 'PSR_CURVE';
          
          let label = "";
          if (node.id === '0') label = "ORIGIN YARD";
          else if (node.id === '999') label = "DESTINATION STRAND";
          else if (isPlatform) label = `PLATFORM ${node.id.slice(-2)}`;
          else if (isPSR) label = "⚠️ PSR ZONE";
          else if (node.type === 'LOOP') label = "OVERTAKE LOOP";
          else if (node.type === 'SWITCH') label = "JUNCTION";
          else if (node.type === 'SINGLE_LINE_BLOCK' && node.id === '18') label = "BRIDGE BOTTLENECK";

          return (
            <g key={node.id}>
              {/* Optional PSR Aura */}
              {isPSR && <circle cx={node.x} cy={node.y} r={14} className="node-psr-pulse" />}
              
              {/* Geometry */}
              {isLargeHub ? (
                <rect 
                  x={node.x - 20} y={node.y - 20} 
                  width={40} height={40} rx={8} 
                  className={`map-node hub ${node.type.toLowerCase()}`} 
                />
              ) : (
                <circle 
                  cx={node.x} cy={node.y}
                  r={isPlatform ? 14 : (isPSR ? 12 : 8)}
                  className={`map-node ${node.type.toLowerCase()}`}
                />
              )}

              {/* Text Label */}
              {label && (
                <text 
                  x={node.x} 
                  y={node.y - (isLargeHub ? 32 : 28)} 
                  className="map-node-label"
                >
                  {label}
                </text>
              )}
            </g>
          );
        })}

        {/* Draw Trains */}
        {trainStates.map(train => {
          const pos = getInterpolatedPosition(train);
          const isSelected   = selectedTrainId  === train.train_id;
          const isCommitted  = committedTrainId === train.train_id;

          // Action label for the committed train badge
          const actionLabel =
            committedAction === 0 ? '🛑 STOP'
            : committedAction === 2 ? '🔀 DIVERT'
            : '✅ COMMIT';

          return (
            <g 
              key={train.train_id}
              className={`train-node ${train.status.toLowerCase()}`}
              onMouseEnter={() => setHoveredTrain(train.train_id)}
              onMouseLeave={() => setHoveredTrain(null)}
              onClick={() => setSelectedTrain(train.train_id)}
            >
              {/* Conflict Aura */}
              {train.status === 'Conflict' && (
                <circle cx={pos.x} cy={pos.y} r={10} className="conflict-aura" />
              )}

              {/* ── Committed-action highlight ring ── */}
              {isCommitted && (
                <>
                  {/* Outer pulsing ring */}
                  <circle
                    cx={pos.x} cy={pos.y}
                    r={14}
                    className="committed-train-ring"
                    style={{ pointerEvents: 'none' }}
                  />
                  {/* Inner solid accent ring */}
                  <circle
                    cx={pos.x} cy={pos.y}
                    r={9}
                    fill="none"
                    stroke="#22c55e"
                    strokeWidth={2}
                    opacity={0.9}
                    style={{ pointerEvents: 'none' }}
                  />
                  {/* Action badge above the train label */}
                  <rect
                    x={pos.x - 28} y={pos.y - 34}
                    width={56} height={14}
                    rx={4}
                    fill="#22c55e"
                    opacity={0.92}
                    style={{ pointerEvents: 'none' }}
                  />
                  <text
                    x={pos.x} y={pos.y - 23}
                    className="committed-action-label"
                    textAnchor="middle"
                    style={{ pointerEvents: 'none' }}
                  >
                    {actionLabel}
                  </text>
                </>
              )}

              <circle 
                cx={pos.x} cy={pos.y} 
                r={isSelected ? 6 : 4} 
                className="train-core"
                stroke={isSelected ? '#fff' : isCommitted ? '#22c55e' : 'none'}
                strokeWidth="2"
              />
              
              <text 
                x={pos.x} 
                y={pos.y - 12} 
                className="train-label" 
                textAnchor="middle"
                opacity={hoveredTrain === train.train_id || isSelected || isCommitted ? 1 : 0.6}
              >
                {train.train_id}
              </text>
            </g>
          );
        })}

      </svg>

      {/* ── Committed-action corner banner ───────────────────────────────────
          Appears for 6 seconds after any commit; shows the target train + action.
      ────────────────────────────────────────────────────────────── */}
      {committedTrainId && (
        <div className="committed-train-banner">
          <span className="committed-train-banner__icon">
            {committedAction === 0 ? '🛑' : committedAction === 2 ? '🔀' : '✅'}
          </span>
          <div className="committed-train-banner__body">
            <div className="committed-train-banner__action">
              {committedAction === 0 ? 'STOP Applied' : committedAction === 2 ? 'DIVERT Applied' : 'Action Applied'}
            </div>
            <div className="committed-train-banner__train">{committedTrainId}</div>
          </div>
        </div>
      )}
    </div>
  );
};
