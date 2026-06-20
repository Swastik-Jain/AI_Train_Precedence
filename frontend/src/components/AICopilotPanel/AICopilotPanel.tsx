import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Info, CheckCircle2, Pencil, X, Zap, AlertTriangle, Wifi, WifiOff } from 'lucide-react';
import { useCopilot } from '../../hooks/useCopilot';
import type { AISuggestion } from '../../store/useCopilotStore';
import './AICopilotPanel.css';

// ---------------------------------------------------------------------------
// Priority meta
// ---------------------------------------------------------------------------
const PRIORITY_META: Record<number, { label: string; color: string; bg: string }> = {
  1: { label: 'P1 CRITICAL', color: '#dc2626', bg: '#fef2f2' },
  2: { label: 'P2 HIGH',     color: '#ea580c', bg: '#fff7ed' },
  3: { label: 'P3 MEDIUM',   color: '#d97706', bg: '#fffbeb' },
  4: { label: 'P4 LOW',      color: '#16a34a', bg: '#f0fdf4' },
  5: { label: 'P5 INFO',     color: '#6366f1', bg: '#eef2ff' },
};

// ---------------------------------------------------------------------------
// TTL Hook
// ---------------------------------------------------------------------------
const useCardAge = (suggestion: AISuggestion) => {
  const [ageSecs, setAgeSecs] = useState(0);
  React.useEffect(() => {
    const start = new Date(suggestion.timestamp).getTime();
    const updateAge = () => setAgeSecs(Math.floor((Date.now() - start) / 1000));
    updateAge();
    const interval = setInterval(updateAge, 500);
    return () => clearInterval(interval);
  }, [suggestion.timestamp]);
  return ageSecs;
};

// ---------------------------------------------------------------------------
// Circular Confidence Gauge
// ---------------------------------------------------------------------------
const ConfidenceGauge: React.FC<{ score: number }> = ({ score }) => {
  const R = 22;
  const circumference = 2 * Math.PI * R;
  const filled = circumference * score;
  const pct = Math.round(score * 100);

  return (
    <div className="copilot-gauge">
      <svg width="60" height="60" viewBox="0 0 60 60">
        {/* Track */}
        <circle cx="30" cy="30" r={R} fill="none" stroke="#e2e8f0" strokeWidth="5" />
        {/* Fill */}
        <circle
          cx="30"
          cy="30"
          r={R}
          fill="none"
          stroke="#8B5CF6"
          strokeWidth="5"
          strokeDasharray={`${filled} ${circumference - filled}`}
          strokeDashoffset={circumference * 0.25} /* start from top */
          strokeLinecap="round"
          style={{ transition: 'stroke-dasharray 0.6s ease' }}
        />
        {/* Label */}
        <text x="30" y="34" textAnchor="middle" fontSize="11" fontWeight="700" fill="#1e293b">
          {pct}%
        </text>
      </svg>
      <span className="copilot-gauge-label">CONF</span>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Why Tooltip
// ---------------------------------------------------------------------------
const WhyTooltip: React.FC<{ reasoning: string }> = ({ reasoning }) => {
  const [show, setShow] = useState(false);

  return (
    <div className="copilot-why-wrapper">
      <button
        className="copilot-why-btn"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        aria-label="Why this suggestion?"
      >
        <Info size={14} />
      </button>
      <AnimatePresence>
        {show && (
          <motion.div
            className="copilot-why-tooltip"
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            transition={{ duration: 0.15 }}
          >
            <p className="copilot-why-title">RL Agent Reasoning</p>
            <p className="copilot-why-text">{reasoning}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Decision Card
// ---------------------------------------------------------------------------
const DecisionCard: React.FC<{
  suggestion: AISuggestion;
  onOverride: (id: string, modAct?: number, modEdge?: string) => Promise<void>;
  onDismiss: (id: string) => void;
  onHoverStart: (s: AISuggestion) => void;
  onHoverEnd: () => void;
}> = ({ suggestion, onOverride, onDismiss, onHoverStart, onHoverEnd }) => {
  const [approveState, setApproveState] = useState<'idle' | 'verifying' | 'success' | 'conflict'>('idle');
  const [isModifying, setIsModifying] = useState(false);
  const [modAct, setModAct] = useState<number>((suggestion as any).rl_action ?? 1);
  const [modEdge, setModEdge] = useState<string>(suggestion.affected_edges?.[0] || '');

  const age = useCardAge(suggestion);
  const TTL = 20; // Increased to 20s to match new tick interval
  const isExpired = age >= TTL || suggestion.status === 'expired';

  const pm = PRIORITY_META[suggestion.priority_level] ?? PRIORITY_META[3];
  const impactText =
    suggestion.impact_analysis >= 0
      ? `+${suggestion.impact_analysis}min saved`
      : `${suggestion.impact_analysis}min delay`;
  const impactColor = suggestion.impact_analysis >= 0 ? '#16a34a' : '#dc2626';

  const handleOverride = useCallback(async () => {
    if (approveState !== 'idle' || isExpired) return;
    setApproveState('verifying');
    const result = await onOverride(suggestion.recommendation_id, isModifying ? modAct : undefined, isModifying ? modEdge : undefined);
    // @ts-ignore — result type from parent
    if (result?.ok === false && result?.safetyConflict) {
      setApproveState('conflict');
      setTimeout(() => setApproveState('idle'), 3000);
    } else if (result?.ok === false) {
      setApproveState('idle');
    }
    // On success the card is removed from the list by the store
  }, [approveState, isExpired, onOverride, suggestion.recommendation_id, isModifying, modAct, modEdge]);

  return (
    <motion.div
      layout
      className="copilot-card"
      onHoverStart={() => onHoverStart(suggestion)}
      onHoverEnd={onHoverEnd}
      initial={{ opacity: 0, x: 40 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -60, scale: 0.95 }}
      transition={{ type: 'spring', stiffness: 300, damping: 26 }}
    >
      {/* TTL Bar + tick countdown */}
      <div style={{ marginBottom: '6px' }}>
        <div style={{ height: '3px', background: '#e2e8f0', width: '100%', borderRadius: '2px', overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            background: isExpired ? '#dc2626' : '#8B5CF6',
            width: `${Math.max(0, 100 - (age / TTL) * 100)}%`,
            transition: 'width 0.5s linear'
          }} />
        </div>
        {/* Secondary: server-reported tick countdown for drift visibility */}
        {suggestion.expires_in_ticks !== undefined && !isExpired && (
          <div style={{
            fontSize: '9px',
            color: suggestion.expires_in_ticks <= 3 ? '#dc2626' : '#94a3b8',
            textAlign: 'right',
            marginTop: '2px',
            fontVariantNumeric: 'tabular-nums',
          }}>
            {suggestion.expires_in_ticks}t remaining
          </div>
        )}
      </div>

      {/* Card Header */}
      <div className="copilot-card-header">
        <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
          <span className="copilot-priority-badge" style={{ color: pm.color, backgroundColor: pm.bg }}>
            {pm.label}
          </span>
          <span className="copilot-priority-badge" style={{ color: '#fff', backgroundColor: suggestion.override_state === 'overridden' ? '#ea580c' : '#16a34a', padding: '2px 6px', fontSize: '10px' }}>
            {suggestion.override_state === 'overridden' ? 'OVERRIDDEN' : 'EXECUTED'}
          </span>
          {suggestion.urgency === 'CRITICAL' && (
            <span className="copilot-priority-badge" style={{ color: '#fff', backgroundColor: '#dc2626', padding: '2px 6px', fontSize: '10px' }}>
              AUTO-ACT
            </span>
          )}
        </div>
        <div className="copilot-card-meta">
          <WhyTooltip reasoning={suggestion.reasoning} />
          <span className="copilot-train-id">{suggestion.target_train_id}</span>
        </div>
      </div>

      {/* Action Description */}
      <p className="copilot-action">{suggestion.decided_action}</p>

      {/* Gauge + Impact Row */}
      <div className="copilot-data-row">
        <ConfidenceGauge score={suggestion.confidence_score} />
        <div className="copilot-impact-block">
          <span className="copilot-impact-label">Impact</span>
          <span className="copilot-impact-value" style={{ color: impactColor }}>
            {impactText}
          </span>
          <span className="copilot-ts">
            {new Date(suggestion.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </span>
        </div>
      </div>

      {/* Inline Modification Panel */}
      <AnimatePresence>
        {isModifying && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            style={{ overflow: 'hidden', marginBottom: '8px' }}
          >
            <div style={{ padding: '8px', background: '#f8fafc', borderRadius: '6px', fontSize: '12px', border: '1px solid #e2e8f0', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label style={{ fontWeight: 600, color: '#475569' }}>Action:</label>
                <select value={modAct} onChange={e => setModAct(Number(e.target.value))} style={{ padding: '2px 6px', borderRadius: '4px', border: '1px solid #cbd5e1' }}>
                  <option value={0}>STOP</option>
                  <option value={1}>MAIN</option>
                  <option value={2}>DIVERT</option>
                </select>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label style={{ fontWeight: 600, color: '#475569' }}>Edge:</label>
                <input type="text" value={modEdge} onChange={e => setModEdge(e.target.value)} style={{ padding: '2px 6px', width: '100px', borderRadius: '4px', border: '1px solid #cbd5e1' }} />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Action Buttons */}
      <div className="copilot-actions">
        {/* Override */}
        <motion.button
          id={`override-${suggestion.recommendation_id}`}
          className="copilot-btn copilot-btn-approve"
          onClick={handleOverride}
          disabled={approveState === 'verifying' || isExpired}
          animate={
            approveState === 'conflict'
              ? { x: [0, -8, 8, -8, 8, 0], backgroundColor: '#dc2626' }
              : approveState === 'verifying'
              ? {}
              : isExpired
              ? { backgroundColor: '#cbd5e1' }
              : { x: 0, backgroundColor: '#8B5CF6' }
          }
          transition={{ duration: 0.4 }}
          whileTap={{ scale: 0.96 }}
          style={{ backgroundColor: approveState === 'conflict' ? '#dc2626' : isExpired ? '#cbd5e1' : '#8B5CF6' }}
        >
          {approveState === 'verifying' ? (
            <span className="copilot-verifying">
              <span className="copilot-spinner" />
              Applying Override…
            </span>
          ) : approveState === 'conflict' ? (
            <>
              <AlertTriangle size={13} />
              Safety Conflict!
            </>
          ) : isExpired ? (
            <>
              Expired
            </>
          ) : (
            <>
              <CheckCircle2 size={13} />
              Apply Override
            </>
          )}
        </motion.button>

        {/* Modify */}
        <motion.button
          className="copilot-btn copilot-btn-modify"
          onClick={() => setIsModifying(!isModifying)}
          whileTap={{ scale: 0.96 }}
        >
          <Pencil size={12} />
          Modify
        </motion.button>

        {/* Dismiss */}
        <motion.button
          className="copilot-btn copilot-btn-dismiss"
          onClick={() => onDismiss(suggestion.recommendation_id)}
          whileTap={{ scale: 0.96 }}
        >
          <X size={12} />
        </motion.button>
      </div>
    </motion.div>
  );
};

// ---------------------------------------------------------------------------
// AI Copilot Panel
// ---------------------------------------------------------------------------
export const AICopilotPanel: React.FC = () => {
  const {
    activeSuggestions,
    isConnected,
    overrideAction,
    rejectAction,
    previewAction,
    clearPreview,
  } = useCopilot();

  const handleOverride = useCallback(
    async (id: string, modAct?: number, modEdge?: string) => {
      const result = await overrideAction(id, modAct, modEdge);
      return result;
    },
    [overrideAction]
  );

  return (
    <div className="copilot-panel">
      {/* Panel Header */}
      <div className="copilot-panel-header">
        <div className="copilot-panel-title-row">
          <div className="copilot-orbit-badge">
            <Zap size={14} />
            <span>ORBIT</span>
          </div>
          <h2 className="copilot-panel-title">AI Co-Pilot</h2>
        </div>
        <div className="copilot-panel-subtitle-row">
          <p className="copilot-panel-subtitle">RL Dispatch Engine · Real-time</p>
          <span
            className={`copilot-ws-dot ${isConnected ? 'connected' : 'disconnected'}`}
            title={isConnected ? 'WebSocket connected' : 'Reconnecting…'}
          >
            {isConnected ? <Wifi size={11} /> : <WifiOff size={11} />}
            {isConnected ? 'Live' : 'Offline'}
          </span>
        </div>
      </div>

      {/* Cards */}
      <div className="copilot-card-list">
        <AnimatePresence mode="popLayout">
          {activeSuggestions.length === 0 ? (
            <motion.div
              key="empty"
              className="copilot-empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <div className="copilot-empty-icon">
                <Zap size={20} />
              </div>
              <p className="copilot-empty-title">Monitoring Network</p>
              <p className="copilot-empty-sub">Contested AI decisions will appear here as the RL agent re-optimises the network.</p>
            </motion.div>
          ) : (
            activeSuggestions.map((s) => (
              <DecisionCard
                key={s.recommendation_id}
                suggestion={s}
                onOverride={handleOverride}
                onDismiss={(id) => rejectAction(id)}
                onHoverStart={previewAction}
                onHoverEnd={clearPreview}
              />
            ))
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
