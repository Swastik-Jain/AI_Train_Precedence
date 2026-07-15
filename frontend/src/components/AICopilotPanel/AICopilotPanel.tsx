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
    } else {
      setApproveState('success');
      setTimeout(() => setApproveState('idle'), 2000);
    }
    // Note: card is no longer removed from the list, it's marked as 'overridden'
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
      <div className="copilot-ttl-wrapper">
        <div className="copilot-ttl-bar">
          <div style={{
            height: '100%',
            background: isExpired ? '#dc2626' : '#8B5CF6',
            width: `${Math.max(0, 100 - (age / TTL) * 100)}%`,
            transition: 'width 0.5s linear'
          }} />
        </div>
        {/* Secondary: server-reported tick countdown for drift visibility */}
        {suggestion.expires_in_ticks !== undefined && !isExpired && (
          <div className={`copilot-ttl-countdown ${suggestion.expires_in_ticks <= 3 ? 'danger' : 'normal'}`}>
            {suggestion.expires_in_ticks}t remaining
          </div>
        )}
      </div>

      {/* Card Header */}
      <div className="copilot-card-header">
        <div className="copilot-badge-container">
          <span className="copilot-priority-badge" style={{ color: pm.color, backgroundColor: pm.bg }}>
            {pm.label}
          </span>
          <span className="copilot-priority-badge" style={{ color: '#fff', backgroundColor: suggestion.override_state === 'overridden' ? '#ea580c' : '#16a34a', padding: '2px 6px', fontSize: '10px' }}>
            {suggestion.override_state === 'overridden' ? 'OVERRIDDEN' : 'EXECUTED'}
          </span>
          {suggestion.urgency === 'CRITICAL' && (
            <span className="copilot-priority-badge" style={{ color: '#fff', backgroundColor: '#dc2626', padding: '2px 6px', fontSize: '10px' }}>
              SAFETY OVERRIDE
            </span>
          )}
        </div>
        <div className="copilot-card-meta">
          <WhyTooltip reasoning={suggestion.reasoning} />
          <span className="copilot-train-id">{suggestion.target_train_id}</span>
        </div>
      </div>

      {/* Action Description & Gauge */}
      <div className="copilot-action-row">
        <p className="copilot-action">{suggestion.decided_action}</p>
        <ConfidenceGauge score={suggestion.confidence_score} />
      </div>

      {/* Inline Modification Panel */}
      <AnimatePresence>
        {isModifying && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="copilot-modification-panel"
          >
            <div className="copilot-modification-inner">
              <div className="copilot-modification-row">
                <label className="copilot-modification-label">Action:</label>
                <select value={modAct} onChange={e => setModAct(Number(e.target.value))} className="copilot-modification-select">
                  <option value={0}>STOP</option>
                  <option value={1}>MAIN</option>
                  <option value={2}>DIVERT</option>
                </select>
              </div>
              <div className="copilot-modification-row">
                <label className="copilot-modification-label">Edge:</label>
                <input type="text" value={modEdge} onChange={e => setModEdge(e.target.value)} className="copilot-modification-input" />
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
          className={`copilot-btn copilot-btn-approve ${approveState === 'verifying' ? 'is-verifying' : ''}`}
          onClick={handleOverride}
          disabled={approveState === 'verifying' || isExpired}
          animate={
            approveState === 'conflict'
              ? { x: [0, -8, 8, -8, 8, 0], backgroundColor: '#dc2626' }
              : approveState === 'verifying'
              ? {}
              : approveState === 'success'
              ? { backgroundColor: '#16a34a' }
              : isExpired
              ? { backgroundColor: '#cbd5e1' }
              : { x: 0, backgroundColor: '#8B5CF6' }
          }
          transition={{ duration: 0.4 }}
          whileTap={{ scale: 0.96 }}
          style={{ backgroundColor: approveState === 'conflict' ? '#dc2626' : approveState === 'success' ? '#16a34a' : isExpired ? '#cbd5e1' : '#8B5CF6' }}
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
          ) : approveState === 'success' ? (
            <>
              <CheckCircle2 size={13} />
              Override Applied
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
        <AnimatePresence>
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
              <p className="copilot-empty-sub">AI Co-Pilot is actively monitoring. No contested decisions or schedule conflicts require intervention at this time.</p>
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
