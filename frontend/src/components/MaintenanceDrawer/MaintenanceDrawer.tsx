import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Wrench, AlertTriangle, Clock, ShieldAlert } from 'lucide-react';
import { useMaintenanceStore } from '../../store/useMaintenanceStore';
import type { BlockSeverity, BlockType } from '../../store/useMaintenanceStore';
import './MaintenanceDrawer.css';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function nowISO(): string {
  return new Date().toISOString().slice(0, 16); // for datetime-local input
}

function twoHoursLaterISO(): string {
  const d = new Date();
  d.setHours(d.getHours() + 2);
  return d.toISOString().slice(0, 16);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export const MaintenanceDrawer: React.FC = () => {
  const {
    activeBlocks,
    isDrawerOpen,
    selectedEdgeForBlock,
    closeDrawer,
    applyBlockRemote,
    applyBlock,
    removeBlockRemote,
  } = useMaintenanceStore();

  const elementId = selectedEdgeForBlock ?? 'SEGMENT_UNKNOWN';
  const existingBlock = activeBlocks.get(elementId);
  const isEdit = !!existingBlock;

  const [type,      setType]      = useState<BlockType>('TRACK_SEGMENT');
  const [severity,  setSeverity]  = useState<BlockSeverity>('TOTAL_BLOCK');
  const [startTime, setStartTime] = useState<string>(nowISO());
  const [endTime,   setEndTime]   = useState<string>(twoHoursLaterISO());
  const [reason,    setReason]    = useState<string>('Scheduled maintenance');
  const [isWhatIf,  setIsWhatIf]  = useState<boolean>(false);
  const [submitting, setSubmitting] = useState(false);

  // Sync form with existing block
  useEffect(() => {
    if (isDrawerOpen) {
      if (existingBlock) {
        setType(existingBlock.type);
        setSeverity(existingBlock.severity);
        setStartTime(existingBlock.start_time ? existingBlock.start_time.slice(0, 16) : nowISO());
        setEndTime(existingBlock.end_time ? existingBlock.end_time.slice(0, 16) : twoHoursLaterISO());
        setReason(existingBlock.reason || '');
        setIsWhatIf(existingBlock.isWhatIf || false);
      } else {
        setStartTime(nowISO());
        setEndTime(twoHoursLaterISO());
        setReason('Scheduled maintenance');
        setIsWhatIf(false);
        setType('TRACK_SEGMENT');
        setSeverity('TOTAL_BLOCK');
      }
    }
  }, [isDrawerOpen, existingBlock]);

  const handleConfirm = useCallback(async () => {
    setSubmitting(true);
    const block = {
      element_id: elementId,
      type,
      severity,
      start_time: new Date(startTime).toISOString(),
      end_time:   new Date(endTime).toISOString(),
      reason,
      isWhatIf,
    };

    if (isWhatIf) {
      // Sandbox-only: store locally, don't touch backend
      applyBlock({ ...block, blockId: crypto.randomUUID() });
    } else {
      await applyBlockRemote(block);
    }

    setSubmitting(false);
    closeDrawer();
  }, [elementId, type, severity, startTime, endTime, reason, isWhatIf,
      applyBlockRemote, applyBlock, closeDrawer]);

  const handleRemove = useCallback(async () => {
    setSubmitting(true);
    if (!isWhatIf) {
      await removeBlockRemote(elementId);
    }
    setSubmitting(false);
    closeDrawer();
  }, [elementId, isWhatIf, removeBlockRemote, closeDrawer]);

  return (
    <AnimatePresence>
      {isDrawerOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            className="mms-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={closeDrawer}
          />

          {/* Drawer Panel */}
          <motion.div
            className="mms-drawer"
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', stiffness: 340, damping: 32 }}
            role="dialog"
            aria-label="Schedule Maintenance"
          >
            {/* Header */}
            <div className="mms-header">
              <div className="mms-header-left">
                <div className="mms-icon-badge">
                  <Wrench size={16} />
                </div>
                <div>
                  <h2 className="mms-title">{isEdit ? 'Edit Maintenance' : 'Schedule Maintenance'}</h2>
                  <p className="mms-element-id">{elementId}</p>
                </div>
              </div>
              <button className="mms-close-btn" onClick={closeDrawer}>
                <X size={16} />
              </button>
            </div>

            {/* What-If toggle */}
            <div className="mms-whatif-row">
              <label className="mms-whatif-label" htmlFor="whatif-toggle">
                What-If Simulation Only
                <span className="mms-whatif-hint">(local, not broadcast to fleet)</span>
              </label>
              <button
                id="whatif-toggle"
                className={`mms-toggle ${isWhatIf ? 'on' : 'off'}`}
                onClick={() => setIsWhatIf(!isWhatIf)}
              >
                <span className="mms-toggle-thumb" />
              </button>
            </div>

            {isWhatIf && (
              <div className="mms-whatif-badge">
                <AlertTriangle size={12} />
                Sandbox mode — changes visible only to you
              </div>
            )}

            {/* Type */}
            <div className="mms-field">
              <label className="mms-label">Block Type</label>
              <div className="mms-radio-group">
                {(['TRACK_SEGMENT', 'PLATFORM'] as BlockType[]).map((t) => (
                  <button
                    key={t}
                    className={`mms-radio-btn ${type === t ? 'active' : ''}`}
                    onClick={() => setType(t)}
                  >
                    {t === 'TRACK_SEGMENT' ? '🛤 Track Segment' : '🚉 Platform'}
                  </button>
                ))}
              </div>
            </div>

            {/* Severity */}
            <div className="mms-field">
              <label className="mms-label">Severity</label>
              <div className="mms-severity-group">
                <button
                  className={`mms-severity-btn total ${severity === 'TOTAL_BLOCK' ? 'active' : ''}`}
                  onClick={() => setSeverity('TOTAL_BLOCK')}
                >
                  <ShieldAlert size={14} />
                  Total Block
                </button>
                <button
                  className={`mms-severity-btn speed ${severity === 'SPEED_RESTRICTION' ? 'active' : ''}`}
                  onClick={() => setSeverity('SPEED_RESTRICTION')}
                >
                  <Clock size={14} />
                  Speed Restriction
                </button>
              </div>
              <p className="mms-severity-hint">
                {severity === 'TOTAL_BLOCK'
                  ? 'Segment fully closed — trains will be rerouted or held.'
                  : 'Segment traversable at reduced speed (≤ 30 km/h).'}
              </p>
            </div>

            {/* Duration */}
            <div className="mms-field">
              <label className="mms-label">Maintenance Window</label>
              <div className="mms-time-row">
                <div className="mms-time-field">
                  <span className="mms-time-sublabel">Start</span>
                  <input
                    type="datetime-local"
                    className="mms-input"
                    value={startTime}
                    onChange={(e) => setStartTime(e.target.value)}
                  />
                </div>
                <div className="mms-time-sep">→</div>
                <div className="mms-time-field">
                  <span className="mms-time-sublabel">End</span>
                  <input
                    type="datetime-local"
                    className="mms-input"
                    value={endTime}
                    onChange={(e) => setEndTime(e.target.value)}
                  />
                </div>
              </div>
            </div>

            {/* Reason */}
            <div className="mms-field">
              <label className="mms-label">Reason / Work Order</label>
              <textarea
                className="mms-textarea"
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                rows={3}
                placeholder="e.g. Track relay replacement — WO-2026-0419"
              />
            </div>

            {/* Actions */}
            <div className="mms-actions" style={{ display: 'flex', gap: '8px' }}>
              <button className="mms-cancel-btn" style={{ flex: 1 }} onClick={closeDrawer} disabled={submitting}>
                Cancel
              </button>
              {isEdit && (
                <button 
                  className="mms-cancel-btn" 
                  style={{ flex: 1, backgroundColor: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', border: '1px solid rgba(239, 68, 68, 0.3)' }} 
                  onClick={handleRemove} 
                  disabled={submitting}
                >
                  {submitting ? '...' : 'Remove'}
                </button>
              )}
              <motion.button
                className={`mms-confirm-btn ${severity === 'TOTAL_BLOCK' ? 'danger' : 'warning'}`}
                style={{ flex: 2 }}
                onClick={handleConfirm}
                disabled={submitting}
                whileTap={{ scale: 0.97 }}
              >
                {submitting ? (
                  <span className="mms-spinner" />
                ) : (
                  <Wrench size={13} />
                )}
                {submitting ? 'Saving…' : (isEdit ? 'Update Block' : 'Confirm Block')}
              </motion.button>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
