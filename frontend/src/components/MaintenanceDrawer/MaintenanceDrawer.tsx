import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Wrench, AlertTriangle, Clock, ShieldAlert } from 'lucide-react';
import { useMaintenanceStore } from '../../store/useMaintenanceStore';
import { useMapStore } from '../../store/useMapStore';
import type { BlockSeverity, BlockType } from '../../store/useMaintenanceStore';
import './MaintenanceDrawer.css';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function getLocalISOString(date: Date): string {
  const offset = date.getTimezoneOffset() * 60000;
  return new Date(date.getTime() - offset).toISOString().slice(0, 16);
}

function nowISO(): string {
  return getLocalISOString(new Date());
}

function twoHoursLaterISO(): string {
  const d = new Date();
  d.setHours(d.getHours() + 2);
  return getLocalISOString(d);
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

  const { topology } = useMapStore();
  const edges = topology?.edges || [];

  const [localElementId, setLocalElementId] = useState<string>(selectedEdgeForBlock ?? 'SEGMENT_UNKNOWN');
  const existingBlock = activeBlocks.get(localElementId);
  const isEdit = !!existingBlock;

  const [type,      setType]      = useState<BlockType>('TRACK_SEGMENT');
  const [severity,  setSeverity]  = useState<BlockSeverity>('TOTAL_BLOCK');
  const [speedLimit, setSpeedLimit] = useState<number>(30);
  const [startTime, setStartTime] = useState<string>(nowISO());
  const [endTime,   setEndTime]   = useState<string>(twoHoursLaterISO());
  const [reason,    setReason]    = useState<string>('Scheduled maintenance');
  const [isWhatIf,  setIsWhatIf]  = useState<boolean>(false);
  const [submitting, setSubmitting] = useState(false);

  // Sync localElementId when opening
  useEffect(() => {
    if (isDrawerOpen) {
      setLocalElementId(selectedEdgeForBlock ?? 'SEGMENT_UNKNOWN');
    }
  }, [isDrawerOpen, selectedEdgeForBlock]);

  // Sync form with existing block
  useEffect(() => {
    if (isDrawerOpen) {
      if (existingBlock) {
        setType(existingBlock.type);
        setSeverity(existingBlock.severity);
        setSpeedLimit(existingBlock.speed_limit ?? 30);
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
        setSpeedLimit(30);
      }
    }
  }, [isDrawerOpen, localElementId, existingBlock]);

  const handleConfirm = useCallback(async () => {
    setSubmitting(true);
    const block = {
      element_id: localElementId,
      type,
      severity,
      speed_limit: severity === 'SPEED_RESTRICTION' ? speedLimit : undefined,
      start_time: new Date(startTime).toISOString(),
      end_time:   new Date(endTime).toISOString(),
      reason,
      isWhatIf,
    };

    // useMaintenanceStore.ts handles routing to the correct endpoint
    // based on the isWhatIf flag (sandbox vs maintenance endpoints).
    await applyBlockRemote(block);

    setSubmitting(false);
    closeDrawer();
  }, [localElementId, type, severity, speedLimit, startTime, endTime, reason, isWhatIf,
      applyBlockRemote, applyBlock, closeDrawer]);

  const handleRemove = useCallback(async () => {
    setSubmitting(true);
    // removeBlockRemote routes to the correct endpoint based on isWhatIf
    await removeBlockRemote(localElementId);
    setSubmitting(false);
    closeDrawer();
  }, [localElementId, isWhatIf, removeBlockRemote, closeDrawer]);

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
                  {!isEdit ? (
                    <>
                      <input
                        list="edges-list"
                        className="mms-element-id mms-select"
                        value={localElementId === 'SEGMENT_UNKNOWN' ? '' : localElementId}
                        onChange={(e) => setLocalElementId(e.target.value || 'SEGMENT_UNKNOWN')}
                        placeholder="Search Area / Edge ID..."
                        style={{ background: 'transparent', border: '1px solid #cbd5e1', color: '#64748b', fontSize: '11px', padding: '2px 4px', borderRadius: '4px', marginTop: '2px', outline: 'none', width: '100%' }}
                      />
                      <datalist id="edges-list">
                        {edges.map(e => (
                          <option key={e.id} value={e.id}>
                            {e.id} ({e.length}m)
                          </option>
                        ))}
                      </datalist>
                    </>
                  ) : (
                    <p className="mms-element-id">{localElementId}</p>
                  )}
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
                  : `Segment traversable at reduced speed (${speedLimit} km/h).`}
              </p>

              {/* Speed limit input — only shown for SPEED_RESTRICTION */}
              {severity === 'SPEED_RESTRICTION' && (
                <div className="mms-speed-limit-row">
                  <label className="mms-speed-limit-label">
                    Speed Limit (km/h)
                  </label>
                  <div className="mms-speed-limit-controls">
                    <input
                      type="range"
                      min={10}
                      max={110}
                      step={5}
                      value={speedLimit}
                      onChange={(e) => setSpeedLimit(Number(e.target.value))}
                      className="mms-speed-slider"
                    />
                    <input
                      type="number"
                      min={10}
                      max={110}
                      step={5}
                      value={speedLimit}
                      onChange={(e) => setSpeedLimit(Math.min(110, Math.max(10, Number(e.target.value))))}
                      className="mms-speed-number"
                    />
                    <span className="mms-speed-unit">km/h</span>
                  </div>
                  <div className="mms-speed-presets">
                    {[10, 20, 30, 50, 75].map(v => (
                      <button
                        key={v}
                        className={`mms-speed-preset ${speedLimit === v ? 'active' : ''}`}
                        onClick={() => setSpeedLimit(v)}
                      >
                        {v}
                      </button>
                    ))}
                  </div>
                </div>
              )}
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
