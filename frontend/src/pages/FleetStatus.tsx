import React, { useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './FleetStatus.css';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface FleetTrain {
  train_id: string;
  train_type: string;
  max_speed: number;
  priority: number;
  start_time: number;
  deadline: number;
  direction: number;
  edge_id: string;
  position_percentage: number;
  status: 'Moving' | 'Halted' | 'Blocked' | 'Scheduled' | string;
  added_at: string;
}

interface ScheduleNode { arrival: number; departure: number; }
interface ScheduleResult {
  status: 'optimal' | 'infeasible';
  fleet_size: number;
  schedule?: Record<string, Record<string, ScheduleNode>>;
  expert_actions?: Record<string, number[]>;
  timestamp?: string;
  message?: string;
}

const TRAIN_TYPES = [
  'Vande Bharat', 'Rajdhani', 'Superfast', 'Express',
  'Local', 'Suburban', 'Passenger', 'Freight (WAG-9)',
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const STATUS_STYLE: Record<string, string> = {
  Moving   : 'fs-badge fs-badge--moving',
  Halted   : 'fs-badge fs-badge--halted',
  Blocked  : 'fs-badge fs-badge--blocked',
  Scheduled: 'fs-badge fs-badge--scheduled',
};
const PRIORITY_COLOR = (p: number) =>
  p >= 9 ? '#dc2626' : p >= 7 ? '#ea580c' : p >= 5 ? '#8B5CF6' : p >= 3 ? '#0ea5e9' : '#64748b';

const ACTION_LABEL = (a: number) =>
  a === 0 ? 'STOP' : a === 1 ? 'MAIN' : 'DIVERT';
const ACTION_COLOR = (a: number) =>
  a === 0 ? '#ef4444' : a === 1 ? '#22c55e' : '#f59e0b';

const fadeUp = {
  hidden : { opacity: 0, y: 16 },
  visible: (d = 0) => ({ opacity: 1, y: 0, transition: { duration: 0.45, delay: d, ease: [0.22,1,0.36,1] } }),
};

// ---------------------------------------------------------------------------
// Add Train Modal
// ---------------------------------------------------------------------------
interface AddTrainModalProps { onClose: () => void; onAdded: () => void; }
const AddTrainModal: React.FC<AddTrainModalProps> = ({ onClose, onAdded }) => {
  const [form, setForm] = useState({
    train_id  : '',
    train_type: 'Express',
    max_speed : 110,
    start_time: 0,
    deadline  : 120,
    direction : 1,
  });
  const [loading, setLoading] = useState(false);
  const [error,   setError  ] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.train_id.trim()) { setError('Train ID is required'); return; }
    setLoading(true); setError(null);
    try {
      const res = await fetch('/api/v1/fleet', {
        method : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body   : JSON.stringify(form),
      });
      if (!res.ok) {
        const body = await res.json();
        setError(body.detail ?? `Error ${res.status}`);
      } else {
        onAdded();
        onClose();
      }
    } catch { setError('Network error — backend may be offline.'); }
    finally   { setLoading(false); }
  };

  return (
    <motion.div className="fs-modal-overlay" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      onClick={(e) => e.target === e.currentTarget && onClose()}>
      <motion.div className="fs-modal" initial={{ y: 40, opacity: 0 }} animate={{ y: 0, opacity: 1 }}
        exit={{ y: 40, opacity: 0 }} transition={{ type: 'spring', stiffness: 300, damping: 28 }}>

        <div className="fs-modal-header">
          <div>
            <h2 className="fs-modal-title">Add Train to Fleet</h2>
            <p className="fs-modal-subtitle">New train will appear on the live map immediately</p>
          </div>
          <button className="fs-modal-close" onClick={onClose}>✕</button>
        </div>

        <form onSubmit={handleSubmit} className="fs-modal-form">
          <div className="fs-form-row">
            <label>Train ID <span className="fs-required">*</span></label>
            <input className="fs-input" placeholder="e.g. VB-202" value={form.train_id}
              onChange={e => setForm(f => ({ ...f, train_id: e.target.value }))} />
          </div>

          <div className="fs-form-row">
            <label>Train Type</label>
            <select className="fs-input" value={form.train_type}
              onChange={e => setForm(f => ({ ...f, train_type: e.target.value }))}>
              {TRAIN_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>

          <div className="fs-form-grid">
            <div className="fs-form-row">
              <label>Max Speed (km/h)</label>
              <input className="fs-input" type="number" min={30} max={200} value={form.max_speed}
                onChange={e => setForm(f => ({ ...f, max_speed: +e.target.value }))} />
            </div>
            <div className="fs-form-row">
              <label>Direction</label>
              <select className="fs-input" value={form.direction}
                onChange={e => setForm(f => ({ ...f, direction: +e.target.value }))}>
                <option value={1}>Forward (1)</option>
                <option value={2}>Reverse (2)</option>
              </select>
            </div>
          </div>

          <div className="fs-form-grid">
            <div className="fs-form-row">
              <label>Start Time (min)</label>
              <input className="fs-input" type="number" min={0} value={form.start_time}
                onChange={e => setForm(f => ({ ...f, start_time: +e.target.value }))} />
            </div>
            <div className="fs-form-row">
              <label>Deadline (min)</label>
              <input className="fs-input" type="number" min={10} value={form.deadline}
                onChange={e => setForm(f => ({ ...f, deadline: +e.target.value }))} />
            </div>
          </div>

          {error && <p className="fs-form-error">{error}</p>}

          <div className="fs-modal-actions">
            <button type="button" className="fs-btn fs-btn--ghost" onClick={onClose}>Cancel</button>
            <button type="submit" className="fs-btn fs-btn--primary" disabled={loading}>
              {loading ? <span className="fs-spinner" /> : null}
              {loading ? 'Adding…' : 'Add Train'}
            </button>
          </div>
        </form>
      </motion.div>
    </motion.div>
  );
};

// ---------------------------------------------------------------------------
// Schedule Panel
// ---------------------------------------------------------------------------
interface SchedulePanelProps { schedule: ScheduleResult; }
const SchedulePanel: React.FC<SchedulePanelProps> = ({ schedule }) => {
  if (schedule.status === 'infeasible') {
    return (
      <div className="fs-schedule-infeasible">
        <span className="fs-schedule-infeasible-icon">⚠️</span>
        <p>{schedule.message}</p>
      </div>
    );
  }

  const trains = Object.keys(schedule.schedule ?? {});
  return (
    <div className="fs-schedule-grid">
      {trains.map(trainId => {
        const nodes   = Object.entries(schedule.schedule![trainId]);
        const actions = schedule.expert_actions?.[trainId] ?? [];
        return (
          <motion.div key={trainId} className="fs-schedule-card"
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}>
            <div className="fs-schedule-card-header">
              <span className="fs-schedule-train-id">{trainId}</span>
              <span className="fs-schedule-action-count">{actions.length} steps</span>
            </div>
            <div className="fs-schedule-table-wrap">
              <table className="fs-schedule-table">
                <thead>
                  <tr>
                    <th>Node</th>
                    <th>Arr (min)</th>
                    <th>Dep (min)</th>
                    <th>Dwell</th>
                  </tr>
                </thead>
                <tbody>
                  {nodes.map(([node, times]) => (
                    <tr key={node}>
                      <td className="fs-schedule-node">{node}</td>
                      <td>{times.arrival}</td>
                      <td>{times.departure}</td>
                      <td className={times.departure - times.arrival > 0 ? 'fs-dwell-positive' : ''}>
                        {times.departure - times.arrival} min
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {/* Expert action strip */}
            <div className="fs-action-strip">
              {actions.slice(0, 30).map((a, i) => (
                <span key={i} className="fs-action-chip"
                  style={{ background: ACTION_COLOR(a), color: '#fff' }}
                  title={`t=${i}: ${ACTION_LABEL(a)}`}>
                  {ACTION_LABEL(a)[0]}
                </span>
              ))}
              {actions.length > 30 && (
                <span className="fs-action-more">+{actions.length - 30}</span>
              )}
            </div>
          </motion.div>
        );
      })}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------
const FleetStatus: React.FC = () => {
  useEffect(() => { document.title = 'Fleet Status — ORBIT'; }, []);

  const [fleet,       setFleet      ] = useState<FleetTrain[]>([]);
  const [loading,     setLoading    ] = useState(true);
  const [showAddModal, setAddModal  ] = useState(false);
  const [schedule,    setSchedule   ] = useState<ScheduleResult | null>(null);
  const [schedLoading, setSchedLoad ] = useState(false);
  const [schedError,   setSchedError] = useState<string | null>(null);
  const [showSchedule, setShowSched ] = useState(false);

  const fetchFleet = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/fleet');
      if (res.ok) {
        const data = await res.json();
        setFleet(data.fleet ?? []);
      }
    } catch { /* silent */ }
    finally { setLoading(false); }
  }, []);

  // Poll every 3 s for live status updates
  useEffect(() => {
    fetchFleet();
    const id = setInterval(fetchFleet, 3000);
    return () => clearInterval(id);
  }, [fetchFleet]);

  const handleGenerateSchedule = async () => {
    setSchedLoad(true); setSchedError(null); setShowSched(true);
    try {
      const res = await fetch('/api/v1/fleet/generate-schedule', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) { setSchedError(data.detail ?? 'OR-Solver error'); setSchedule(null); }
      else          { setSchedule(data); }
    } catch { setSchedError('Network error — backend offline?'); }
    finally   { setSchedLoad(false); }
  };

  // ── Derived stats ──────────────────────────────────────────────────────────
  const moving  = fleet.filter(t => t.status === 'Moving').length;
  const halted  = fleet.filter(t => t.status === 'Halted' || t.status === 'Blocked').length;
  const avgSpeed = fleet.length
    ? Math.round(fleet.reduce((s, t) => s + t.max_speed * (t.status === 'Moving' ? 1 : 0), 0) / Math.max(moving, 1))
    : 0;

  return (
    <div className="fs-page">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <motion.div className="fs-page-header" variants={fadeUp} initial="hidden" animate="visible">
        <div>
          <h1 className="fs-page-title">Fleet Status</h1>
          <p className="fs-page-sub">Live train registry · OR schedule generation · RL base timetable</p>
        </div>
        <div className="fs-header-actions">
          <button className="fs-btn fs-btn--ghost" onClick={handleGenerateSchedule} disabled={schedLoading}>
            {schedLoading
              ? <><span className="fs-spinner" /> Generating…</>
              : <><span className="material-symbols-outlined" style={{ fontSize: 16 }}>auto_schedule</span> Generate OR Schedule</>
            }
          </button>
          <button className="fs-btn fs-btn--primary" onClick={() => setAddModal(true)}>
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>add</span>
            Add Train
          </button>
        </div>
      </motion.div>

      {/* ── Stat Bar ────────────────────────────────────────────────────────── */}
      <motion.div className="fs-stat-bar" variants={fadeUp} initial="hidden" animate="visible" custom={0.05}>
        <div className="fs-stat">
          <span className="fs-stat-value">{fleet.length}</span>
          <span className="fs-stat-label">Total Fleet</span>
        </div>
        <div className="fs-stat-divider" />
        <div className="fs-stat">
          <span className="fs-stat-value" style={{ color: '#22c55e' }}>{moving}</span>
          <span className="fs-stat-label">Moving</span>
        </div>
        <div className="fs-stat-divider" />
        <div className="fs-stat">
          <span className="fs-stat-value" style={{ color: '#f59e0b' }}>{halted}</span>
          <span className="fs-stat-label">Halted / Blocked</span>
        </div>
        <div className="fs-stat-divider" />
        <div className="fs-stat">
          <span className="fs-stat-value">{avgSpeed} <small>km/h</small></span>
          <span className="fs-stat-label">Avg Active Speed</span>
        </div>
        <div className="fs-stat-divider" />
        <div className="fs-stat">
          <span className={`fs-live-dot ${moving > 0 ? 'active' : ''}`} />
          <span className="fs-stat-label">LIVE</span>
        </div>
      </motion.div>

      {/* ── Fleet Table ─────────────────────────────────────────────────────── */}
      <motion.section className="fs-table-section" variants={fadeUp} initial="hidden" animate="visible" custom={0.1}>
        <div className="fs-table-header">
          <div>
            <h3 className="fs-section-title">Active Fleet Registry</h3>
            <p className="fs-section-sub">Real-time data from simulation · auto-refreshes every 3 s</p>
          </div>
          <span className="fs-count-badge">{fleet.length} trains</span>
        </div>

        {loading ? (
          <div className="fs-loading">
            <div className="fs-loading-spinner" />
            <p>Loading fleet from ORBIT backend…</p>
          </div>
        ) : fleet.length === 0 ? (
          <div className="fs-empty">
            <span className="material-symbols-outlined" style={{ fontSize: 40, opacity: 0.3 }}>train</span>
            <p>No trains in fleet. Click <strong>Add Train</strong> to begin.</p>
          </div>
        ) : (
          <div className="fs-table-wrap">
            <table className="fs-table">
              <thead>
                <tr>
                  <th>Train ID</th>
                  <th>Type</th>
                  <th>Priority</th>
                  <th>Max Speed</th>
                  <th>Status</th>
                  <th>Current Edge</th>
                  <th>Position</th>
                  <th>Start / Deadline</th>
                </tr>
              </thead>
              <tbody>
                <AnimatePresence>
                  {fleet.map((train, i) => (
                    <motion.tr key={train.train_id}
                      initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0 }} transition={{ delay: i * 0.03 }}
                      className="fs-table-row">
                      <td>
                        <div className="fs-train-id-cell">
                          <span className="fs-prio-dot" style={{ background: PRIORITY_COLOR(train.priority) }} />
                          <span className="fs-train-id">{train.train_id}</span>
                        </div>
                      </td>
                      <td><span className="fs-type-tag">{train.train_type}</span></td>
                      <td>
                        <span className="fs-priority-badge"
                          style={{ color: PRIORITY_COLOR(train.priority), borderColor: PRIORITY_COLOR(train.priority) }}>
                          P{train.priority}
                        </span>
                      </td>
                      <td className="fs-speed">{train.max_speed} <small>km/h</small></td>
                      <td>
                        <span className={STATUS_STYLE[train.status] ?? STATUS_STYLE['Scheduled']}>
                          {train.status}
                        </span>
                      </td>
                      <td className="fs-edge-id">{train.edge_id}</td>
                      <td>
                        <div className="fs-progress-wrap">
                          <div className="fs-progress-bar">
                            <div className="fs-progress-fill" style={{ width: `${(train.position_percentage * 100).toFixed(0)}%` }} />
                          </div>
                          <span className="fs-progress-pct">{(train.position_percentage * 100).toFixed(0)}%</span>
                        </div>
                      </td>
                      <td className="fs-times">
                        <span>{train.start_time}′</span>
                        <span className="fs-times-sep">→</span>
                        <span>{train.deadline}′</span>
                      </td>
                    </motion.tr>
                  ))}
                </AnimatePresence>
              </tbody>
            </table>
          </div>
        )}
      </motion.section>

      {/* ── OR Schedule Panel ────────────────────────────────────────────────── */}
      <AnimatePresence>
        {showSchedule && (
          <motion.section
            key="or-schedule-panel"
            className="fs-schedule-section"
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }} transition={{ duration: 0.4 }}>

            <div className="fs-schedule-header">
              <div>
                <div className="fs-schedule-title-row">
                  <span className="fs-or-badge">OR-Tools · CP-SAT</span>
                  <h3 className="fs-section-title">Session Base Schedule</h3>
                </div>
                <p className="fs-section-sub">
                  Optimal timetable computed for current fleet · used as RL base reference
                  {schedule?.timestamp && ` · generated ${new Date(schedule.timestamp).toLocaleTimeString()}`}
                </p>
              </div>
              <button className="fs-btn fs-btn--ghost fs-btn--sm" onClick={() => setShowSched(false)}>
                Hide ✕
              </button>
            </div>

            {schedLoading && (
              <div className="fs-loading">
                <div className="fs-loading-spinner" />
                <p>Running CP-SAT solver… this may take a few seconds</p>
              </div>
            )}

            {schedError && !schedLoading && (
              <div className="fs-schedule-error">⚠️ {schedError}</div>
            )}

            {schedule && !schedLoading && !schedError && (
              <>
                <div className="fs-schedule-meta-row">
                  <span className={`fs-solver-status ${schedule.status}`}>
                    {schedule.status === 'optimal' ? '✓ OPTIMAL' : '✗ INFEASIBLE'}
                  </span>
                  <span className="fs-schedule-meta">{schedule.fleet_size} trains scheduled</span>
                </div>
                <SchedulePanel schedule={schedule} />
              </>
            )}
          </motion.section>
        )}
      </AnimatePresence>

      {/* ── Add Train Modal ─────────────────────────────────────────────────── */}
      <AnimatePresence>
        {showAddModal && (
          <AddTrainModal key="add-train-modal" onClose={() => setAddModal(false)} onAdded={fetchFleet} />
        )}
      </AnimatePresence>
    </div>
  );
};

export default FleetStatus;
