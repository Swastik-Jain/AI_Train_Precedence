import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { KineticMap } from '../components/KineticMap/KineticMap';
import { MareyTimeline } from '../components/MareyTimeline/MareyTimeline';
import { useMaintenanceStore } from '../store/useMaintenanceStore';
import { useMapStore } from '../store/useMapStore';
import './ControlCentre.css';

type ScenarioResult = {
  id: string;
  label: string;
  delayTrainId: string;
  latency: number;
  forcedActions: Record<string, number>;
  impact: { reliability: string; congestion: string };
  adjustments: { id: number; type: string; desc: string; train_id?: string; edge_id?: string; constraint_type?: string; value?: number }[];
};

/* ────────────────────────────────────────────────────────────────
   FRAMER-MOTION VARIANTS
───────────────────────────────────────────────────────────────── */
const fadeUp = {
  hidden:  { opacity: 0, y: 20 },
  visible: (d: number = 0) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1], delay: d },
  }),
};

const stagger = {
  hidden:  {},
  visible: { transition: { staggerChildren: 0.07 } },
};

const ControlCentre: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'simulate' | 'maintenance' | 'audit'>('simulate');
  
  // Stores
  const { activeBlocks, fetchActiveBlocks, removeBlockRemote, openDrawer, impactReport } = useMaintenanceStore();
  const { trainStates } = useMapStore();
  const blockList = Array.from(activeBlocks.values());

  // Simulate Tab State
  const [delayTrainId, setDelayTrainId] = useState(trainStates[0]?.train_id || '');
  const [latency, setLatency] = useState(15);
  const [isRecalculating, setIsRecalculating] = useState(false);
  const [recalculatedForTrain, setRecalculatedForTrain] = useState<string | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioResult[]>([]);
  const [scenarioLabel, setScenarioLabel] = useState('Scenario A');
  const [forcedActions, setForcedActions] = useState<Record<string, number>>({});

  // Audit Log State
  const [systemLogs, setSystemLogs] = useState<any[]>([]);
  const [logFilter, setLogFilter] = useState('All');
  const [logLimit, setLogLimit] = useState(50);
  
  // Calendar State
  const [calendarDate, setCalendarDate] = useState(new Date());

  useEffect(() => {
    document.title = 'Control Centre - Zentra Ops | ORBIT';
    fetchActiveBlocks();
  }, [fetchActiveBlocks]);

  // Load logs
  useEffect(() => {
    if (activeTab !== 'audit') return;
    const loadLogs = async () => {
      try {
        const res = await fetch(`/api/v1/system/audit-logs?limit=${logLimit}`);
        if(res.ok) {
          const data = await res.json();
          const logs = (data.logs || []).map((l: any) => ({
            ...l,
            t: new Date(l.t).toLocaleString()
          }));
          setSystemLogs(logs);
        }
      } catch (err) {
        console.error(err);
      }
    };
    loadLogs();
    const interval = setInterval(loadLogs, 3000);
    return () => clearInterval(interval);
  }, [logLimit, activeTab]);

  const handleRecalculate = async () => {
    setIsRecalculating(true);
    try {
        const res = await fetch('http://localhost:8000/api/v1/simulation/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                label: scenarioLabel,
                delay_train_id: delayTrainId,
                latency_minutes: latency,
                forced_actions: forcedActions,
            })
        });
        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();
        setScenarios(prev => [
            ...prev,
            {
                id: `${Date.now()}`,
                label: data.label || scenarioLabel,
                delayTrainId,
                latency,
                forcedActions,
                impact: { reliability: data.impact.reliability, congestion: data.impact.congestion },
                adjustments: data.adjustments || [],
            },
        ]);
        setRecalculatedForTrain(delayTrainId);
        setScenarioLabel(`Scenario ${String.fromCharCode(65 + scenarios.length + 1)}`); // auto-increment A, B, C...
    } catch (err) {
        console.error('[Sandbox] Analysis failed:', err);
    } finally {
        setIsRecalculating(false);
    }
  };

  const handleDeploy = async () => {
    try {
        const payload = {
            blocks: blockList,
            constraints: (scenarios[scenarios.length - 1]?.adjustments || []).map(a => ({
                id: `constraint-${Date.now()}-${a.id}`,
                type: a.constraint_type || 'SPEED_LIMIT',
                edge_id: a.edge_id || 'edge-1-2', 
                value: a.value !== undefined ? a.value : 0,
                expires_at: Math.floor(Date.now() / 1000) + 3600 // 1 hour TTL
            }))
        };
        const res = await fetch('http://localhost:8000/api/v1/simulation/deploy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (res.ok) {
            alert('Simulation Sandbox Deployed Successfully!');
        } else {
            alert('Failed to deploy simulation.');
        }
    } catch (error) {
        console.error(error);
        alert('Error connecting to backend.');
    }
  };

  const handleExportLog = () => {
    if (systemLogs.length === 0) {
      alert("No logs available to export.");
      return;
    }
    let report = `====================================================\n ORBIT COMMAND CENTER - SYSTEM AUDIT LOG\n Generated: ${new Date().toLocaleString()}\n Total Entries: ${systemLogs.length}\n====================================================\n\n`;
    systemLogs.forEach((log) => {
      report += `[${log.t}]\nSOURCE:   ${log.source}\nACTION:   ${log.action}\nOPERATOR: ${log.operator}\nSTATUS:   ${log.status.toUpperCase()}\n----------------------------------------------------\n`;
    });
    const blob = new Blob([report], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `orbit_audit_log_${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Calendar Logic
  const now = new Date();
  const year = calendarDate.getFullYear();
  const month = calendarDate.getMonth();
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const firstDay = new Date(year, month, 1).getDay();
  const offset = firstDay === 0 ? 6 : firstDay - 1; 
  const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  const currentMonthName = `${monthNames[month]} ${year}`;
  const daysArray = Array.from({ length: 42 }, (_, i) => {
    const dayNum = i - offset + 1;
    if (dayNum > 0 && dayNum <= daysInMonth) return dayNum;
    return null;
  });

  const blockEventsByDay = new Map<number, any[]>();
  blockList.forEach(block => {
    const startDate = new Date(block.start_time);
    if (startDate.getMonth() === month && startDate.getFullYear() === year) {
       const d = startDate.getDate();
       const existing = blockEventsByDay.get(d) || [];
       existing.push(block);
       blockEventsByDay.set(d, existing);
    }
  });

  // Priority Alerts
  const alerts = [];
  trainStates.forEach(t => {
    if (t.status === 'Halted' || t.status === 'Blocked') {
      alerts.push({ id: `train-${t.train_id}`, type: 'critical', icon: 'warning', title: `Train ${t.status}`, desc: `Train ${t.train_id} is currently ${t.status.toLowerCase()} on edge ${t.edge_id}.`, time: 'Live' });
    }
  });
  if (impactReport && impactReport.status === 'blocks_active') {
    alerts.push({ id: `impact-report`, type: 'info', icon: 'info', title: 'Maintenance Ripple Effect', desc: impactReport.message, time: 'Active' });
  }
  if (alerts.length === 0) {
    alerts.push({ id: `all-clear`, type: 'success', icon: 'check_circle', title: 'System Nominal', desc: 'No critical wear, collisions, or halted trains detected in the network.', time: 'Live' });
  }

  return (
    <div className="p-8 max-w-[1600px] mx-auto w-full space-y-6">
      
      {/* ── Top Shared Map Section ── */}
      <section className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-outline-variant/10 relative flex flex-col h-[400px]">
        <h3 className="text-lg font-bold text-on-surface mb-2 flex items-center justify-between pointer-events-none z-10 relative">
            <span className="flex items-center gap-2">
                <span className="material-symbols-outlined">map</span>
                Live Topology Map
            </span>
            <span className="flex items-center gap-2 text-xs font-bold text-emerald-600 bg-emerald-100 px-3 py-1 rounded-full shadow-sm pointer-events-auto border border-emerald-200">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                LIVE FEED
            </span>
        </h3>
        <div className="flex-1 w-full bg-slate-900 rounded-xl relative overflow-hidden">
            <KineticMap />
        </div>
      </section>

      {/* ── 3-Tab Navigation ── */}
      <div className="flex border-b border-outline-variant/10 gap-8 mt-4">
        {[
          { id: 'simulate', label: 'Simulation & What-If', icon: 'science' },
          { id: 'maintenance', label: 'Maintenance Scheduling', icon: 'calendar_month' },
          { id: 'audit', label: 'System Audit Log', icon: 'receipt_long' },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center gap-2 pb-4 px-2 font-bold transition-all relative ${
              activeTab === tab.id ? 'text-primary' : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            <span className="material-symbols-outlined text-sm">{tab.icon}</span>
            {tab.label}
            {activeTab === tab.id && (
              <motion.div layoutId="tab-indicator" className="absolute bottom-0 left-0 w-full h-1 bg-primary rounded-t-lg" />
            )}
          </button>
        ))}
      </div>

      {/* ── Tab Content Area ── */}
      <motion.div 
        key={activeTab}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="pt-4"
      >
        {/* ================= SIMULATE TAB ================= */}
        {activeTab === 'simulate' && (
          <div className="grid grid-cols-12 gap-8">
            <div className="col-span-12 lg:col-span-4 flex flex-col gap-6">
                <section className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-outline-variant/10">
                    <div className="flex justify-between items-center mb-6">
                        <h3 className="text-lg font-bold text-on-surface flex items-center gap-2">
                            <span className="material-symbols-outlined text-violet-500">settings_input_component</span>
                            Simulation Control
                        </h3>
                        <button 
                            onClick={handleDeploy}
                            className="bg-[#8B5CF6] text-white px-4 py-1.5 rounded text-xs font-bold shadow-md hover:bg-[#7c3aed] transition-colors"
                        >
                            Deploy
                        </button>
                    </div>
                    <div className="space-y-6">
                        <div>
                            <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Primary Delay Node</label>
                            <div className="relative">
                                <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 z-10">train</span>
                                <select 
                                    className="w-full pl-10 pr-10 py-3 bg-surface-container-low border-none rounded-sm font-mono text-sm focus:ring-2 focus:ring-violet-500/30 text-on-surface appearance-none cursor-pointer" 
                                    value={delayTrainId}
                                    onChange={e => setDelayTrainId(e.target.value)}
                                >
                                    <option value="">Select a train...</option>
                                    {trainStates.map(t => <option key={t.train_id} value={t.train_id}>{t.train_id}</option>)}
                                </select>
                                <span className="material-symbols-outlined absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none">expand_more</span>
                            </div>
                        </div>
                        <div>
                            <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Latency Duration (min)</label>
                            <input 
                                className="w-full h-1.5 bg-surface-container rounded-lg appearance-none cursor-pointer accent-violet-500" 
                                type="range" min="0" max="60" value={latency} onChange={e => setLatency(parseInt(e.target.value))}
                            />
                            <div className="flex justify-between mt-2 text-xs font-medium text-on-surface-variant">
                                <span>0</span><span className="text-violet-600 font-bold">{latency} min</span><span>60</span>
                            </div>
                        </div>
                        <div>
                            <label className="block text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">Force Action (optional)</label>
                            <div className="flex gap-2">
                                <select
                                    className="flex-1 px-3 py-2 bg-surface-container-low border-none rounded-sm font-mono text-xs text-on-surface"
                                    value=""
                                    onChange={e => {
                                        const action = parseInt(e.target.value);
                                        if (delayTrainId) setForcedActions(prev => ({ ...prev, [delayTrainId]: action }));
                                    }}
                                >
                                    <option value="" disabled>Select action for {delayTrainId || 'train'}...</option>
                                    <option value="0">HOLD</option>
                                    <option value="1">PROCEED (MAIN)</option>
                                    <option value="2">DIVERT</option>
                                </select>
                            </div>
                            {Object.keys(forcedActions).length > 0 && (
                                <div className="flex flex-wrap gap-1 mt-2">
                                    {Object.entries(forcedActions).map(([tid, act]) => (
                                        <span key={tid} className="text-[10px] bg-violet-100 text-violet-700 px-2 py-1 rounded-full flex items-center gap-1">
                                            {tid}: {['HOLD','MAIN','DIVERT'][act]}
                                            <button onClick={() => setForcedActions(prev => { const n = {...prev}; delete n[tid]; return n; })}>×</button>
                                        </span>
                                    ))}
                                </div>
                            )}
                        </div>
                        <button 
                            onClick={handleRecalculate} disabled={isRecalculating}
                            className="w-full bg-[#8B5CF6] text-white py-4 rounded-sm font-bold flex items-center justify-center gap-2 hover:opacity-90 transition-all disabled:opacity-50">
                            <span className={`material-symbols-outlined ${isRecalculating ? 'animate-spin' : ''}`}>refresh</span>
                            {isRecalculating ? 'Recalculating...' : 'Recalculate Network'}
                        </button>
                        <div className="pt-4 border-t border-surface-container">
                            <div className="flex justify-between items-center mb-2">
                                <p className="text-xs font-bold text-slate-500 uppercase">Scenarios ({scenarios.length})</p>
                                {scenarios.length > 0 && (
                                    <button onClick={() => setScenarios([])} className="text-[10px] text-rose-500 font-bold">Clear all</button>
                                )}
                            </div>
                            <div className="space-y-2 max-h-[280px] overflow-y-auto">
                                {scenarios.length === 0 && (
                                    <p className="text-xs text-on-surface-variant italic">Run a scenario to see results here.</p>
                                )}
                                {scenarios.map(s => (
                                    <div key={s.id} className="bg-surface-container-low p-3 rounded-lg">
                                        <p className="text-xs font-bold text-violet-600 mb-1">{s.label}</p>
                                        <div className="grid grid-cols-2 gap-2">
                                            <div>
                                                <p className="text-[9px] font-bold text-slate-500 uppercase">Reliability</p>
                                                <p className="text-lg font-extrabold text-error">{s.impact.reliability}</p>
                                            </div>
                                            <div>
                                                <p className="text-[9px] font-bold text-slate-500 uppercase">Congestion</p>
                                                <p className="text-lg font-extrabold text-on-surface-variant">{s.impact.congestion}</p>
                                            </div>
                                        </div>
                                        {Object.keys(s.forcedActions).length > 0 && (
                                            <p className="text-[9px] text-on-surface-variant mt-1">
                                                Forced: {Object.entries(s.forcedActions).map(([t, a]) => `${t}→${['HOLD','MAIN','DIVERT'][a]}`).join(', ')}
                                            </p>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </section>
                
                <section className="sandbox-mms-panel rounded-lg shadow-sm h-full max-h-[300px] flex flex-col border border-outline-variant/10">
                    <div className="sandbox-mms-header p-4">
                        <h3 className="sandbox-mms-title text-sm">
                            <span className="material-symbols-outlined text-amber-500 text-[16px]">construction</span>
                            What-If Blocks
                        </h3>
                        <button className="sandbox-mms-add-btn text-xs px-2 py-1" onClick={() => openDrawer()}>
                            <span className="material-symbols-outlined text-[12px]">add</span> Add
                        </button>
                    </div>
                    {blockList.length === 0 ? (
                        <div className="sandbox-mms-empty flex-1 flex flex-col items-center justify-center py-6">
                            <p className="text-xs text-on-surface-variant text-center px-4">No active blocks. Add a What-If block to simulate repairs.</p>
                        </div>
                    ) : (
                        <div className="sandbox-block-list overflow-y-auto flex-1 p-2">
                            {blockList.map((blk) => (
                                <div key={blk.blockId} className={`sandbox-block-card p-2 text-sm ${blk.isWhatIf ? 'whatif' : ''}`}>
                                    <div className="flex-1 min-w-0">
                                        <div className="font-bold truncate">{blk.element_id}</div>
                                        <div className="text-[10px] flex gap-2 mt-1">
                                            <span className={blk.severity === 'TOTAL_BLOCK' ? 'text-rose-500 font-bold' : 'text-amber-500 font-bold'}>
                                                {blk.severity === 'TOTAL_BLOCK' ? '🔴 BLOCKED' : '🟡 RESTRICTED'}
                                            </span>
                                            <span className="text-on-surface-variant truncate">{blk.type}</span>
                                        </div>
                                    </div>
                                    <button onClick={() => removeBlockRemote(blk.element_id)} className="text-on-surface-variant hover:text-rose-500">
                                        <span className="material-symbols-outlined text-[16px]">close</span>
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </section>
            </div>
            
            <div className="col-span-12 lg:col-span-8 flex flex-col">
                <section className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-outline-variant/10 h-full min-h-[400px] flex flex-col">
                    <div className="flex justify-between items-center mb-6">
                        <h3 className="text-lg font-bold text-on-surface">Interactive Marey Topology</h3>
                        <div className="flex gap-4">
                            <span className="flex items-center gap-1.5 text-xs font-bold text-slate-500"><span className="w-3 h-3 rounded-full bg-[#8B5CF6]"></span> Projected</span>
                            <span className="flex items-center gap-1.5 text-xs font-bold text-slate-500"><span className="w-3 h-3 rounded-full bg-slate-300"></span> Historical</span>
                        </div>
                    </div>
                    <div className="flex-1 relative bg-surface-container-low rounded-xl overflow-hidden p-4">
                        <div className="absolute inset-0 opacity-20 pointer-events-none" style={{ backgroundImage: "radial-gradient(#8B5CF6 0.5px, transparent 0.5px)", backgroundSize: "24px 24px" }}></div>
                        <MareyTimeline />
                        <div className="absolute bottom-6 left-6 bg-white/70 backdrop-blur-md p-4 rounded-lg border border-white shadow-xl max-w-xs">
                            <p className="text-xs font-bold text-violet-600 mb-1">Impact Highlight</p>
                            <p className="text-sm text-on-surface leading-tight font-medium">
                                {scenarios.length > 0
                                    ? `${scenarios[scenarios.length - 1].label}: ${scenarios.length} scenario${scenarios.length > 1 ? 's' : ''} compared. Latest reliability: ${scenarios[scenarios.length - 1].impact.reliability}.`
                                    : `Node ${delayTrainId || 'selected train'} might cause bottlenecks.`}
                            </p>
                        </div>
                    </div>
                </section>
            </div>
          </div>
        )}

        {/* ================= MAINTENANCE TAB ================= */}
        {activeTab === 'maintenance' && (
          <div className="grid grid-cols-12 gap-8">
            <motion.section className="col-span-12 lg:col-span-4 bg-surface-container-lowest rounded-lg p-6 border border-outline-variant/10 shadow-sm flex flex-col" variants={fadeUp as any}>
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-bold tracking-tight">Priority Alerts</h3>
                {alerts.length > 0 && <span className="bg-tertiary text-on-tertiary text-[10px] px-2 py-0.5 rounded-full font-bold">{alerts.length}</span>}
              </div>
              <div className="space-y-4 overflow-y-auto pr-2 max-h-[400px]">
                {alerts.map((alert) => (
                  <div key={alert.id} className={`flex gap-4 p-4 rounded-lg ${
                    alert.type === 'critical' ? 'bg-tertiary-container/10 border-l-4 border-tertiary' : 
                    alert.type === 'info' ? 'bg-surface-container border-l-4 border-amber-400' : 'bg-surface-container opacity-60'
                  }`}>
                    <span className={`material-symbols-outlined ${
                      alert.type === 'critical' ? 'text-tertiary' : alert.type === 'info' ? 'text-amber-500' : 'text-primary'
                    }`}>{alert.icon}</span>
                    <div>
                      <h4 className={`text-sm font-bold ${alert.type === 'critical' ? 'text-on-tertiary-container' : 'text-on-surface'}`}>{alert.title}</h4>
                      <p className="text-xs text-on-surface-variant mt-1 leading-relaxed">{alert.desc}</p>
                      <span className={`text-[10px] font-bold mt-2 inline-block ${alert.type === 'critical' ? 'text-tertiary-dim' : 'text-on-surface-variant'}`}>{alert.time}</span>
                    </div>
                  </div>
                ))}
              </div>
            </motion.section>

            <motion.section className="col-span-12 lg:col-span-8 bg-surface-container-lowest rounded-lg p-6 border border-outline-variant/10 shadow-sm" variants={fadeUp as any}>
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h3 className="text-lg font-bold tracking-tight">Maintenance Calendar</h3>
                  <p className="text-xs text-on-surface-variant mt-1">Upcoming service windows</p>
                </div>
                <div className="flex gap-2 items-center">
                  <button onClick={() => setCalendarDate(new Date(year, month - 1, 1))} className="p-1 hover:bg-surface-container rounded transition-colors"><span className="material-symbols-outlined">chevron_left</span></button>
                  <span className="text-sm font-bold w-32 text-center">{currentMonthName}</span>
                  <button onClick={() => setCalendarDate(new Date(year, month + 1, 1))} className="p-1 hover:bg-surface-container rounded transition-colors"><span className="material-symbols-outlined">chevron_right</span></button>
                </div>
              </div>
              <div className="overflow-x-auto w-full pb-2">
                <div className="grid grid-cols-7 gap-px bg-outline-variant/10 rounded-lg border border-outline-variant/10 min-w-[700px]">
                  {['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'].map(day => (
                    <div key={day} className="bg-surface-container py-2 text-center text-[10px] font-bold text-on-surface-variant">{day}</div>
                  ))}
                  {daysArray.map((day, idx) => {
                    if (day === null) return <div key={`empty-${idx}`} className="bg-surface-container-lowest min-h-[90px] p-2 opacity-30 text-[10px] font-bold"></div>;
                    const events = blockEventsByDay.get(day) || [];
                    const isToday = day === now.getDate() && month === now.getMonth() && year === now.getFullYear();
                    return (
                      <div key={`day-${day}`} className={`min-h-[90px] p-2 text-[10px] font-bold border border-transparent ${isToday ? 'bg-primary/5 border-primary/20' : 'bg-surface-container-lowest'}`}>
                        <span className="mb-1 inline-block">{day}</span>
                        <div className="space-y-1">
                          {events.map((evt, eIdx) => (
                            <div key={eIdx} className={`group relative p-1 rounded text-[9px] leading-tight flex flex-col ${evt.severity === 'TOTAL_BLOCK' ? 'bg-rose-100 text-rose-800' : 'bg-amber-100 text-amber-800'}`}>
                              <div className="flex justify-between items-start">
                                <span className="font-extrabold truncate pr-1">{evt.element_id}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </motion.section>
          </div>
        )}

        {/* ================= AUDIT LOG TAB ================= */}
        {activeTab === 'audit' && (
          <motion.section className="bg-surface-container-lowest rounded-lg p-6 border border-outline-variant/10 shadow-sm" variants={fadeUp as any}>
            <div className="flex justify-between items-center mb-6">
              <div>
                <h3 className="text-lg font-bold tracking-tight">System Event Log</h3>
                <p className="text-xs text-on-surface-variant mt-1">Immutable ledger of system interventions</p>
              </div>
              <div className="flex items-center gap-4">
                <select 
                  value={logFilter} onChange={(e) => setLogFilter(e.target.value)}
                  className="bg-surface-container border-none rounded text-xs font-bold py-1.5 pl-3 pr-8 focus:ring-0 outline-none cursor-pointer"
                >
                  <option value="All">All Events</option>
                  <option value="Critical Only">Critical Only</option>
                  <option value="Manual Only">Manual Only</option>
                </select>
                <button onClick={handleExportLog} className="bg-surface-container text-on-surface-variant px-4 py-1.5 rounded text-xs font-bold hover:bg-surface-container-high transition-colors flex items-center gap-2">
                  <span className="material-symbols-outlined text-[14px]">download</span> Export
                </button>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-outline-variant/10">
                    <th className="py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Timestamp</th>
                    <th className="py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Event Source</th>
                    <th className="py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Action</th>
                    <th className="py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Operator</th>
                    <th className="py-3 text-[10px] font-bold text-on-surface-variant uppercase tracking-widest text-right">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/5">
                  {systemLogs.filter(log => {
                    if (logFilter === 'Critical Only') return log.statusType === 'error';
                    if (logFilter === 'Manual Only') return log.operator === 'Dispatcher';
                    return true;
                  }).length === 0 ? (
                    <tr><td colSpan={5} className="py-6 text-center text-xs text-on-surface-variant">No system activity matches this filter.</td></tr>
                  ) : systemLogs.filter(log => {
                    if (logFilter === 'Critical Only') return log.statusType === 'error';
                    if (logFilter === 'Manual Only') return log.operator === 'Dispatcher';
                    return true;
                  }).map((log, i) => (
                    <tr key={i} className="hover:bg-surface-container-low transition-colors">
                      <td className="py-3 px-2 text-xs font-mono text-slate-500">{log.t}</td>
                      <td className="py-3 px-2 text-xs font-bold text-on-surface">{log.source}</td>
                      <td className="py-3 px-2 text-xs text-on-surface-variant">{log.action}</td>
                      <td className={`py-3 px-2 text-xs font-medium ${
                        log.statusType === 'success' ? 'text-primary' : log.statusType === 'error' ? 'text-rose-600' : 'text-slate-500'
                      }`}>{log.operator}</td>
                      <td className="py-3 px-2 text-right">
                        <span className={`text-[9px] px-2 py-0.5 rounded font-bold uppercase ${
                          log.statusType === 'success' ? 'bg-emerald-100 text-emerald-700' :
                          log.statusType === 'error' ? 'bg-rose-100 text-rose-700' :
                          log.statusType === 'warning' ? 'bg-amber-100 text-amber-700' : 'bg-slate-100 text-slate-500'
                        }`}>{log.status}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-4 flex justify-center">
              <button onClick={() => setLogLimit(prev => prev + 50)} className="text-[10px] font-bold text-primary flex items-center gap-1 hover:underline">
                LOAD MORE
              </button>
            </div>
          </motion.section>
        )}
      </motion.div>
    </div>
  );
};

export default ControlCentre;
