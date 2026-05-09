import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { KineticMap } from '../components/KineticMap/KineticMap';
import { useMaintenanceStore } from '../store/useMaintenanceStore';
import { useMapStore } from '../store/useMapStore';

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

/* ────────────────────────────────────────────────────────────────
   DATA CONFIGURATIONS
───────────────────────────────────────────────────────────────── */


const INITIAL_LOGS = [
  { t: '2023-10-05 14:22:01', source: 'CORE_DAMPENER_4A', action: 'Manual override: Resistance adjusted to +15%', operator: 'Chief Miller', status: 'Executed', statusType: 'success' },
  { t: '2023-10-05 14:18:55', source: 'SYS_WATCHDOG_ALPHA', action: 'Automated Failsafe: Lateral drift recalibrated', operator: 'System AI', status: 'Executed', statusType: 'success' },
  { t: '2023-10-05 14:15:30', source: 'SEC_ACCESS_GATE_9', action: 'Access Denied: Invalid biometric signature', operator: 'Unknown', status: 'Rejected', statusType: 'error' },
  { t: '2023-10-05 13:40:12', source: 'MAINT_BOT_UNIT_7', action: 'Status: Recharging @ Deck 2 Station', operator: 'System AI', status: 'Idle', statusType: 'neutral' },
];

/* ────────────────────────────────────────────────────────────────
   COMPONENT
───────────────────────────────────────────────────────────────── */
const MaintenanceLogs: React.FC = () => {
  useEffect(() => {
    document.title = 'Maintenance & Logs - Zentra Ops | ORBIT';
  }, []);

  const activeBlocks = useMaintenanceStore((s) => s.activeBlocks);
  const fetchActiveBlocks = useMaintenanceStore((s) => s.fetchActiveBlocks);
  const removeBlockRemote = useMaintenanceStore((s) => s.removeBlockRemote);
  const impactReport = useMaintenanceStore((s) => s.impactReport);
  const trainStates = useMapStore((s) => s.trainStates);

  const [isLockdown, setIsLockdown] = useState(false);
  const [isSafetyShield, setIsSafetyShield] = useState(true);
  const [isAutoCommit, setIsAutoCommit] = useState(false);
  const [calendarDate, setCalendarDate] = useState(new Date());
  const [logFilter, setLogFilter] = useState('All');
  const [logLimit, setLogLimit] = useState(50);

  const handleLockdown = async (enabled: boolean) => {
    if (enabled) {
      if (!window.confirm("🚨 WARNING: Are you sure you want to trigger a System Lockdown? This will halt all trains immediately!")) {
        return;
      }
    }
    setIsLockdown(enabled);
    await fetch('/api/v1/system/lockdown', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled }) });
  };
  const handleSafetyShield = async (enabled: boolean) => {
    const action = enabled ? "ENABLE" : "DISABLE";
    if (!window.confirm(`⚠️ WARNING: Are you sure you want to ${action} the OR-Shield Safety Protocol? ${!enabled ? 'Disabling this will allow the AI to bypass hard-constraint safety checks!' : ''}`)) {
      return;
    }
    setIsSafetyShield(enabled);
    await fetch('/api/v1/system/safety-shield', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled }) });
  };
  const handleAutoCommit = async (enabled: boolean) => {
    const action = enabled ? "ENABLE" : "DISABLE";
    if (!window.confirm(`⚠️ WARNING: Are you sure you want to ${action} AI Auto-Commit? ${enabled ? 'Enabling this will allow the AI to execute decisions without human approval!' : ''}`)) {
      return;
    }
    setIsAutoCommit(enabled);
    await fetch('/api/v1/system/auto-commit', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled }) });
  };

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('/api/v1/system/inference-status');
        if (res.ok) {
          const data = await res.json();
          setIsLockdown(data.lockdown);
          setIsSafetyShield(data.safety_shield);
          setIsAutoCommit(data.auto_commit);
        }
      } catch (e) {
        console.error("Failed to fetch system status:", e);
      }
    };
    
    fetchStatus();
    fetchActiveBlocks();
  }, [fetchActiveBlocks]);

  const [systemLogs, setSystemLogs] = useState<any[]>([]);

  useEffect(() => {
    const loadLogs = async () => {
      try {
        const res = await fetch(`/api/v1/system/audit-logs?limit=${logLimit}`);
        const data = await res.json();
        
        const logs = (data.logs || []).map((l: any) => ({
          ...l,
          t: new Date(l.t).toLocaleString()
        }));

        setSystemLogs(logs);
      } catch (err) {
        console.error(err);
      }
    };

    loadLogs();
    const interval = setInterval(loadLogs, 3000);
    return () => clearInterval(interval);
  }, [logLimit]);

  const handleExportLog = () => {
    if (systemLogs.length === 0) {
      alert("No logs available to export.");
      return;
    }

    let report = `====================================================\n`;
    report += ` ORBIT COMMAND CENTER - SYSTEM AUDIT LOG\n`;
    report += ` Generated: ${new Date().toLocaleString()}\n`;
    report += ` Total Entries: ${systemLogs.length}\n`;
    report += `====================================================\n\n`;

    systemLogs.forEach((log) => {
      report += `[${log.t}]\n`;
      report += `SOURCE:   ${log.source}\n`;
      report += `ACTION:   ${log.action}\n`;
      report += `OPERATOR: ${log.operator}\n`;
      report += `STATUS:   ${log.status.toUpperCase()}\n`;
      report += `----------------------------------------------------\n`;
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
  Array.from(activeBlocks.values()).forEach(block => {
    const startDate = new Date(block.start_time);
    if (startDate.getMonth() === month && startDate.getFullYear() === year) {
       const d = startDate.getDate();
       const existing = blockEventsByDay.get(d) || [];
       existing.push(block);
       blockEventsByDay.set(d, existing);
    }
  });

  // Generate Priority Alerts
  const alerts = [];
  trainStates.forEach(t => {
    if (t.status === 'Halted' || t.status === 'Blocked') {
      alerts.push({
        id: `train-${t.train_id}`,
        type: 'critical',
        icon: 'warning',
        title: `Train ${t.status}`,
        desc: `Train ${t.train_id} is currently ${t.status.toLowerCase()} on edge ${t.edge_id}. Possible gridlock or maintenance conflict.`,
        time: 'Live'
      });
    }
  });

  if (impactReport && impactReport.status === 'blocks_active') {
    alerts.push({
      id: `impact-report`,
      type: 'info',
      icon: 'info',
      title: 'Maintenance Ripple Effect',
      desc: impactReport.message,
      time: 'Active'
    });
  }

  if (alerts.length === 0) {
    alerts.push({
      id: `all-clear`,
      type: 'success',
      icon: 'check_circle',
      title: 'System Nominal',
      desc: 'No critical wear, collisions, or halted trains detected in the network.',
      time: 'Live'
    });
  }

  return (
    <div className="p-8 max-w-[1600px] mx-auto w-full">
      {/* Header Section */}
      <header className="mb-10 flex justify-between items-end">
        <div>
          <h2 className="text-4xl font-extrabold tracking-tighter text-on-surface">Maintenance scheduler</h2>
        </div>
        <div className="flex gap-3">
          <button 
            onClick={handleExportLog}
            className="bg-surface-container text-on-primary-container px-6 py-2.5 rounded-sm font-semibold text-sm flex items-center gap-2 hover:bg-surface-container-high transition-colors"
          >
            <span className="material-symbols-outlined text-sm">download</span>
            Export Full Log
          </button>
        </div>
      </header>

      {/* Bento Grid Layout */}
      <motion.div 
        className="grid grid-cols-12 gap-6"
        initial="hidden"
        animate="visible"
        variants={stagger}
      >
        {/* Kinetic Network Map */}
        <motion.section 
          className="col-span-12 lg:col-span-8 bg-surface-container-lowest rounded-lg border border-outline-variant/10 shadow-sm relative overflow-hidden flex flex-col"
          variants={fadeUp}
          custom={0.0}
          style={{ minHeight: '550px' }}
        >
          <div className="absolute top-0 left-0 w-full p-6 z-10 pointer-events-none bg-gradient-to-b from-surface-container-lowest to-transparent flex justify-between items-start">
            <div>
              <h3 className="text-xl font-bold tracking-tight mb-1 flex items-center gap-2">
                <span className="material-symbols-outlined">map</span>
                Network Topology Map
              </h3>
              <p className="text-sm text-on-surface-variant">Select a track or block to schedule maintenance.</p>
            </div>
            <div className="flex items-center gap-2 bg-emerald-50 text-emerald-700 px-3 py-1 rounded-full text-xs font-bold animate-pulse pointer-events-auto shadow-md border border-emerald-200">
              <span className="w-2 h-2 bg-emerald-500 rounded-full"></span>
              LIVE FEED
            </div>
          </div>
          <div className="w-full flex-1 pt-20 pb-4 h-full">
            <KineticMap />
          </div>
        </motion.section>

        {/* Manual Overrides Section */}
        <motion.section 
          className="col-span-12 lg:col-span-4 bg-surface-container-low rounded-lg p-6 flex flex-col"
          variants={fadeUp}
          custom={0.1}
        >
          <h3 className="text-lg font-bold tracking-tight mb-6 flex-shrink-0">Manual Overrides</h3>
          <div className="flex flex-col flex-grow justify-between gap-6">
            <div className="flex flex-col gap-6 flex-grow">
              <div className="flex-1 flex flex-col justify-center bg-surface-container-lowest p-6 rounded-lg border border-outline-variant/10 shadow-sm">
                <div className="flex justify-between items-center mb-4">
                  <div className="flex items-center gap-3">
                    <span className="material-symbols-outlined text-primary" style={{ fontVariationSettings: "'FILL' 1" }}>waves</span>
                    <span className="text-sm font-bold">AI Auto-Commit</span>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" checked={isAutoCommit} onChange={e => handleAutoCommit(e.target.checked)} className="sr-only peer" />
                    <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
                  </label>
                </div>
                <p className="text-xs text-on-surface-variant leading-relaxed">Automatically executes AI proposals with high confidence without manual approval.</p>
              </div>

              <div className="flex-1 flex flex-col justify-center bg-surface-container-lowest p-6 rounded-lg border border-outline-variant/10 shadow-sm">
                <div className="flex justify-between items-center mb-4">
                  <div className="flex items-center gap-3">
                    <span className="material-symbols-outlined text-tertiary" style={{ fontVariationSettings: "'FILL' 1" }}>security</span>
                    <span className="text-sm font-bold">OR-Shield Safety Protocol</span>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" checked={isSafetyShield} onChange={e => handleSafetyShield(e.target.checked)} className="sr-only peer" />
                    <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-tertiary"></div>
                  </label>
                </div>
                <p className="text-xs text-on-surface-variant leading-relaxed">Enforces hard constraints before dispatching commands.</p>
              </div>
            </div>
            
            <button 
              onClick={() => handleLockdown(!isLockdown)}
              className={`w-full py-4 rounded-lg font-bold text-sm tracking-widest flex items-center justify-center gap-3 transition-colors mt-auto flex-shrink-0 ${
                isLockdown ? 'bg-rose-600 text-white hover:bg-rose-700' : 'bg-slate-900 text-white hover:bg-slate-800'
              }`}>
              <span className="material-symbols-outlined text-sm">{isLockdown ? 'lock_open' : 'lock'}</span>
              {isLockdown ? 'LIFT SYSTEM LOCKDOWN' : 'CONFIRM SYSTEM LOCKDOWN'}
            </button>
          </div>
        </motion.section>

        {/* Priority Alerts Feed */}
        <motion.section 
          className="col-span-12 lg:col-span-4 bg-surface-container-lowest rounded-lg p-6 border border-outline-variant/10 shadow-sm overflow-hidden flex flex-col h-full"
          variants={fadeUp}
          custom={0.2}
        >
          <div className="flex items-center justify-between mb-6 flex-shrink-0">
            <h3 className="text-lg font-bold tracking-tight">Priority Alerts</h3>
            {alerts.length > 0 && <span className="bg-tertiary text-on-tertiary text-[10px] px-2 py-0.5 rounded-full font-bold">{alerts.length} ALERTS</span>}
          </div>
          <div className="space-y-4 overflow-y-auto flex-grow pr-2">
            {alerts.map((alert) => (
              <div key={alert.id} className={`flex gap-4 p-4 rounded-lg ${
                alert.type === 'critical' ? 'bg-tertiary-container/10 border-l-4 border-tertiary' : 
                alert.type === 'info' ? 'bg-surface-container border-l-4 border-amber-400' : 'bg-surface-container opacity-60'
              }`}>
                <span className={`material-symbols-outlined ${
                  alert.type === 'critical' ? 'text-tertiary' : 
                  alert.type === 'info' ? 'text-amber-500' : 'text-primary'
                }`}>{alert.icon}</span>
                <div>
                  <h4 className={`text-sm font-bold ${
                    alert.type === 'critical' ? 'text-on-tertiary-container' : 'text-on-surface'
                  }`}>{alert.title}</h4>
                  <p className="text-xs text-on-surface-variant mt-1 leading-relaxed">{alert.desc}</p>
                  <span className={`text-[10px] font-bold mt-2 inline-block ${
                    alert.type === 'critical' ? 'text-tertiary-dim' : 'text-on-surface-variant'
                  }`}>{alert.time}</span>
                </div>
              </div>
            ))}
          </div>
          <button className="w-full mt-6 py-3 text-xs font-bold text-primary border border-primary/20 rounded-lg hover:bg-primary/5 transition-colors flex-shrink-0">
            View All Alerts
          </button>
        </motion.section>

        {/* Maintenance Schedule Calendar */}
        <motion.section 
          className="col-span-12 lg:col-span-8 bg-surface-container-lowest rounded-lg p-8 border border-outline-variant/10 shadow-sm"
          variants={fadeUp}
          custom={0.3}
        >
          <div className="flex justify-between items-center mb-8">
            <div>
              <h3 className="text-xl font-bold tracking-tight">Maintenance Schedule</h3>
              <p className="text-sm text-on-surface-variant">Upcoming service windows and fleet rotations</p>
            </div>
            <div className="flex gap-2">
              <button 
                onClick={() => setCalendarDate(new Date(year, month - 1, 1))}
                className="p-2 hover:bg-surface-container rounded-lg transition-colors flex items-center justify-center">
                <span className="material-symbols-outlined">chevron_left</span>
              </button>
              <span className="text-sm font-bold flex items-center px-2">{currentMonthName}</span>
              <button 
                onClick={() => setCalendarDate(new Date(year, month + 1, 1))}
                className="p-2 hover:bg-surface-container rounded-lg transition-colors flex items-center justify-center">
                <span className="material-symbols-outlined">chevron_right</span>
              </button>
            </div>
          </div>
          <div className="overflow-x-auto w-full pb-4">
            <div className="grid grid-cols-7 gap-px bg-outline-variant/10 rounded-lg border border-outline-variant/10 min-w-[700px]">
              {/* Day Headers */}
              {['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'].map(day => (
                <div key={day} className="bg-surface-container py-3 text-center text-[10px] font-bold text-on-surface-variant">{day}</div>
              ))}
              
              {/* Calendar Cells */}
              {daysArray.map((day, idx) => {
                if (day === null) {
                  return <div key={`empty-${idx}`} className="bg-surface-container-lowest min-h-[100px] p-2 opacity-30 text-[10px] font-bold"></div>;
                }

                const events = blockEventsByDay.get(day) || [];
                const hasEvents = events.length > 0;
                const isToday = day === now.getDate();

                return (
                  <div key={`day-${day}`} className={`min-h-[100px] p-2 text-[10px] font-bold border border-transparent ${isToday ? 'bg-primary/5 border-primary/20' : 'bg-surface-container-lowest'}`}>
                    <span className="mb-1 inline-block">{day}</span>
                    <div className="space-y-1">
                      {events.map((evt, eIdx) => (
                        <div key={eIdx} className={`group relative p-1.5 rounded text-[9px] leading-tight flex flex-col text-left transition-colors ${evt.severity === 'TOTAL_BLOCK' ? 'bg-rose-100 text-rose-800 border border-rose-200 hover:bg-rose-200' : 'bg-amber-100 text-amber-800 border border-amber-200 hover:bg-amber-200'}`}>
                          <div className="flex justify-between items-start">
                            <span className="font-extrabold truncate pr-1">{evt.element_id}</span>
                            <button 
                              onClick={(e) => { e.stopPropagation(); removeBlockRemote(evt.element_id); }}
                              className="opacity-0 group-hover:opacity-100 p-[1px] hover:bg-black/10 rounded transition-opacity flex-shrink-0"
                              title="Remove Maintenance Block"
                            >
                              <span className="material-symbols-outlined text-[12px]">close</span>
                            </button>
                          </div>
                          <span className="truncate opacity-80 mt-0.5">{evt.reason || 'Maintenance'}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </motion.section>

        {/* System Event Log (Full Width) */}
        <motion.section 
          className="col-span-12 bg-surface-container-lowest rounded-lg p-8 border border-outline-variant/10 shadow-sm"
          variants={fadeUp}
          custom={0.4}
        >
          <div className="flex justify-between items-center mb-8">
            <div>
              <h3 className="text-xl font-bold tracking-tight">System Event Log</h3>
              <p className="text-sm text-on-surface-variant">Immutable ledger of all automated and manual system interventions</p>
            </div>
            <div className="flex items-center gap-4">
              <div className="relative">
                <select 
                  value={logFilter}
                  onChange={(e) => setLogFilter(e.target.value)}
                  className="bg-surface-container border-none rounded-lg text-xs font-bold text-on-surface-variant py-2 pl-3 pr-8 focus:ring-0 appearance-none outline-none cursor-pointer"
                >
                  <option value="All">All Events</option>
                  <option value="Critical Only">Critical Only</option>
                  <option value="Manual Only">Manual Only</option>
                </select>
                <span className="material-symbols-outlined absolute right-2 top-1/2 -translate-y-1/2 text-[16px] pointer-events-none text-on-surface-variant">arrow_drop_down</span>
              </div>
              <div className="h-8 w-px bg-outline-variant/20"></div>
              <button className="p-2 text-slate-500 hover:text-on-surface transition-colors flex items-center justify-center">
                <span className="material-symbols-outlined">filter_list</span>
              </button>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-outline-variant/10">
                  <th className="py-4 text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Timestamp</th>
                  <th className="py-4 text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Event Source</th>
                  <th className="py-4 text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Action / Change</th>
                  <th className="py-4 text-[10px] font-bold text-on-surface-variant uppercase tracking-widest">Operator</th>
                  <th className="py-4 text-[10px] font-bold text-on-surface-variant uppercase tracking-widest text-right">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-outline-variant/5 border-t border-transparent space-y-1">
                {systemLogs.filter(log => {
                  if (logFilter === 'Critical Only') return log.statusType === 'error';
                  if (logFilter === 'Manual Only') return log.operator === 'Dispatcher';
                  return true;
                }).length === 0 ? (
                  <tr>
                    <td colSpan={5} className="py-8 text-center text-sm text-on-surface-variant">No system activity matches this filter.</td>
                  </tr>
                ) : systemLogs.filter(log => {
                  if (logFilter === 'Critical Only') return log.statusType === 'error';
                  if (logFilter === 'Manual Only') return log.operator === 'Dispatcher';
                  return true;
                }).map((log, i) => (
                  <tr key={i} className="group hover:bg-surface-container-low transition-colors">
                    <td className="py-4 px-2 text-xs font-mono text-slate-500">{log.t}</td>
                    <td className="py-4 px-2 text-xs font-bold text-on-surface">{log.source}</td>
                    <td className="py-4 px-2 text-xs text-on-surface-variant">{log.action}</td>
                    <td className={`py-4 px-2 text-xs font-medium ${
                      log.statusType === 'success' ? 'text-primary' : 
                      log.statusType === 'error' ? 'text-rose-600' : 'text-slate-500'
                    }`}>{log.operator}</td>
                    <td className="py-4 px-2 text-right">
                      <span className={`text-[10px] px-2 py-0.5 rounded font-bold uppercase ${
                        log.statusType === 'success' ? 'bg-emerald-100 text-emerald-700' :
                        log.statusType === 'error' ? 'bg-rose-100 text-rose-700' :
                        log.statusType === 'warning' ? 'bg-amber-100 text-amber-700' : 'bg-slate-100 text-slate-500'
                      }`}>
                        {log.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-6 flex justify-center">
            <button 
              onClick={() => setLogLimit(prev => prev + 50)}
              className="text-[10px] font-bold text-primary flex items-center gap-1 hover:underline"
            >
              LOAD MORE EVENTS
              <span className="material-symbols-outlined text-[12px]">keyboard_arrow_down</span>
            </button>
          </div>
        </motion.section>
      </motion.div>
    </div>
  );
};

export default MaintenanceLogs;
