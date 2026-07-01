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
  latencies: Record<string, number>;
  forcedActions: Record<string, number>;
  impact: { reliability: string; congestion: string };
  adjustments: { id: number; type: string; desc: string; train_id?: string; edge_id?: string; constraint_type?: string; value?: number }[];
  projected_schedule?: Record<string, Record<string, {arrival: number, departure: number}>>;
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
  const { activeBlocks, fetchActiveBlocks, removeBlockRemote, openDrawer, impactReport, applyBlockRemote } = useMaintenanceStore();
  const { trainStates, allTrains, topology } = useMapStore();
  const blockList = Array.from(activeBlocks.values());

  // Simulate Tab State
  const [delayTrainId, setDelayTrainId] = useState(trainStates[0]?.train_id || '');
  const [latencies, setLatencies] = useState<Record<string, number>>({});
  const [selectedDuration, setSelectedDuration] = useState(0);
  const [isRecalculating, setIsRecalculating] = useState(false);
  const [recalculatedForTrain, setRecalculatedForTrain] = useState<string | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioResult[]>([]);
  const [scenarioLabel, setScenarioLabel] = useState('Scenario A');
  const [forcedActions, setForcedActions] = useState<Record<string, number>>({});

  // Audit Log State
  const [systemLogs, setSystemLogs] = useState<any[]>([]);
  const [logFilter, setLogFilter] = useState('All');
  const [logLimit, setLogLimit] = useState(50);
  
  // Infrastructure Blocks State
  const [selectedBlockEdge, setSelectedBlockEdge] = useState('');
  const [selectedBlockSeverity, setSelectedBlockSeverity] = useState<'TOTAL_BLOCK'|'SPEED_RESTRICTION'>('TOTAL_BLOCK');
  const edges = topology?.edges || [];

  const handleAddBlock = async () => {
    if (!selectedBlockEdge) return;
    const block = {
        element_id: selectedBlockEdge,
        type: 'TRACK_SEGMENT',
        severity: selectedBlockSeverity,
        start_time: new Date().toISOString(),
        end_time: new Date(Date.now() + 2 * 3600 * 1000).toISOString(),
        reason: 'What-If Simulation Block',
        isWhatIf: true,
    };
    await applyBlockRemote(block as any);
    setSelectedBlockEdge('');
  };

  // Maintenance Form State
  const getLocalISOString = (date: Date): string => {
    const offset = date.getTimezoneOffset() * 60000;
    return new Date(date.getTime() - offset).toISOString().slice(0, 16);
  };
  const [maintEdge, setMaintEdge] = useState('');
  const [maintSeverity, setMaintSeverity] = useState<'TOTAL_BLOCK'|'SPEED_RESTRICTION'>('TOTAL_BLOCK');
  const [maintStartTime, setMaintStartTime] = useState(getLocalISOString(new Date()));
  const [maintEndTime, setMaintEndTime] = useState(getLocalISOString(new Date(Date.now() + 2 * 3600 * 1000)));
  const [maintReason, setMaintReason] = useState('');
  const [maintSubmitting, setMaintSubmitting] = useState(false);

  const handleConfirmMaintenance = async () => {
      if (!maintEdge) return;
      setMaintSubmitting(true);
      const block = {
          element_id: maintEdge,
          type: 'TRACK_SEGMENT',
          severity: maintSeverity,
          start_time: new Date(maintStartTime).toISOString(),
          end_time: new Date(maintEndTime).toISOString(),
          reason: maintReason,
          isWhatIf: false,
      };
      await applyBlockRemote(block as any);
      setMaintSubmitting(false);
      setMaintEdge('');
      setMaintReason('');
  };

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
        const res = await fetch('/api/v1/simulation/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                label: scenarioLabel,
                latencies: latencies,
                forced_actions: forcedActions,
            })
        });
        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();
        
        setScenarios(prev => {
            const updated = [
                ...prev,
                {
                    id: `${Date.now()}`,
                    label: data.label || scenarioLabel,
                    latencies,
                    forcedActions,
                    impact: { reliability: data.impact.reliability, congestion: data.impact.congestion },
                    adjustments: data.adjustments || [],
                },
            ];
            setScenarioLabel(`Scenario ${String.fromCharCode(65 + updated.length)}`);
            return updated;
        });
        setRecalculatedForTrain(delayTrainId);
    } catch (err) {
        console.error('[Sandbox] Analysis failed:', err);
    } finally {
        setIsRecalculating(false);
    }
  };

  const handleDeploy = async () => {
    try {
        // Only deploy real maintenance blocks — what-if blocks must never be
        // promoted to live inference. The backend also enforces this, but
        // filtering here makes the intent explicit.
        const realBlocks = blockList.filter(b => !b.isWhatIf);
        const payload = {
            blocks: realBlocks,
            forced_actions: forcedActions,
            latencies: latencies
        };
        const res = await fetch('/api/v1/simulation/deploy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (res.ok) {
            alert('Simulation Sandbox Deployed Successfully!');
            fetchActiveBlocks();
            setScenarios([]);
            setLatencies({});
            setForcedActions({});
            setScenarioLabel('Scenario A');
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

  // KPI Logic
  const activeBlocksCount = blockList.filter(b => {
    const nowTime = new Date().getTime();
    const st = new Date(b.start_time).getTime();
    const et = new Date(b.end_time).getTime();
    return st <= nowTime && et >= nowTime;
  }).length;
  
  const scheduledBlocksCount = blockList.filter(b => {
    const nowTime = new Date().getTime();
    const st = new Date(b.start_time).getTime();
    return st > nowTime;
  }).length;
  
  const totalEdges = topology?.edges?.length || 25;
  const sectionsClear = totalEdges - activeBlocksCount;

  return (
    <div className="p-8 max-w-[1600px] mx-auto w-full space-y-6">
      
      {/* ── Top Shared Map Section ── */}
      <section className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-outline-variant/10 relative flex flex-col h-[560px]">
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

      {/* ================= SIMULATE TAB ================= */}
      {activeTab === 'simulate' && (
          <div className="grid grid-cols-12 gap-8">
            {/* Left Pane: Scenario Builder */}
            <div className="col-span-12 lg:col-span-4 flex flex-col">
                <section className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-outline-variant/10 flex-1">
                    <div className="flex items-center gap-2 mb-8 text-on-surface">
                        <span className="material-symbols-outlined text-violet-500">settings_input_component</span>
                        <h3 className="text-lg font-bold">Scenario builder</h3>
                    </div>

                    <div className="space-y-8">
                        {/* Primary Delay Node */}
                        <div>
                            <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-500 mb-2">Primary Delay Node</label>
                            <div className="relative">
                                <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 z-10 text-[18px]">train</span>
                                <select 
                                    className="w-full pl-10 pr-10 py-2.5 bg-surface-container-low border border-outline-variant/20 rounded-md font-mono text-sm focus:ring-2 focus:ring-violet-500/30 text-on-surface appearance-none cursor-pointer hover:bg-surface-container-high transition-colors" 
                                    value={delayTrainId}
                                    onChange={e => setDelayTrainId(e.target.value)}
                                >
                                    <option value="">Select a train...</option>
                                    {allTrains.map(t => (
                                        <option key={t.train_id} value={t.train_id}>
                                            {t.train_id} {t.status === 'Scheduled' ? '(Upcoming)' : '(Running)'}
                                        </option>
                                    ))}
                                </select>
                                <span className="material-symbols-outlined absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none text-[18px]">expand_more</span>
                            </div>
                        </div>

                        {/* Latency Duration */}
                        <div>
                            <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-500 mb-2">Latency Duration — {selectedDuration} Min</label>
                            <input 
                                className="w-full h-1.5 bg-surface-container-high rounded-lg appearance-none cursor-pointer accent-violet-500 hover:accent-violet-400 transition-colors" 
                                type="range" min="0" max="60" 
                                value={selectedDuration} 
                                onChange={e => {
                                    const val = parseInt(e.target.value);
                                    setSelectedDuration(val);
                                    if (delayTrainId && val > 0) {
                                        setLatencies(prev => ({ ...prev, [delayTrainId]: val }));
                                    } else if (delayTrainId && val === 0) {
                                        setLatencies(prev => { const n = {...prev}; delete n[delayTrainId]; return n; });
                                    }
                                }}
                            />
                        </div>

                        {/* Force Action */}
                        <div className="pt-2 border-t border-outline-variant/10">
                            <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-500 mb-2 mt-4">Force Action (Optional)</label>
                            <div className="flex gap-2 items-center">
                                <div className="relative flex-1">
                                    <select
                                        className="w-full pl-3 pr-8 py-2 bg-surface-container-low border border-outline-variant/20 rounded-md font-mono text-sm text-on-surface appearance-none hover:bg-surface-container-high transition-colors cursor-pointer"
                                        value=""
                                        onChange={e => {
                                            const action = parseInt(e.target.value);
                                            if (delayTrainId) setForcedActions(prev => ({ ...prev, [delayTrainId]: action }));
                                        }}
                                    >
                                        <option value="" disabled>Select action...</option>
                                        <option value="0">HOLD</option>
                                        <option value="1">PROCEED (MAIN)</option>
                                        <option value="2">DIVERT</option>
                                    </select>
                                    <span className="material-symbols-outlined absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none text-[16px]">expand_more</span>
                                </div>
                                <button className="px-4 py-2 bg-surface-container border border-outline-variant/20 rounded-md text-sm font-bold hover:bg-surface-container-high transition-colors flex items-center gap-1 text-on-surface">
                                    <span className="material-symbols-outlined text-[16px]">add</span> Add
                                </button>
                            </div>
                            
                            {Object.keys(forcedActions).length > 0 && (
                                <div className="flex flex-col gap-2 mt-3">
                                    {Object.entries(forcedActions).map(([tid, act]) => (
                                        <div key={tid} className="text-xs bg-surface-container px-3 py-2 rounded-md flex items-center justify-between border border-outline-variant/10">
                                            <span><span className="font-mono font-bold text-violet-600">{tid}</span>: {['HOLD','MAIN','DIVERT'][act]}</span>
                                            <button onClick={() => setForcedActions(prev => { const n = {...prev}; delete n[tid]; return n; })} className="text-slate-400 hover:text-rose-500">
                                                <span className="material-symbols-outlined text-[16px]">close</span>
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Infrastructure Blocks */}
                        <div className="pt-2 border-t border-outline-variant/10">
                            <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-500 mb-2 mt-4">Infrastructure Blocks</label>
                            <label className="block text-[10px] font-bold uppercase text-slate-400 mb-2">Block a track segment</label>
                            
                            <div className="flex gap-2 items-center mb-3">
                                <div className="relative flex-[1.5]">
                                    <select
                                        className="w-full pl-3 pr-8 py-2 bg-surface-container-low border border-outline-variant/20 rounded-md font-mono text-sm text-on-surface appearance-none hover:bg-surface-container-high transition-colors cursor-pointer"
                                        value={selectedBlockEdge}
                                        onChange={(e) => setSelectedBlockEdge(e.target.value)}
                                    >
                                        <option value="">Segment...</option>
                                        {edges.map(e => (
                                            <option key={e.id} value={e.id}>{e.id}</option>
                                        ))}
                                    </select>
                                    <span className="material-symbols-outlined absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none text-[16px]">expand_more</span>
                                </div>
                                <div className="relative flex-1">
                                    <select
                                        className="w-full pl-3 pr-8 py-2 bg-surface-container-low border border-outline-variant/20 rounded-md text-sm text-on-surface appearance-none hover:bg-surface-container-high transition-colors cursor-pointer"
                                        value={selectedBlockSeverity}
                                        onChange={(e) => setSelectedBlockSeverity(e.target.value as any)}
                                    >
                                        <option value="TOTAL_BLOCK">Total block</option>
                                        <option value="SPEED_RESTRICTION">Speed restriction</option>
                                    </select>
                                    <span className="material-symbols-outlined absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none text-[16px]">expand_more</span>
                                </div>
                                <button 
                                    onClick={handleAddBlock}
                                    disabled={!selectedBlockEdge}
                                    className="px-3 py-2 bg-surface-container border border-outline-variant/20 rounded-md text-sm font-bold hover:bg-surface-container-high transition-colors flex items-center gap-1 disabled:opacity-50 text-on-surface"
                                >
                                    <span className="material-symbols-outlined text-[16px]">add</span> Add
                                </button>
                            </div>

                            {blockList.filter(b => b.isWhatIf).length === 0 ? (
                                <p className="text-xs text-on-surface-variant italic">No blocks added. These affect only this scenario.</p>
                            ) : (
                                <div className="flex flex-col gap-2">
                                    {blockList.filter(b => b.isWhatIf).map(blk => (
                                        <div key={blk.blockId} className="text-xs bg-surface-container px-3 py-2 rounded-md flex items-center justify-between border border-outline-variant/10">
                                            <div className="flex gap-2 items-center">
                                                <span className={blk.severity === 'TOTAL_BLOCK' ? 'text-rose-500 font-bold' : 'text-amber-500 font-bold'}>
                                                    {blk.severity === 'TOTAL_BLOCK' ? '🔴' : '🟡'}
                                                </span>
                                                <span className="font-mono font-bold text-on-surface">{blk.element_id}</span>
                                            </div>
                                            <button onClick={() => removeBlockRemote(blk.element_id)} className="text-slate-400 hover:text-rose-500">
                                                <span className="material-symbols-outlined text-[16px]">close</span>
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                        
                        {/* Scenario Label */}
                        <div className="pt-2 border-t border-outline-variant/10">
                            <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-500 mb-2 mt-4">Scenario Label</label>
                            <input 
                                type="text"
                                className="w-full px-3 py-2.5 bg-surface-container-low border border-outline-variant/20 rounded-md text-sm focus:ring-2 focus:ring-violet-500/30 text-on-surface"
                                value={scenarioLabel}
                                onChange={e => setScenarioLabel(e.target.value)}
                            />
                        </div>

                        {/* Recalculate Button */}
                        <div className="pt-4">
                            <button 
                                onClick={handleRecalculate} disabled={isRecalculating}
                                className="w-full bg-primary text-on-primary py-3.5 rounded-md font-bold flex items-center justify-center gap-2 hover:opacity-90 transition-opacity disabled:opacity-50 shadow-sm">
                                <span className={`material-symbols-outlined ${isRecalculating ? 'animate-spin' : ''}`}>sync</span>
                                {isRecalculating ? 'Recalculating network...' : 'Recalculate network'}
                            </button>
                        </div>
                    </div>
                </section>
            </div>

            {/* Right Pane: Visuals & Comparison */}
            <div className="col-span-12 lg:col-span-8 flex flex-col gap-6">
                
                {/* Marey Topology */}
                <section className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-outline-variant/10">
                    <div className="flex justify-between items-center mb-6">
                        <h3 className="text-lg font-bold text-on-surface">Marey space-time topology</h3>
                        <div className="flex gap-4">
                            <span className="flex items-center gap-1.5 text-xs font-bold text-slate-500"><span className="w-3 h-3 rounded-full bg-[#8B5CF6]"></span> Projected</span>
                            <span className="flex items-center gap-1.5 text-xs font-bold text-slate-500"><span className="w-3 h-3 rounded-full bg-slate-300"></span> Historical</span>
                            <span className="flex items-center gap-1.5 text-xs font-bold text-slate-500"><span className="w-3 h-2 rounded-sm bg-amber-500"></span> Block</span>
                        </div>
                    </div>
                    <div className="relative bg-surface-container-low rounded-xl overflow-hidden p-4 [&_.marey-container]:!p-0 [&_.marey-canvas-wrapper]:!mb-0 [&_.marey-container]:border-0 [&_.marey-container]:shadow-none">
                        <div className="absolute inset-0 opacity-20 pointer-events-none" style={{ backgroundImage: "radial-gradient(#8B5CF6 0.5px, transparent 0.5px)", backgroundSize: "24px 24px" }}></div>
                        <MareyTimeline scenarios={scenarios} hideHeader={true} hideTelemetry={true} />
                    </div>
                </section>

                {/* Scenario Comparison */}
                <section className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-outline-variant/10 flex-1 flex flex-col min-h-[200px]">
                    <div className="flex items-center gap-2 mb-6">
                        <span className="material-symbols-outlined text-violet-500">splitscreen</span>
                        <h3 className="text-lg font-bold text-on-surface">Scenario comparison</h3>
                    </div>

                    {scenarios.length === 0 ? (
                        <div className="flex-1 flex items-center justify-center">
                            <p className="text-sm text-on-surface-variant font-medium">Run a scenario to see results here.</p>
                        </div>
                    ) : (
                        <div className="flex-1 overflow-y-auto mb-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {scenarios.map(s => (
                                <div key={s.id} className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/20 relative">
                                    <p className="text-sm font-bold text-violet-600 mb-3">{s.label}</p>
                                    <div className="grid grid-cols-2 gap-4 mb-2">
                                        <div>
                                            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wide">Reliability</p>
                                            <p className="text-xl font-extrabold text-error">{s.impact.reliability}</p>
                                        </div>
                                        <div>
                                            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wide">Congestion</p>
                                            <p className="text-xl font-extrabold text-on-surface-variant">{s.impact.congestion}</p>
                                        </div>
                                    </div>
                                    {(Object.keys(s.forcedActions).length > 0 || Object.keys(s.latencies).length > 0) && (
                                        <div className="mt-3 pt-3 border-t border-outline-variant/10 text-xs text-on-surface-variant space-y-1">
                                            {Object.keys(s.latencies).length > 0 && (
                                                <p>Latencies: {Object.entries(s.latencies).map(([t, l]) => `${t}(+${l}m)`).join(', ')}</p>
                                            )}
                                            {Object.keys(s.forcedActions).length > 0 && (
                                                <p>Forced: {Object.entries(s.forcedActions).map(([t, a]) => `${t}→${['HOLD','MAIN','DIVERT'][a]}`).join(', ')}</p>
                                            )}
                                        </div>
                                    )}
                                    <button onClick={() => setScenarios(prev => prev.filter(x => x.id !== s.id))} className="absolute top-3 right-3 text-slate-400 hover:text-rose-500">
                                        <span className="material-symbols-outlined text-[16px]">close</span>
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}

                    <div className="mt-auto pt-6 border-t border-outline-variant/10 flex justify-end">
                        <button 
                            onClick={handleDeploy}
                            disabled={scenarios.length === 0}
                            className="bg-surface-container border border-outline-variant/20 text-on-surface px-5 py-2.5 rounded-md text-sm font-bold shadow-sm hover:bg-surface-container-high transition-colors flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                           <span className="material-symbols-outlined text-[18px]">rocket_launch</span>
                           Deploy to live network
                        </button>
                    </div>
                </section>
            </div>
          </div>
        )}


        {/* ================= MAINTENANCE TAB ================= */}
        {activeTab === 'maintenance' && (
          <motion.div className="flex flex-col gap-6" variants={fadeUp as any}>
            
            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-surface-container-lowest p-6 rounded-lg border border-outline-variant/10 shadow-sm flex flex-col">
                    <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Active Now</p>
                    <p className="text-4xl font-light text-rose-500 mb-2">{activeBlocksCount}</p>
                    <p className="text-xs text-on-surface-variant font-medium">affecting live inference</p>
                </div>
                <div className="bg-surface-container-lowest p-6 rounded-lg border border-outline-variant/10 shadow-sm flex flex-col">
                    <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Scheduled</p>
                    <p className="text-4xl font-light text-amber-500 mb-2">{scheduledBlocksCount}</p>
                    <p className="text-xs text-on-surface-variant font-medium">upcoming windows</p>
                </div>
                <div className="bg-surface-container-lowest p-6 rounded-lg border border-outline-variant/10 shadow-sm flex flex-col">
                    <p className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-1">Sections Clear</p>
                    <p className="text-4xl font-light text-emerald-500 mb-2">{sectionsClear}</p>
                    <p className="text-xs text-on-surface-variant font-medium">of {totalEdges} total edges</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                
                {/* Left Pane: Schedule Form */}
                <div className="lg:col-span-4 bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-outline-variant/10 flex flex-col h-fit">
                    <div className="flex items-center gap-2 mb-6">
                        <span className="material-symbols-outlined text-violet-500">calendar_month</span>
                        <h3 className="text-lg font-bold text-on-surface">Schedule maintenance window</h3>
                    </div>

                    <div className="space-y-5">
                        <div>
                            <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-500 mb-2">Track Segment</label>
                            <select 
                                className="w-full px-3 py-2.5 bg-surface-container border border-outline-variant/20 rounded-md text-sm font-medium focus:ring-2 focus:ring-violet-500/30 text-on-surface"
                                value={maintEdge}
                                onChange={e => setMaintEdge(e.target.value)}
                            >
                                <option value="" disabled>Select an edge...</option>
                                {edges.map(e => <option key={e.id} value={e.id}>{e.id}</option>)}
                            </select>
                        </div>

                        <div>
                            <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-500 mb-2">Severity</label>
                            <div className="grid grid-cols-2 gap-2">
                                <button 
                                    className={`py-3 px-2 rounded border flex flex-col items-center justify-center gap-1 transition-colors ${maintSeverity === 'TOTAL_BLOCK' ? 'bg-red-600 border-red-600 text-white shadow-sm' : 'bg-red-50 border-red-200 text-red-600 hover:bg-red-100'}`}
                                    onClick={() => setMaintSeverity('TOTAL_BLOCK')}
                                >
                                    <span className="material-symbols-outlined text-[20px]">block</span>
                                    <span className="text-[13px] font-bold">Total block</span>
                                </button>
                                <button 
                                    className={`py-3 px-2 rounded border flex flex-col items-center justify-center gap-1 transition-colors ${maintSeverity === 'SPEED_RESTRICTION' ? 'bg-yellow-500 border-yellow-500 text-white shadow-sm' : 'bg-yellow-50 border-yellow-200 text-yellow-700 hover:bg-yellow-100'}`}
                                    onClick={() => setMaintSeverity('SPEED_RESTRICTION')}
                                >
                                    <span className="material-symbols-outlined text-[20px]">speed</span>
                                    <span className="text-[13px] font-bold">Speed restriction</span>
                                </button>
                            </div>
                            <p className="text-[11px] text-on-surface-variant mt-2 font-medium">
                                {maintSeverity === 'TOTAL_BLOCK' ? 'Segment fully closed — trains will be rerouted or held.' : 'Segment traversable at reduced speed.'}
                            </p>
                        </div>

                        <div>
                            <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-500 mb-2">Maintenance Window</label>
                            <div className="flex flex-col gap-2">
                                <input 
                                    type="datetime-local" 
                                    className="w-full px-3 py-2 bg-surface-container border border-outline-variant/20 rounded-md text-xs font-mono text-on-surface"
                                    value={maintStartTime}
                                    onChange={e => setMaintStartTime(e.target.value)}
                                />
                                <div className="flex justify-center -my-1">
                                    <span className="material-symbols-outlined text-[16px] text-slate-400 rotate-90">arrow_forward</span>
                                </div>
                                <input 
                                    type="datetime-local" 
                                    className="w-full px-3 py-2 bg-surface-container border border-outline-variant/20 rounded-md text-xs font-mono text-on-surface"
                                    value={maintEndTime}
                                    onChange={e => setMaintEndTime(e.target.value)}
                                />
                            </div>
                        </div>

                        <div>
                            <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-500 mb-2">Reason / Work Order</label>
                            <textarea 
                                className="w-full px-3 py-2.5 bg-surface-container border border-outline-variant/20 rounded-md text-sm focus:ring-2 focus:ring-violet-500/30 text-on-surface resize-none"
                                rows={3}
                                placeholder="e.g. Track relay replacement — WO-2026-0419"
                                value={maintReason}
                                onChange={e => setMaintReason(e.target.value)}
                            />
                        </div>

                        <div className="flex gap-3 pt-2">
                            <button 
                                className="flex-1 py-2.5 bg-surface-container border border-outline-variant/20 rounded-md text-sm font-bold text-on-surface-variant hover:bg-surface-container-high transition-colors"
                                onClick={() => { setMaintEdge(''); setMaintReason(''); }}
                                disabled={maintSubmitting}
                            >
                                Cancel
                            </button>
                            <button 
                                className="flex-[2] py-2.5 bg-primary text-on-primary rounded-md text-sm font-bold hover:opacity-90 transition-opacity shadow-sm flex items-center justify-center gap-2 disabled:opacity-50"
                                onClick={handleConfirmMaintenance}
                                disabled={maintSubmitting || !maintEdge}
                            >
                                <span className={`material-symbols-outlined text-[18px] ${maintSubmitting ? 'animate-spin' : ''}`}>
                                    {maintSubmitting ? 'sync' : 'event_available'}
                                </span>
                                {maintSubmitting ? 'Scheduling...' : 'Confirm window'}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Right Pane: Charts and Lists */}
                <div className="lg:col-span-8 flex flex-col gap-6">
                    
                    {/* Corridor View */}
                    <div className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-outline-variant/10">
                        <div className="flex justify-between items-center mb-6">
                            <div className="flex items-center gap-2">
                                <span className="material-symbols-outlined text-violet-500">timeline</span>
                                <h3 className="text-lg font-bold text-on-surface">24-hour corridor view</h3>
                            </div>
                            <div className="flex gap-4">
                                <span className="flex items-center gap-1.5 text-xs font-bold text-slate-500"><span className="w-3 h-3 rounded bg-rose-400 border border-rose-500"></span> Active block</span>
                                <span className="flex items-center gap-1.5 text-xs font-bold text-slate-500"><span className="w-3 h-3 rounded bg-amber-100 border border-amber-300"></span> Scheduled</span>
                            </div>
                        </div>

                        <div className="relative pt-4 pb-8">
                            {/* X-axis lines */}
                            <div className="absolute inset-0 flex flex-col justify-between pointer-events-none pb-8" style={{marginLeft: '120px'}}>
                                {[0,1,2,3,4,5].map(i => (
                                    <div key={i} className="w-full border-b border-outline-variant/10 h-10"></div>
                                ))}
                            </div>

                            {/* Blocks */}
                            <div className="relative z-10 flex flex-col gap-4 mt-2">
                                {Array.from(new Set(blockList.map(b => b.element_id))).slice(0,5).map((edgeId, idx) => (
                                    <div key={edgeId} className="flex items-center h-6">
                                        <div className="w-[120px] text-[11px] font-bold text-on-surface-variant truncate pr-4 text-right">
                                            {edgeId}
                                        </div>
                                        <div className="flex-1 relative h-full bg-surface-container-low rounded">
                                            {blockList.filter(b => b.element_id === edgeId).map((block, bIdx) => {
                                                const d = new Date();
                                                const startOfDay = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
                                                const endOfDay = startOfDay + 24 * 3600 * 1000;
                                                const bStart = new Date(block.start_time).getTime();
                                                const bEnd = new Date(block.end_time).getTime();
                                                
                                                if (bEnd < startOfDay || bStart > endOfDay) return null;
                                                
                                                const effStart = Math.max(startOfDay, bStart);
                                                const effEnd = Math.min(endOfDay, bEnd);
                                                
                                                const leftPct = ((effStart - startOfDay) / (24 * 3600 * 1000)) * 100;
                                                const widthPct = ((effEnd - effStart) / (24 * 3600 * 1000)) * 100;
                                                const isActive = bStart <= Date.now() && bEnd >= Date.now();

                                                return (
                                                    <div 
                                                        key={bIdx}
                                                        className={`absolute top-0 bottom-0 rounded opacity-90 flex items-center justify-center overflow-hidden border ${isActive ? 'bg-rose-100 border-rose-300 text-rose-700' : 'bg-amber-100 border-amber-300 text-amber-800'}`}
                                                        style={{ left: `${leftPct}%`, width: `${Math.max(2, widthPct)}%` }}
                                                    >
                                                        {widthPct > 8 && (
                                                            <span className="text-[8px] font-bold whitespace-nowrap px-1">
                                                                {new Date(effStart).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})} - {new Date(effEnd).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                                                            </span>
                                                        )}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>
                                ))}
                                {blockList.length === 0 && (
                                    <div className="text-center text-sm text-on-surface-variant italic py-8">No active or scheduled blocks.</div>
                                )}
                            </div>

                            {/* X-axis labels */}
                            <div className="absolute bottom-0 left-[120px] right-0 flex justify-between text-[10px] font-bold text-slate-400">
                                <span>00:00</span>
                                <span>06:00</span>
                                <span>12:00</span>
                                <span>18:00</span>
                                <span>24:00</span>
                            </div>
                            
                            {/* Current Time Indicator */}
                            <div className="absolute top-0 bottom-6 w-px bg-violet-500 z-20 pointer-events-none" style={{
                                left: `calc(120px + ${((Date.now() - new Date().setHours(0,0,0,0)) / (24 * 3600 * 1000)) * 100}%)`
                            }}>
                                <div className="absolute -top-4 -translate-x-1/2 text-[9px] font-bold text-violet-600 bg-violet-100 px-1 py-0.5 rounded">NOW</div>
                            </div>
                        </div>
                    </div>

                    {/* Scheduled Windows List */}
                    <div className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-outline-variant/10 flex-1">
                        <div className="flex justify-between items-center mb-4">
                            <div className="flex items-center gap-2">
                                <span className="material-symbols-outlined text-violet-500">list_alt</span>
                                <h3 className="text-lg font-bold text-on-surface">Scheduled windows</h3>
                            </div>
                            <span className="text-xs font-bold text-slate-500">{blockList.length} windows</span>
                        </div>
                        
                        <div className="flex flex-col gap-3">
                            {blockList.length === 0 && (
                                <p className="text-sm text-on-surface-variant text-center py-4">No maintenance windows scheduled.</p>
                            )}
                            {blockList.map((block, idx) => {
                                const isActive = new Date(block.start_time).getTime() <= Date.now() && new Date(block.end_time).getTime() >= Date.now();
                                return (
                                    <div key={idx} className="bg-surface-container p-4 rounded-lg border border-outline-variant/10 flex items-center gap-4 group">
                                        <div className={`px-2 py-1 rounded text-[10px] font-bold whitespace-nowrap border ${isActive ? 'bg-rose-100 text-rose-700 border-rose-200' : 'bg-amber-100 text-amber-800 border-amber-200'}`}>
                                            {isActive ? 'Active now' : 'Scheduled'}
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <p className="text-sm font-bold text-on-surface truncate">
                                                {block.element_id}
                                            </p>
                                            <p className="text-[11px] text-on-surface-variant truncate mt-0.5 font-medium">
                                                {new Date(block.start_time).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})} → {new Date(block.end_time).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})} • {block.severity === 'TOTAL_BLOCK' ? 'Total block' : 'Speed restriction'} • {block.reason}
                                            </p>
                                        </div>
                                        <button 
                                            onClick={() => removeBlockRemote(block.element_id)}
                                            className="w-8 h-8 rounded border border-outline-variant/20 flex items-center justify-center text-slate-400 hover:text-rose-500 hover:border-rose-200 hover:bg-rose-50 transition-colors"
                                        >
                                            <span className="material-symbols-outlined text-[16px]">close</span>
                                        </button>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            </div>
          </motion.div>
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
                    if (logFilter === 'Manual Only') return log.operator !== 'System' && log.operator !== 'SYSTEM';
                    return true;
                  }).length === 0 ? (
                    <tr><td colSpan={5} className="py-6 text-center text-xs text-on-surface-variant">No system activity matches this filter.</td></tr>
                  ) : systemLogs.filter(log => {
                    if (logFilter === 'Critical Only') return log.statusType === 'error';
                    if (logFilter === 'Manual Only') return log.operator !== 'System' && log.operator !== 'SYSTEM';
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
    </div>
  );
};

export default ControlCentre;
