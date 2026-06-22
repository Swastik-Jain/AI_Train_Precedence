import React, { useState, useEffect } from 'react';
import { KineticMap } from '../components/KineticMap/KineticMap';
import { AICopilotPanel } from '../components/AICopilotPanel/AICopilotPanel';
import { MareyTimeline } from '../components/MareyTimeline/MareyTimeline';
import { useMapStore } from '../store/useMapStore';

const GhatMonitor = () => {
  const trainStates = useMapStore(s => s.trainStates);
  const topology = useMapStore(s => s.topology);
  
  const getKm = (edgeId: string) => {
    if (!topology) return 0;
    const sourceNodeId = edgeId.split('-')[1];
    const node = topology.nodes.find(n => n.id === sourceNodeId);
    return node ? node.km : 0;
  };

  const isGhat = (id: string) => {
    const km = getKm(id);
    return km > 121 && km < 136;
  };
  const isKSR = (id: string) => {
    const km = getKm(id);
    return km >= 115 && km <= 121;
  };
  const isIGP = (id: string) => {
    const km = getKm(id);
    return km >= 136 && km <= 142;
  };

  const ghatTrains = trainStates.filter(t => t.edge_id && isGhat(t.edge_id));
  const activeInGhat = ghatTrains;
  const queuedAtKSR = trainStates.filter(t => t.status === 'Halted' && (t.direction === 'UP' || t.direction === 1) && t.edge_id && isKSR(t.edge_id));
  const queuedAtIGP = trainStates.filter(t => t.status === 'Halted' && (t.direction === 'DOWN' || t.direction === 2) && t.edge_id && isIGP(t.edge_id));

  return (
    <div className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-slate-100 flex-shrink-0 relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1 bg-amber-500"></div>
        <h3 className="text-[10px] uppercase font-bold tracking-widest text-slate-500 mb-4 flex items-center gap-2">
            <span className="material-symbols-outlined text-[14px]">warning</span>
            Kasara-Igatpuri Ghat Monitor
        </h3>
        
        <div className="flex justify-between items-center mb-6">
            <div className="text-center">
                <p className="text-xl font-black text-on-surface">{queuedAtKSR.length}</p>
                <p className="text-[9px] uppercase tracking-wide text-slate-400 font-bold">KSR Queue</p>
            </div>
            
            <div className="flex-1 px-4">
                <div className="relative h-2 bg-slate-100 rounded-full flex items-center justify-center">
                    <div className="absolute w-full h-[1px] bg-amber-200"></div>
                    <div className="bg-amber-100 border border-amber-300 text-amber-700 px-2 py-0.5 rounded text-[8px] font-bold z-10">
                        TOKEN: {activeInGhat.length > 0 ? ((activeInGhat[0].direction === 'UP' || activeInGhat[0].direction === 1) ? 'UP' : 'DOWN') : 'IDLE'}
                    </div>
                </div>
            </div>

            <div className="text-center">
                <p className="text-xl font-black text-on-surface">{queuedAtIGP.length}</p>
                <p className="text-[9px] uppercase tracking-wide text-slate-400 font-bold">IGP Queue</p>
            </div>
        </div>

        <div className="bg-surface-container p-3 rounded flex justify-between items-center">
            <p className="text-[10px] font-bold text-slate-500 uppercase">Active in block</p>
            <p className="text-xs font-bold text-amber-600 truncate max-w-[120px]">{activeInGhat.length > 0 ? activeInGhat.map(t => t.train_id).join(', ') : 'None (Clear)'}</p>
        </div>
    </div>
  );
};

const Dashboard: React.FC = () => {
  const [punctuality, setPunctuality] = useState(0);
  const [punctualityTrend, setPunctualityTrend] = useState(0);
  const [activeTrains, setActiveTrains] = useState(0);
  const [incomingTrains, setIncomingTrains] = useState(0);
  const [outgoingTrains, setOutgoingTrains] = useState(0);
  const [terminalTrains, setTerminalTrains] = useState(0);
  const [networkFluidity, setNetworkFluidity] = useState('-');
  const [haltedPct, setHaltedPct] = useState(0);
  const [scheduleReady, setScheduleReady] = useState(false);
  const [scheduleTrainCount, setScheduleTrainCount] = useState(0);
  const [isGeneratingSchedule, setIsGeneratingSchedule] = useState(false);
  const [aiLoad, setAiLoad] = useState(0);
  const [isInferenceActive, setIsInferenceActive] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [isLockdown, setIsLockdown] = useState(false);
  const [isBackendConnected, setIsBackendConnected] = useState(true);
  const [simSpeed, setSimSpeed] = useState(0.4);
  
  const { zoomLevel, setZoomLevel } = useMapStore();

  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const response = await fetch("http://localhost:8000/api/v1/telemetry");
        if (response.ok) {
          const data = await response.json();
          setPunctuality(prev => {
            const trend = data.punctuality - prev;
            // Only update trend if it's a meaningful change, otherwise keep old or 0
            if (prev !== 0) setPunctualityTrend(Number(trend.toFixed(1)));
            return data.punctuality;
          });
          setActiveTrains(data.active_trains);
          setIncomingTrains(data.incoming_trains ?? 0);
          setOutgoingTrains(data.outgoing_trains ?? 0);
          setTerminalTrains(data.terminal_trains);
          setNetworkFluidity(data.network_fluidity);
          setHaltedPct(data.halted_pct ?? 0);
          setScheduleReady(data.schedule_ready ?? false);
          setScheduleTrainCount(data.schedule_train_count ?? 0);
          setIsBackendConnected(true);
          setAiLoad(data.ai_load);
          setIsLockdown(data.lockdown);
        }
      } catch (error) {
        console.error("Failed to fetch telemetry:", error);
      }
    };

    const fetchStatus = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/v1/system/inference-status");
        if (res.ok) {
          const data = await res.json();
          setIsInferenceActive(data.active);
        }
      } catch (e) {
        console.error("Failed to fetch system status:", e);
      }
    };

    fetchTelemetry();
    fetchStatus();
    const interval = setInterval(() => {
      fetchTelemetry();
      fetchStatus();
    }, 2500);

    return () => clearInterval(interval);
  }, []);
  return (
    <div className="p-8 pb-32">
      {/* KPI Header */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        {/* Punctuality */}
        <div className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-slate-100">
          <p className="text-slate-400 font-label text-[10px] uppercase tracking-widest font-bold mb-2">Punctuality</p>
          <div className="flex items-end justify-between">
            <h3 className="font-headline text-3xl font-bold text-on-surface">{punctuality}%</h3>
            <span className={`px-2 py-1 rounded-full text-[10px] font-bold ${punctualityTrend >= 0 ? 'text-emerald-600 bg-emerald-50' : 'text-red-600 bg-red-50'}`}>
              {punctualityTrend >= 0 ? '+' : ''}{punctualityTrend}%
            </span>
          </div>
          <div className="mt-4 h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
            <div className="h-full bg-[#8B5CF6] rounded-full transition-all duration-1000 ease-in-out" style={{ width: `${punctuality}%` }}></div>
          </div>
        </div>
        {/* Active Trains */}
        <div className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-slate-100 flex flex-col justify-between">
          <div>
            <p className="text-slate-400 font-label text-[10px] uppercase tracking-widest font-bold mb-2">Active Trains</p>
            <div className="flex items-end justify-between">
              <h3 className="font-headline text-3xl font-bold text-on-surface">{activeTrains}</h3>
              <span className="material-symbols-outlined text-[#8B5CF6]" style={{ fontVariationSettings: "'FILL' 1" }}>train</span>
            </div>
            <div className="flex gap-3 mt-3">
              <div className="flex flex-col">
                <span className="text-[10px] font-bold text-slate-500 uppercase">Incoming (UP)</span>
                <span className="text-sm font-black text-emerald-600">{incomingTrains}</span>
              </div>
              <div className="w-px bg-slate-200"></div>
              <div className="flex flex-col">
                <span className="text-[10px] font-bold text-slate-500 uppercase">Outgoing (DOWN)</span>
                <span className="text-sm font-black text-amber-600">{outgoingTrains}</span>
              </div>
            </div>
          </div>
          <p className="text-[10px] text-slate-400 mt-4 pt-3 border-t border-slate-100">{terminalTrains} entering terminal station</p>
        </div>
        {/* OR-Engine Status */}
        <div className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-slate-100 flex flex-col justify-between">
          <div>
            <p className="text-slate-400 font-label text-[10px] uppercase tracking-widest font-bold mb-2">Network Fluidity</p>
            <div className="flex items-center gap-3 mt-1">
              <span className={`material-symbols-outlined text-[32px] ${
                networkFluidity === 'Nominal' ? 'text-emerald-500' :
                networkFluidity === 'Warning' ? 'text-amber-500' :
                networkFluidity === 'Degraded' ? 'text-red-500 animate-pulse' : 'text-slate-300'
              }`}>
                {networkFluidity === 'Degraded' ? 'car_crash' : networkFluidity === 'Warning' ? 'warning' : 'alt_route'}
              </span>
              <h3 className={`font-headline text-3xl font-bold leading-none ${
                networkFluidity === 'Nominal' ? 'text-emerald-600' :
                networkFluidity === 'Warning' ? 'text-amber-500' :
                networkFluidity === 'Degraded' ? 'text-red-600' : 'text-slate-400'
              }`}>{networkFluidity === '-' ? '—' : networkFluidity}</h3>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100">
            <div className="flex justify-between items-center mb-1.5">
              <span className="text-[10px] text-slate-400 uppercase font-bold tracking-wider">Halted Fleet</span>
              <span className={`text-[10px] font-black ${
                haltedPct > 20 ? 'text-red-600' : haltedPct >= 10 ? 'text-amber-500' : 'text-emerald-600'
              }`}>{haltedPct.toFixed(1)}%</span>
            </div>
            <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
              <div className={`h-full rounded-full transition-all duration-700 ${
                haltedPct > 20 ? 'bg-red-500' : haltedPct >= 10 ? 'bg-amber-400' : 'bg-emerald-500'
              }`} style={{ width: `${Math.min(haltedPct, 100)}%` }} />
            </div>
          </div>
        </div>
        <GhatMonitor />
      </div>

      {/* Middle Section: Map & Co-Pilot */}
      <div className="grid gap-8 mb-8" style={{ gridTemplateColumns: '7fr 2fr', height: 'calc(94vh - 235px)' }}>
        {/* Live Track Feed (Left) */}
        <div className="bg-surface-container-lowest rounded-lg shadow-sm overflow-hidden relative border border-slate-100 h-full min-h-0">

          {/* Live Kinetic Map Component */}
          <div className="absolute inset-0" style={{ zIndex: 0 }}>
            <KineticMap />
          </div>
          <div className="absolute top-2 right-4 z-10 flex flex-row gap-2 pointer-events-auto">
            <button 
              onClick={() => setZoomLevel(prev => Math.max(prev - 0.2, 0.4))}
              className="w-7 h-7 bg-slate-800/90 backdrop-blur-sm rounded-full flex items-center justify-center shadow-md border border-slate-700 text-slate-300 hover:bg-slate-700 hover:text-white transition-colors">
              <span className="material-symbols-outlined text-xs font-bold">remove</span>
            </button>
            <button 
              onClick={() => setZoomLevel(prev => Math.min(prev + 0.2, 3.0))}
              className="w-7 h-7 bg-slate-800/90 backdrop-blur-sm rounded-full flex items-center justify-center shadow-md border border-slate-700 text-slate-300 hover:bg-slate-700 hover:text-white transition-colors">
              <span className="material-symbols-outlined text-xs font-bold">add</span>
            </button>
          </div>
        </div>

        {/* AI Co-Pilot Panel (Right) — ORBIT Live Feed */}
        <div className="flex flex-col gap-6 h-full min-h-0">
          <div className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-slate-100 flex-1 flex flex-col overflow-hidden min-h-0">
            <AICopilotPanel />
          </div>
          {/* Simulation Controls */}
          <div className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-slate-100 flex flex-col justify-center gap-4">
            <p className="text-slate-400 font-label text-[10px] uppercase tracking-widest font-bold mb-1">Simulation Engine</p>
            
            {/* Step 1: OR-Tools Base Schedule */}
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-500">Base Schedule</span>
                <div className="flex items-center gap-1.5">
                  <span className={`w-1.5 h-1.5 rounded-full ${scheduleReady ? 'bg-emerald-500' : 'bg-slate-300'}`} />
                  <span className="text-[10px] text-slate-400 font-semibold">{scheduleReady ? `${scheduleTrainCount} trains mapped` : 'Idle'}</span>
                </div>
              </div>
              <button
                disabled={isGeneratingSchedule || isInferenceActive}
                onClick={async () => {
                  setIsGeneratingSchedule(true);
                  try {
                    const res = await fetch('http://localhost:8000/api/v1/fleet/generate-schedule', { method: 'POST' });
                    const data = await res.json();
                    if (data.status === 'optimal') {
                      setScheduleReady(true);
                      setScheduleTrainCount(data.fleet_size ?? 0);
                    } else {
                      alert(`OR-Tools: ${data.message ?? 'Infeasible schedule'}`);
                    }
                  } catch (e) { console.error(e); }
                  finally { setIsGeneratingSchedule(false); }
                }}
                className="w-full py-2.5 bg-slate-100 hover:bg-slate-200 disabled:opacity-50 text-slate-700 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-2"
              >
                {isGeneratingSchedule ? (
                  <><div className="w-4 h-4 border-2 border-slate-400 border-t-slate-700 rounded-full animate-spin" />Solving Math…</>
                ) : (
                  <><span className="material-symbols-outlined text-[16px]">polyline</span>Generate Route Paths</>
                )}
              </button>
            </div>

            <div className="w-full h-px bg-slate-100"></div>

            {/* Step 2: RL Model Inference */}
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-500">AI Co-Pilot</span>
                {isInferenceActive && (
                  <span className="text-[10px] text-emerald-500 font-bold animate-pulse">● Running</span>
                )}
              </div>
              <button 
                disabled={isStarting || (!scheduleReady && !isInferenceActive)}
                onClick={async () => {
                  setIsStarting(true);
                  try {
                    const endpoint = isInferenceActive ? "stop-inference" : "start-inference";
                    const response = await fetch(`http://localhost:8000/api/v1/system/${endpoint}`, { method: 'POST' });
                    if (response.ok) {
                      const data = await response.json();
                      if (data.status === "error") {
                        alert(`⚠️ ${data.message}`);
                      } else {
                        setIsInferenceActive(data.active ?? false);
                      }
                    }
                  } catch (e) {
                    console.error(e);
                  } finally {
                    setIsStarting(false);
                  }
                }}
                className={`w-full py-2.5 ${
                  isInferenceActive 
                    ? 'bg-red-500 hover:bg-red-600' 
                    : 'bg-[#8B5CF6] hover:bg-[#7c3aed]'
                } text-white rounded-lg text-xs font-bold transition-all shadow-md flex items-center justify-center gap-2 disabled:opacity-50`}
              >
                {isStarting ? (
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                ) : (
                  <>
                    <span className="material-symbols-outlined text-[16px]">{isInferenceActive ? 'stop' : 'play_arrow'}</span>
                    {isInferenceActive ? 'Stop Inference' : 'Start Inference'}
                  </>
                )}
              </button>
              
              {/* Sim Speed Control */}
              <div className="flex items-center justify-between mt-2 p-2 bg-slate-50 rounded-lg border border-slate-100">
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">Sim Speed</span>
                <div className="flex gap-1">
                  {[0.3, 0.6, 1.0].map((speed) => (
                    <button
                      key={speed}
                      onClick={async () => {
                        setSimSpeed(speed);
                        try {
                          await fetch("http://localhost:8000/api/v1/system/sim-speed", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ factor: speed })
                          });
                        } catch (e) { console.error(e); }
                      }}
                      className={`px-2 py-1 rounded text-[10px] font-bold transition-all ${simSpeed === speed ? 'bg-[#8B5CF6] text-white shadow-sm' : 'bg-white text-slate-500 border border-slate-200 hover:bg-slate-100'}`}
                    >
                      {speed}x
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Section: Marey Schedule Timeline — Live from ORBIT */}
      <MareyTimeline />

    </div>
  );
};

export default Dashboard;
