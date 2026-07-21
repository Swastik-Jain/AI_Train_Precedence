import React, { useState, useEffect } from 'react';
import { KineticMap } from '../components/KineticMap/KineticMap';
import { AICopilotPanel } from '../components/AICopilotPanel/AICopilotPanel';
import { MareyTimeline } from '../components/MareyTimeline/MareyTimeline';
import { useMapStore } from '../store/useMapStore';
import { useCopilotStore } from '../store/useCopilotStore';
import { apiUrl } from '../lib/api';

const GhatMonitor = () => {
  const trainStates = useMapStore(s => s.trainStates);
  const tokenTrains = useMapStore(s => s.tokenTrains);
  const ghatQueue = useMapStore(s => s.ghatQueue);
  
  const isGhat = (t: any) => {
    // Bulletproof check: If the train is visually on the Igatpuri station segments (49, 50, platforms),
    // it is DEFINITELY out of the ghat block.
    if (t.edge_id) {
      const parts = t.edge_id.split('-');
      if (parts.length >= 3) {
        const nodes = [parts[1], parts[2]];
        if (nodes.some(n => ['49', '50', '1035', '1036', '1037', '1038'].includes(n))) {
          return false;
        }
      }
    }
    if (!t.train_id || !tokenTrains) return false;
    return tokenTrains.includes(t.train_id);
  };

  const ghatTrains = trainStates.filter(t => isGhat(t));
  const activeInGhat = ghatTrains;


  return (
    <div className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-slate-100 flex-shrink-0 relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1 bg-amber-500"></div>
        <h3 className="text-[10px] uppercase font-bold tracking-widest text-slate-500 mb-4 flex items-center gap-2">
            <span className="material-symbols-outlined text-[14px]">warning</span>
            Kasara-Igatpuri Ghat Monitor
        </h3>
        
        <div className="flex justify-between items-center mb-6">
            <div className="text-center">
                <p className="text-xl font-black text-on-surface">{ghatQueue.ksr.count}</p>
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
                <p className="text-xl font-black text-on-surface">{ghatQueue.igp.count}</p>
                <p className="text-[9px] uppercase tracking-wide text-slate-400 font-bold">IGP Queue</p>
            </div>
        </div>

            <div className="flex justify-between items-center bg-amber-50 rounded px-2 py-1.5 border border-amber-100">
              <span className="text-xs font-bold text-slate-500">Inside Token Block</span>
              <p className="text-xs font-bold text-amber-600 break-words text-right">{activeInGhat.length > 0 ? activeInGhat.map(t => t.train_id).join(', ') : 'None (Clear)'}</p>
            </div>
    </div>
  );
};

const Dashboard: React.FC = () => {
  const [punctuality, setPunctuality] = useState(0);
  const [punctualityTrend, setPunctualityTrend] = useState<number | null>(null);
  const [activeTrains, setActiveTrains] = useState(0);
  const [incomingTrains, setIncomingTrains] = useState(0);
  const [outgoingTrains, setOutgoingTrains] = useState(0);
  const [terminalTrains, setTerminalTrains] = useState(0);
  const [scheduleReady, setScheduleReady] = useState(false);
  const [scheduleTrainCount, setScheduleTrainCount] = useState(0);
  const [isGeneratingSchedule, setIsGeneratingSchedule] = useState(false);
  const [isInferenceActive, setIsInferenceActive] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [isLoadingTelemetry, setIsLoadingTelemetry] = useState(true);
  const [simSpeed, setSimSpeed] = useState(0.4);
  
  const { setZoomLevel } = useMapStore();
  const { fetchBaseSchedule } = useCopilotStore();

  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const response = await fetch(apiUrl("/api/v1/telemetry"));
        if (response.ok) {
          const data = await response.json();
          setPunctuality(prev => {
            if (prev !== 0 && prev !== 100) {
                const trend = data.punctuality - prev;
                setPunctualityTrend(Number(trend.toFixed(1)));
            }
            return data.punctuality;
          });
          setActiveTrains(data.active_trains);
          setIncomingTrains(data.incoming_trains ?? 0);
          setOutgoingTrains(data.outgoing_trains ?? 0);
          setTerminalTrains(data.terminal_trains);
          setScheduleReady(data.schedule_ready ?? false);
          setScheduleTrainCount(data.schedule_train_count ?? 0);
        }
      } catch (error) {
        console.error("Failed to fetch telemetry:", error);
      } finally {
        setIsLoadingTelemetry(false);
      }
    };

    const fetchStatus = async () => {
      try {
        const res = await fetch(apiUrl("/api/v1/system/inference-status"));
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
    <section aria-label="Dashboard Content" className="p-8 pb-32">
      {/* KPI Header */}
      <section aria-label="Key Performance Indicators" className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        {/* Punctuality */}
        <div className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-slate-100">
          <p className="text-slate-400 font-label text-[10px] uppercase tracking-widest font-bold mb-2">Punctuality</p>
          {isLoadingTelemetry ? (
            <div className="flex flex-col gap-2"><div className="dash-skeleton dash-skeleton-text"></div><div className="dash-skeleton dash-skeleton-small"></div></div>
          ) : (
            <>
              <div className="flex items-end justify-between">
                <h3 className="font-headline text-3xl font-bold text-on-surface">{punctuality}%</h3>
                {punctualityTrend !== null && (
                  <span className={`px-2 py-1 rounded-full text-[10px] font-bold ${punctualityTrend >= 0 ? 'text-emerald-600 bg-emerald-50' : 'text-red-600 bg-red-50'}`}>
                    {punctualityTrend >= 0 ? '+' : ''}{punctualityTrend}%
                  </span>
                )}
              </div>
              <div className="mt-4 h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                <div className="h-full bg-[#8B5CF6] rounded-full transition-all duration-1000 ease-in-out" style={{ width: `${punctuality}%` }}></div>
              </div>
            </>
          )}
        </div>
        {/* Active Trains */}
        <div className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-slate-100 flex flex-col justify-between">
          <div>
            <p className="text-slate-400 font-label text-[10px] uppercase tracking-widest font-bold mb-2">Active Trains</p>
            {isLoadingTelemetry ? (
              <div className="flex flex-col gap-2"><div className="dash-skeleton dash-skeleton-text"></div><div className="dash-skeleton dash-skeleton-small"></div></div>
            ) : (
              <>
                <div className="flex items-end justify-between">
                  <h3 className="font-headline text-3xl font-bold text-on-surface">{activeTrains}</h3>
                  <span className="material-symbols-outlined text-[#8B5CF6] material-symbols-fill">train</span>
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
            </>
            )}
          </div>
          <p className="text-[10px] text-slate-400 mt-4 pt-3 border-t border-slate-100">{terminalTrains} entering terminal station</p>
        </div>
        <GhatMonitor />
        {/* Simulation Controls */}
        <div className="bg-surface-container-lowest p-3 rounded-lg shadow-sm border border-slate-100 flex flex-col justify-center gap-1.5">
            <p className="text-slate-400 font-label text-[10px] uppercase tracking-widest font-bold mb-1">Simulation Engine</p>
            
          {/* Step 1: OR-Tools Base Schedule */}
          <div className="flex flex-col gap-1">
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
                    const res = await fetch(apiUrl('/api/v1/fleet/generate-schedule'), { method: 'POST' });
                    const data = await res.json();
                    if (data.status === 'optimal') {
                      setScheduleReady(true);
                      setScheduleTrainCount(data.fleet_size ?? 0);
                      fetchBaseSchedule();
                    } else {
                      alert(`OR-Tools: ${data.message ?? 'Infeasible schedule'}`);
                    }
                  } catch (e) { console.error(e); }
                  finally { setIsGeneratingSchedule(false); }
                }}
                className="w-full py-1.5 bg-slate-100 hover:bg-slate-200 disabled:opacity-50 text-slate-700 rounded-lg text-[10px] font-bold transition-all flex items-center justify-center gap-2"
              >
                {isGeneratingSchedule ? (
                  <><div className="w-4 h-4 border-2 border-slate-400 border-t-slate-700 rounded-full animate-spin" />Solving Math…</>
                ) : (
                  <><span className="material-symbols-outlined text-[16px]">polyline</span>Generate Schedule</>
                )}
              </button>
            </div>

          {/* Step 2: RL Model Inference */}
          <div className="flex flex-col gap-1">
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
                    const response = await fetch(apiUrl(`/api/v1/system/${endpoint}`), { method: 'POST' });
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
                className={`w-full py-1.5 ${
                  isInferenceActive 
                    ? 'bg-red-500 hover:bg-red-600' 
                    : 'bg-[#8B5CF6] hover:bg-[#7c3aed]'
                } text-white rounded-lg text-[10px] font-bold transition-all shadow-md flex items-center justify-center gap-2 disabled:opacity-50`}
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
            <div className="flex items-center justify-between p-1 bg-slate-50 rounded-lg border border-slate-100">
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">Sim Speed</span>
              <div className="flex gap-1">
                  {[0.3, 0.6, 1.0].map((speed) => (
                    <button
                      key={speed}
                      onClick={async () => {
                        setSimSpeed(speed);
                        try {
                          await fetch(apiUrl("/api/v1/system/sim-speed"), {
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
      </section>

      {/* Middle Section: Map & Co-Pilot */}
      <section aria-label="Simulation Map and Copilot" className="grid gap-8 mb-8 dash-middle-section">
        {/* Live Track Feed (Left) */}
        <div className="bg-surface-container-lowest rounded-lg shadow-sm overflow-hidden relative border border-slate-100 h-full min-h-0">

          {/* Live Kinetic Map Component */}
          <div className="absolute inset-0 z-0">
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
        </div>
      </section>

      {/* Bottom Section: Marey Schedule Timeline — Live from ORBIT */}
      <section aria-label="Marey Schedule Timeline">
        <MareyTimeline />
      </section>

    </section>
  );
};

export default Dashboard;
