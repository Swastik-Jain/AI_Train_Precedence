import React, { useState, useEffect } from 'react';
import { KineticMap } from '../components/KineticMap/KineticMap';
import { AICopilotPanel } from '../components/AICopilotPanel/AICopilotPanel';
import { MareyTimeline } from '../components/MareyTimeline/MareyTimeline';

const Dashboard: React.FC = () => {
  const [punctuality, setPunctuality] = useState(98.4);
  const [punctualityTrend, setPunctualityTrend] = useState(0.4);
  const [activeTrains, setActiveTrains] = useState(142);
  const [terminalTrains, setTerminalTrains] = useState(12);
  const [systemHealth, setSystemHealth] = useState('Nominal');
  const [nodeResponseTime, setNodeResponseTime] = useState(12);
  const [aiLoad, setAiLoad] = useState(82);
  const [isInferenceActive, setIsInferenceActive] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [isLockdown, setIsLockdown] = useState(false);

  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const response = await fetch("http://localhost:8000/api/v1/telemetry");
        if (response.ok) {
          const data = await response.json();
          setPunctuality(prev => {
            const trend = data.punctuality - prev;
            // Only update trend if it's a meaningful change, otherwise keep old or 0
            if (prev !== 98.4) setPunctualityTrend(Number(trend.toFixed(1)));
            return data.punctuality;
          });
          setActiveTrains(data.active_trains);
          setTerminalTrains(data.terminal_trains);
          setSystemHealth(data.system_health);
          setNodeResponseTime(data.node_response_time);
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
        <div className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-slate-100">
          <p className="text-slate-400 font-label text-[10px] uppercase tracking-widest font-bold mb-2">Active Trains</p>
          <div className="flex items-end justify-between">
            <h3 className="font-headline text-3xl font-bold text-on-surface">{activeTrains}</h3>
            <span className="material-symbols-outlined text-[#8B5CF6]" style={{ fontVariationSettings: "'FILL' 1" }}>train</span>
          </div>
          <p className="text-[10px] text-slate-400 mt-2">{terminalTrains} entering terminal station</p>
        </div>
        {/* System Health */}
        <div className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-slate-100">
          <p className="text-slate-400 font-label text-[10px] uppercase tracking-widest font-bold mb-2">System Health</p>
          <div className="flex items-end justify-between">
            <h3 className={`font-headline text-3xl font-bold ${systemHealth === 'Nominal' ? 'text-emerald-600' : 'text-amber-500'}`}>{systemHealth}</h3>
            <div className="flex gap-0.5">
              <div className={`w-1 h-4 ${systemHealth === 'Nominal' ? 'bg-emerald-500' : 'bg-amber-500'} rounded-full animate-pulse transition-colors duration-500`}></div>
              <div className={`w-1 h-6 ${systemHealth === 'Nominal' ? 'bg-emerald-500' : 'bg-amber-500'} rounded-full transition-colors duration-500`}></div>
              <div className={`w-1 h-3 ${systemHealth === 'Nominal' ? 'bg-emerald-500' : 'bg-amber-500'} rounded-full transition-colors duration-500`}></div>
            </div>
          </div>
          <p className="text-[10px] text-slate-400 mt-2">All nodes responding &lt; {nodeResponseTime}ms</p>
        </div>
        <div className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-slate-100 flex flex-col justify-center">
          <p className="text-slate-400 font-label text-[10px] uppercase tracking-widest font-bold mb-2">Model Inference</p>
          <button 
            disabled={isStarting}
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
            className={`w-full py-3 ${isInferenceActive ? 'bg-red-500 hover:bg-red-600' : 'bg-[#8B5CF6] hover:bg-[#7c3aed]'} text-white rounded-lg font-bold transition-all shadow-md flex items-center justify-center gap-2 disabled:opacity-50`}
          >
            {isStarting ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
            ) : (
              <>
                <span className="material-symbols-outlined">{isInferenceActive ? 'stop' : 'play_arrow'}</span>
                {isInferenceActive ? 'Stop Inference' : 'Start Inference'}
              </>
            )}
          </button>
          {isInferenceActive && (
            <p className="text-[10px] text-emerald-500 mt-2 text-center font-semibold animate-pulse">
              ● RL model running — AI Co-Pilot active
            </p>
          )}
          {!isInferenceActive && (
            <p className="text-[10px] text-slate-400 mt-2 text-center">
              Generate a schedule on Fleet Status first
            </p>
          )}
        </div>

      </div>

      {/* Middle Section: Map & Co-Pilot */}
      <div className="grid gap-8 mb-8" style={{ gridTemplateColumns: '7fr 2fr', height: 'calc(94vh - 235px)' }}>
        {/* Live Track Feed (Left) */}
        <div className="bg-surface-container-lowest rounded-lg shadow-sm overflow-hidden relative border border-slate-100 h-full min-h-0">
          <div className="absolute top-6 left-6 z-10 flex flex-col gap-2">
            <div className="glass-card px-4 py-2 rounded-full shadow-sm border border-white/40 flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-500 animate-ping"></div>
              <span className="text-xs font-bold font-headline tracking-tight text-slate-700">Live Track Feed</span>
            </div>
          </div>
          {/* Live Kinetic Map Component */}
          <div className="absolute inset-0" style={{ zIndex: 0 }}>
            <KineticMap />
          </div>
          <div className="absolute bottom-6 left-6 right-6 flex justify-between items-end pointer-events-none">
            <div className="flex gap-2 pointer-events-auto">
              <button className="w-10 h-10 glass-card rounded-full flex items-center justify-center shadow-md border border-white text-slate-600">
                <span className="material-symbols-outlined text-sm">add</span>
              </button>
              <button className="w-10 h-10 glass-card rounded-full flex items-center justify-center shadow-md border border-white text-slate-600">
                <span className="material-symbols-outlined text-sm">remove</span>
              </button>
            </div>
            <div className="glass-card p-4 rounded-lg shadow-xl border border-white/40 flex gap-6 pointer-events-auto">
              <div className="text-center">
                <p className="text-[10px] text-slate-400 font-bold uppercase mb-1">Density</p>
                <p className="text-sm font-bold text-slate-700">Optimal</p>
              </div>
              <div className="w-px h-8 bg-slate-200"></div>
              <div className="text-center">
                <p className="text-[10px] text-slate-400 font-bold uppercase mb-1">Switch Stat</p>
                <p className="text-sm font-bold text-slate-700">92% Auto</p>
              </div>
            </div>
          </div>
        </div>

        {/* AI Co-Pilot Panel (Right) — ORBIT Live Feed */}
        <div className="flex flex-col gap-6 h-full min-h-0">
          <div className="bg-surface-container-lowest p-6 rounded-lg shadow-sm border border-slate-100 flex-1 flex flex-col overflow-hidden min-h-0">
            <AICopilotPanel />
          </div>
          {/* Emergency stop card remains as manual override */}
          <button 
            onClick={async () => {
              try {
                const response = await fetch("http://localhost:8000/api/v1/system/lockdown", {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ enabled: !isLockdown })
                });
                if (response.ok) {
                  const data = await response.json();
                  setIsLockdown(data.lockdown);
                }
              } catch (e) {
                console.error("Emergency Stop failed:", e);
              }
            }}
            className={`w-full rounded-lg p-6 text-white shadow-lg transition-all active:scale-95 flex items-center justify-between flex-shrink-0 ${isLockdown ? 'bg-red-600 animate-pulse' : 'bg-[#8B5CF6] hover:bg-red-500 shadow-purple-100'}`}
          >
            <div className="text-left">
              <p className="text-[10px] font-bold uppercase text-purple-100 opacity-80">{isLockdown ? 'System Locked' : 'Manual Override'}</p>
              <p className="text-lg font-headline font-bold">{isLockdown ? 'Resume All Trains' : 'Emergency Stop'}</p>
            </div>
            <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center backdrop-blur-md">
              <span className="material-symbols-outlined text-white">{isLockdown ? 'play_circle' : 'gpp_maybe'}</span>
            </div>
          </button>
        </div>
      </div>

      {/* Bottom Section: Marey Schedule Timeline — Live from ORBIT */}
      <MareyTimeline />

    </div>
  );
};

export default Dashboard;
