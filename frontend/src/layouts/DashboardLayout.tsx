import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation, Outlet } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  LayoutDashboard,
  Gauge,
  GitBranch,
  Wrench,
  TrendingUp,
  HelpCircle,
  LogOut,
  ChevronRight,
  Menu,
} from 'lucide-react';
import { useSystemStore } from '../store/useSystemStore';
import { useMapStore } from '../store/useMapStore';
import { useCopilotStore } from '../store/useCopilotStore';
import '../pages/Dashboard.css';

/* ────────────────────────────────────────────────────────────────
   SIDEBAR NAV DATA
───────────────────────────────────────────────────────────────── */
const NAV_ITEMS = [
  { id: 'dashboard',    label: 'Dashboard',    icon: LayoutDashboard, route: '/dashboard' },
  { id: 'fleet',        label: 'Fleet Status', icon: Gauge,           route: '/fleet' },
  { id: 'control',      label: 'Control Centre', icon: Wrench,        route: '/control' },
];

/* ────────────────────────────────────────────────────────────────
   LIVE CLOCK HOOK
───────────────────────────────────────────────────────────────── */
function useLiveClock(): string {
  const [time, setTime] = useState('');
  useEffect(() => {
    const fmt = () => new Date().toLocaleTimeString('en-GB', {
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    });
    setTime(fmt());
    const id = setInterval(() => setTime(fmt()), 1000);
    return () => clearInterval(id);
  }, []);
  return time;
}

const DashboardLayout: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const clock = useLiveClock();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const { 
    isLockdown, isSafetyShield,
    networkFluidity, haltedPct, activeTrains,
    fetchStatus, setLockdown, setSafetyShield 
  } = useSystemStore();
  const mapConnected = useMapStore(state => state.isConnected);
  const conflicts = useMapStore(state => state.conflicts);
  const copilotConnected = useCopilotStore(state => state.isConnected);

  useEffect(() => {
    fetchStatus();
    const id = setInterval(fetchStatus, 3000);
    return () => clearInterval(id);
  }, [fetchStatus]);

  // Determine current page name for top bar breadcrumb
  const currentNav = NAV_ITEMS.find(n => n.route === location.pathname);
  const pageTitle = currentNav ? currentNav.label : 'Dashboard';

  const deriveSystemStatus = () => {
    if (isLockdown) return { label: 'Emergency Lockdown', style: 'text-red-600 bg-red-50', dot: 'bg-red-500 shadow-[0_0_6px_rgba(239,68,68,0.6)]', pulsing: true };
    if (!mapConnected) return { label: 'Backend Offline', style: 'text-red-600 bg-red-50', dot: 'bg-red-500 shadow-[0_0_6px_rgba(239,68,68,0.6)]', pulsing: false };
    if (networkFluidity === 'Degraded') return { label: `Network Degraded · ${haltedPct.toFixed(0)}% halted`, style: 'text-red-600 bg-red-50', dot: 'bg-red-500 shadow-[0_0_6px_rgba(239,68,68,0.6)]', pulsing: true };
    if (conflicts && conflicts.length > 0) return { label: `Conflict Detected · ${conflicts.length} edge${conflicts.length > 1 ? 's' : ''}`, style: 'text-amber-600 bg-amber-50', dot: 'bg-amber-500 shadow-[0_0_6px_rgba(245,158,11,0.6)]', pulsing: true };
    if (networkFluidity === 'Warning') return { label: `Network Warning · ${haltedPct.toFixed(0)}% halted`, style: 'text-amber-600 bg-amber-50', dot: 'bg-amber-500 shadow-[0_0_6px_rgba(245,158,11,0.6)]', pulsing: false };
    if (!copilotConnected) return { label: 'Co-Pilot Offline', style: 'text-amber-600 bg-amber-50', dot: 'bg-amber-500 shadow-[0_0_6px_rgba(245,158,11,0.6)]', pulsing: false };
    return { label: 'All Systems Nominal', style: 'text-emerald-600 bg-emerald-50', dot: 'bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.6)]', pulsing: false };
  };
  const statusInfo = deriveSystemStatus();

  return (
    <div className="dash">

      {/* ══════════════════════════════════════════════
          SIDEBAR — Control Center
      ══════════════════════════════════════════════ */}
      {/* Overlay for mobile when sidebar is open */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-slate-900/50 z-40 md:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}
      <motion.aside
        className={`dash__sidebar ${isSidebarOpen ? 'open' : ''}`}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
        role="navigation"
        aria-label="Control Center navigation"
      >
        <div className="dash__sidebar-header">
          <p className="dash__sidebar-title" style={{ whiteSpace: 'nowrap' }}>Control Center</p>
        </div>

        <nav className="dash__nav">
          {NAV_ITEMS.map((item) => {
            const active = location.pathname === item.route;
            return (
              <button
                key={item.id}
                id={`nav-${item.id}`}
                className={`dash__nav-item${active ? ' dash__nav-item--active' : ''}`}
                onClick={() => navigate(item.route)}
                aria-current={active ? 'page' : undefined}
              >
                <item.icon
                  size={15}
                  strokeWidth={active ? 2.5 : 1.75}
                  className="dash__nav-item__icon"
                />
                <span>{item.label}</span>
                {active && <ChevronRight size={10} style={{ marginLeft: 'auto', opacity: 0.5 }} />}
              </button>
            );
          })}

          <div className="dash__nav-separator" style={{ marginTop: 'auto' }} />

          <button
            id="nav-logout"
            className="dash__nav-item dash__nav-item--logout"
            onClick={() => navigate('/')}
          >
            <LogOut size={15} strokeWidth={1.75} className="dash__nav-item__icon" />
            <span>Log Out</span>
          </button>
        </nav>
      </motion.aside>

      {/* ══════════════════════════════════════════════
          MAIN AREA W/ TOPBAR & OUTLET
      ══════════════════════════════════════════════ */}
      <main className="dash__main">

        <motion.div
          className="dash__topbar"
          initial={{ opacity: 0, y: -16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
        >
          <div className="dash__topbar-left">
            <button 
              className="dash-menu-toggle"
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              aria-label="Toggle Menu"
            >
              <Menu size={20} />
            </button>
            <p className="dash__topbar-breadcrumb">
              ORBIT <span>›</span> <strong>{pageTitle}</strong>
            </p>
          </div>
          <div className="dash__topbar-right" style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
            <div className="flex gap-2 mr-4">
              <button 
                onClick={() => setSafetyShield(!isSafetyShield)}
                className={`px-3 py-1 text-[10px] uppercase font-bold tracking-wider rounded-full border transition-colors ${isSafetyShield ? 'bg-tertiary/10 border-tertiary text-tertiary' : 'bg-surface-container border-outline-variant/20 text-on-surface-variant'}`}
              >
                OR-Shield
              </button>
              <button 
                onClick={() => setLockdown(!isLockdown)}
                className={`px-3 py-1 text-[10px] uppercase font-bold tracking-wider rounded-full border transition-colors ${isLockdown ? 'bg-rose-500 border-rose-600 text-white animate-pulse' : 'bg-surface-container border-outline-variant/20 text-on-surface-variant hover:border-rose-500/50 hover:text-rose-500'}`}
              >
                Lockdown
              </button>
            </div>
            <p className="dash__topbar-clock">{clock} · IST</p>
            <div className={`dash__topbar-status ${statusInfo.style} ${statusInfo.pulsing ? 'animate-pulse' : ''}`}>
              <span className={`dash__topbar-status-dot ${statusInfo.dot}`} style={statusInfo.pulsing ? undefined : { animation: 'none' }} />
              {statusInfo.label}
            </div>
          </div>
        </motion.div>

        {/* Scrollable content loads here */}
        <div className="dash__content">
          <Outlet />
        </div>

      </main>
    </div>
  );
};

export default DashboardLayout;

