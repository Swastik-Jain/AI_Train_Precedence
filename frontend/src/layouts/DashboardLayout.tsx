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
  Activity,
} from 'lucide-react';
import '../pages/Dashboard.css';

/* ────────────────────────────────────────────────────────────────
   SIDEBAR NAV DATA
───────────────────────────────────────────────────────────────── */
const NAV_ITEMS = [
  { id: 'dashboard',    label: 'Dashboard',    icon: LayoutDashboard, route: '/dashboard' },
  { id: 'fleet',        label: 'Fleet Status', icon: Gauge,           route: '/fleet' },
  { id: 'maintenance',  label: 'Maintenance',  icon: Wrench,          route: '/maintenance' },
  { id: 'simulation',   label: 'Simulation',   icon: Activity,        route: '/sandbox' },
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

  // Determine current page name for top bar breadcrumb
  const currentNav = NAV_ITEMS.find(n => n.route === location.pathname);
  const pageTitle = currentNav ? currentNav.label : 'Dashboard';

  return (
    <div className="dash">

      {/* ══════════════════════════════════════════════
          SIDEBAR — Control Center
      ══════════════════════════════════════════════ */}
      <motion.aside
        className="dash__sidebar"
        initial={{ x: -40, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
        role="navigation"
        aria-label="Control Center navigation"
      >
        <div className="dash__sidebar-header">
          <p className="dash__sidebar-title" style={{ whiteSpace: 'nowrap' }}>Control Center</p>
          <span className="dash__sidebar-badge">
            <span className="dash__sidebar-badge-dot" />
            Active Ops: 42
          </span>
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
            <p className="dash__topbar-breadcrumb">
              ORBIT <span>›</span> <strong>{pageTitle}</strong>
            </p>
          </div>
          <div className="dash__topbar-right">
            <p className="dash__topbar-clock">{clock} · IST</p>
            <div className="dash__topbar-status">
              <span className="dash__topbar-status-dot" />
              All Systems Nominal
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
