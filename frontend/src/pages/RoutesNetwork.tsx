import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Map,
  SplitSquareHorizontal,
  Wifi,
  GitMerge,
  Clock,
  ArrowRight
} from 'lucide-react';
import './RoutesNetwork.css';
import { KineticMap } from '../components/KineticMap/KineticMap';


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
   TIMELINE DATA
───────────────────────────────────────────────────────────────── */
const EVENTS = [
  { id: 1, title: 'TR-042 Departed',          desc: 'Washington Union Station', type: 'primary', time: '2 mins ago' },
  { id: 2, title: 'Switch Delay',             desc: 'Gate 4-B Terminal Philly', type: 'warning', time: '12 mins ago' },
  { id: 3, title: 'Refuel Complete',          desc: 'L-902 Engine Yard',        type: 'success', time: '24 mins ago' },
  { id: 4, title: 'Maintenance End',          desc: 'Segment P-42 Newark',      type: 'success', time: '1 hour ago' },
];

const ROUTES = [
  { from: 'New York Penn',  to: 'Philadelphia 30th St' },
  { from: 'Harrisburg',     to: 'Pittsburgh Terminal' },
  { from: 'Albany-Renss',   to: 'NYC Grand Central' },
];

const RoutesNetwork: React.FC = () => {
  useEffect(() => {
    document.title = 'Routes & Network - Zentra Ops | ORBIT';
  }, []);

  return (
    <div className="routes-grid">
      
      {/* ── Top Info Row ── */}
      <motion.div 
        className="routes-top"
        initial="hidden"
        animate="visible"
        variants={stagger}
      >
        <motion.div className="dash-panel" custom={0.0} variants={fadeUp}>
          <div className="dash-panel__header">
            <p className="dash-panel__eyebrow">Selected Hub</p>
            <h2 className="dash-panel__title">
              <Map size={15} className="dash-panel__title-icon" strokeWidth={2} />
              Philadelphia Terminal
            </h2>
          </div>
          <div className="dash-panel__body opt-details">
             <p>Active Tracks: <b>24/28</b></p>
             <p>Congestion: <b className="text-success">Low</b></p>
          </div>
        </motion.div>

        <motion.div className="dash-panel panel-opt" custom={0.1} variants={fadeUp}>
          <div className="dash-panel__header">
            <p className="dash-panel__eyebrow">Route Optimization AI</p>
            <h2 className="dash-panel__title">
              <SplitSquareHorizontal size={15} className="dash-panel__title-icon text-primary" strokeWidth={2} />
              Recommended Path
            </h2>
          </div>
          <div className="dash-panel__body opt-details">
             <p><b>E-392 Diversion via Lancaster</b></p>
             <p>Current Queue: 3 units awaiting yard clearance at Newark</p>
          </div>
        </motion.div>

        <motion.div className="dash-panel" custom={0.2} variants={fadeUp}>
          <div className="dash-panel__header">
            <p className="dash-panel__eyebrow">Network Health</p>
            <h2 className="dash-panel__title">
              <Wifi size={15} className="dash-panel__title-icon" strokeWidth={2} />
              98.4%
            </h2>
          </div>
          <div className="dash-panel__body opt-details">
             <p>Uptime tracking for all sensor nodes and signaling relays</p>
          </div>
        </motion.div>
      </motion.div>

      {/* ── Main Spline & Right Col ── */}
      <div className="routes-main">
        
        {/* Kinetic Map Area */}
        <motion.div 
          className="dash-panel"
          custom={0.3}
          initial="hidden"
          animate="visible"
          variants={fadeUp}
          style={{ gridColumn: 'span 2', minHeight: '500px' }}
        >
          <div className="dash-panel__header">
            <p className="dash-panel__eyebrow">Real-time Topographic Network Map (Live Engine)</p>
            <h2 className="dash-panel__title">
              <Map size={15} className="dash-panel__title-icon" strokeWidth={2} />
              Kinetic Network Map
            </h2>
          </div>
          
          <div className="dash-panel__body p-0 relative" style={{ height: 'calc(100% - 60px)' }}>
            <KineticMap />
          </div>
        </motion.div>

      </div>
    </div>
  );
};

export default RoutesNetwork;
