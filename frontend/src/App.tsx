import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ToastContainer } from './components/ToastContainer/ToastContainer';
import { MaintenanceDrawer } from './components/MaintenanceDrawer/MaintenanceDrawer';
import './components/ToastContainer/ToastContainer.css';
import './components/MaintenanceDrawer/MaintenanceDrawer.css';
import Page0 from './pages/Page0';
import DashboardLayout from './layouts/DashboardLayout';
import Dashboard from './pages/Dashboard';
import FleetStatus from './pages/FleetStatus';
import ControlCentre from './pages/ControlCentre';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Page0 />} />
        
        {/* All dashboard routes use the shared layout */}
        <Route element={<DashboardLayout />}>
          <Route path="/dashboard"   element={<Dashboard />} />
          <Route path="/fleet"       element={<FleetStatus />} />
          <Route path="/control"     element={<ControlCentre />} />
        </Route>
      </Routes>

      {/* Global ORBIT toast notifications */}
      <ToastContainer />
      {/* Global MMS Maintenance Drawer */}
      <MaintenanceDrawer />
    </Router>
  );
}

export default App;




