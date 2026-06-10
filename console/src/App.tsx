import { Routes, Route, Navigate } from 'react-router-dom';
import DashboardLayout from './components/Dashboard';
import SystemOverview from './components/SystemOverview';

const PlaceholderPage = ({ title }: { title: string }) => (
  <div className="flex h-full items-center justify-center text-gray-500 font-poppins">
    <h2 className="text-xl">{title} Module - Pending Implementation</h2>
  </div>
);

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<DashboardLayout />}>
        {/* Redirect root to the overview */}
        <Route index element={<Navigate to="/overview" replace />} />
        
        <Route path="overview" element={<SystemOverview />} />
        <Route path="live-feeds" element={<PlaceholderPage title="Real-time Feeds" />} />
        <Route path="reports" element={<PlaceholderPage title="Incident Reports" />} />
        <Route path="config" element={<PlaceholderPage title="Configuration" />} />
      </Route>
    </Routes>
  );
}