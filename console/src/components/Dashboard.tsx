import Sidebar from './Sidebar';
import { Outlet, useLocation } from 'react-router-dom';
import { useEffect, useMemo, useState } from 'react';

type DashboardProps = {
  onLogout: () => void;
};

export default function Dashboard({ onLogout }: DashboardProps) {
  const [currentTime, setCurrentTime] = useState(new Date());
  const location = useLocation();

  const sectionTitle = useMemo(() => {
    if (location.pathname.startsWith('/live-feeds')) return 'Real-time Feeds';
    if (location.pathname.startsWith('/model-performance')) return 'Model Performance';
    if (location.pathname.startsWith('/reports')) return 'Incident Reports';
    if (location.pathname.startsWith('/config')) return 'Storage & Integration';
    return 'System Overview';
  }, [location.pathname]);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="flex h-screen w-full bg-[#0d1117] text-gray-200 font-poppins overflow-hidden">
      <div className="absolute top-0 w-full h-16 flex items-center justify-between px-6 bg-[#161b22] border-b border-gray-800 z-10">
        <div className="flex items-center gap-3">
          <span className="text-xl font-semibold tracking-wide">
            {sectionTitle}
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-400">{currentTime.toLocaleTimeString('en-us',{hour12: true})}</span>
          <div className="px-3 py-1 bg-green-900/30 text-green-400 rounded-full text-xs font-medium flex items-center gap-2">
            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
            LIVE
          </div>
        </div>
      </div>

      <div className="flex w-full h-full pt-16">
        <Sidebar onLogout={onLogout} />
        <main className="flex-1 p-6 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}