import './App.css';
import { useState } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import DashboardLayout from './components/Dashboard';
import SystemOverview from './components/SystemOverview';
import { Auth } from './pages/Auth';
import { Configuration } from './pages/Configuration';
import { IncidentReports } from './pages/IncidentReports';
import {LiveFeeds} from './pages/LiveFeed'

const PlaceholderPage = ({ title }: { title: string }) => (
  <div className="flex h-full items-center justify-center text-gray-500 font-poppins">
    <h2 className="text-xl">{title} Module - Pending Implementation</h2>
  </div>
);

const AUTH_SESSION_KEY = 'console_auth_active';

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(
    () => localStorage.getItem(AUTH_SESSION_KEY) === 'true'
  );

  const handleLoginSuccess = () => {
    localStorage.setItem(AUTH_SESSION_KEY, 'true');
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem(AUTH_SESSION_KEY);
    setIsAuthenticated(false);
  };

  return (
    <Routes>
      <Route
        path="/login"
        element={
          isAuthenticated ? (
            <Navigate to="/overview" replace />
          ) : (
            <Auth onAuthSuccess={handleLoginSuccess} />
          )
        }
      />
      <Route
        path="/"
        element={
          isAuthenticated ? <DashboardLayout onLogout={handleLogout} /> : <Navigate to="/login" replace />
        }
      >
        <Route index element={<Navigate to="overview" replace />} />
        <Route path="overview" element={<SystemOverview />} />
        <Route path="live-feeds" element={<LiveFeeds />} />
        <Route path="reports" element={<IncidentReports />} />
        <Route path="config" element={<Configuration />} />
      </Route>
      <Route
        path="*"
        element={
          isAuthenticated ? <Navigate to="/overview" replace /> : <Navigate to="/login" replace />
        }
      />
    </Routes>
  );
}