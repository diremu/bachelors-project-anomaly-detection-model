import React, { useState } from 'react';
import { useAppSelector, useAppDispatch } from '../store';
import { updateVerification } from '../store/dashboardSlice';
import type { Severity, VerificationStatus } from '../types';

export const IncidentReports: React.FC = () => {
  const dispatch = useAppDispatch();
  const allAnomalies = useAppSelector((state) => state.dashboard.recentAnomalies);
  const [searchTerm, setSearchTerm] = useState('');

  // Filter anomalies based on search input
  const filteredAnomalies = allAnomalies.filter(a => 
    a.anomalyType.toLowerCase().includes(searchTerm.toLowerCase()) || 
    a.inmateId.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getSeverityColor = (severity: Severity): string => {
    switch (severity) {
      case 'critical': return 'text-red-500 font-semibold bg-red-500/10 px-2 py-1 rounded-md';
      case 'warning': return 'text-amber-500 font-medium';
      case 'low': return 'text-gray-500';
      default: return 'text-gray-300';
    }
  };

  const getVerificationBadge = (status: VerificationStatus, id: string): React.ReactElement => {
    const baseClasses = "px-3 py-1 rounded-full text-xs font-medium cursor-pointer transition-all hover:opacity-80";
    switch (status) {
      case 'Verified':
        return <span onClick={() => dispatch(updateVerification({id, status: 'Pending'}))} className={`${baseClasses} text-green-400 bg-green-400/10 border border-green-400/20`}>Verified</span>;
      case 'Pending':
        return <span onClick={() => dispatch(updateVerification({id, status: 'Verified'}))} className={`${baseClasses} text-amber-400 bg-amber-400/10 border border-amber-400/20 animate-pulse`}>Pending</span>;
      default:
        return <span className={`${baseClasses} text-gray-500 bg-gray-500/10 border border-gray-500/20`}>{status}</span>;
    }
  };

  return (
    <div className="flex flex-col h-full font-poppins gap-6">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-2xl font-semibold text-white tracking-wide">Incident Reports <span className="text-gray-500 font-light text-lg">| Master Log</span></h1>
          <p className="text-sm text-gray-400 mt-1">Complete historical record of all detected behavioral anomalies.</p>
        </div>
        <div className="w-64">
          <input 
            type="text" 
            placeholder="Search by ID or Type..." 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full px-4 py-2 bg-[#161b22] border border-gray-700 rounded-lg text-gray-300 focus:outline-none focus:border-teal-500 transition-colors text-sm"
          />
        </div>
      </div>

      <div className="bg-[#161b22] rounded-xl border border-gray-800/60 shadow-lg flex-1 overflow-hidden flex flex-col">
        <div className="overflow-auto flex-1 custom-scrollbar">
          <table className="w-full text-left text-sm text-gray-300 border-collapse">
            <thead className="text-xs text-gray-500 uppercase bg-[#0d1117] sticky top-0 z-10 shadow-sm border-b border-gray-800">
              <tr>
                <th className="px-6 py-4 font-medium">Log ID</th>
                <th className="px-6 py-4 font-medium">Timestamp</th>
                <th className="px-6 py-4 font-medium">Location</th>
                <th className="px-6 py-4 font-medium">Inmate ID</th>
                <th className="px-6 py-4 font-medium">Anomaly Type</th>
                <th className="px-6 py-4 font-medium">Verification</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/50">
              {filteredAnomalies.map((evt) => (
                <tr key={evt.id} className="hover:bg-[#1c2128] transition-colors group">
                  <td className="px-6 py-4 font-mono text-gray-500 text-xs">{evt.id}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-gray-400 font-mono text-xs">{evt.timestamp}</td>
                  <td className="px-6 py-4 text-gray-200">{evt.location}</td>
                  <td className="px-6 py-4 font-mono text-gray-400 text-xs">{evt.inmateId}</td>
                  <td className="px-6 py-4"><span className={getSeverityColor(evt.severity)}>{evt.anomalyType}</span></td>
                  <td className="px-6 py-4">{getVerificationBadge(evt.verification, evt.id)}</td>
                </tr>
              ))}
              {filteredAnomalies.length === 0 && (
                <tr>
                  <td colSpan={6} className="text-center py-12 text-gray-500">No anomalies found matching your criteria.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};