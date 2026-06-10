import React, { useState } from 'react';

// Mock data structured around pre-escape behavioral indicators
const initialAnomalies = [
  {
    id: 'evt_001',
    timestamp: '10:14:29 AM',
    location: 'Block A - Perimeter Fence 3',
    inmateId: '#451',
    anomalyType: 'Loitering',
    duration: '15s',
    verification: 'Verified',
    severity: 'warning' // determines text color rendering
  },
  {
    id: 'evt_002',
    timestamp: '10:13:42 AM',
    location: 'Yard 1 - Blind Spot',
    inmateId: '#203',
    anomalyType: 'Trajectory Deviation',
    duration: '8s',
    verification: 'Pending',
    severity: 'warning'
  },
  {
    id: 'evt_003',
    timestamp: '10:12:05 AM',
    location: 'Block B - Access Corridor',
    inmateId: '#551',
    anomalyType: 'Boundary Breach',
    duration: '45s',
    verification: 'Verified',
    severity: 'critical'
  },
  {
    id: 'evt_004',
    timestamp: '10:08:11 AM',
    location: 'Block A - Rec Yard',
    inmateId: '#451',
    anomalyType: 'Loitering',
    duration: '22s',
    verification: 'Verified',
    severity: 'warning'
  },
  {
    id: 'evt_005',
    timestamp: '09:55:30 AM',
    location: 'Perimeter Fence 1',
    inmateId: 'Unknown',
    anomalyType: 'Abnormal Aggregation',
    duration: '1m 12s',
    verification: 'Dismissed',
    severity: 'low'
  }
];

export default function RecentAnomaliesLog() {
  const [anomalies, setAnomalies] = useState(initialAnomalies);

  // Helper function to map severity to Tailwind text colors
  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical':
        return 'text-red-500 font-medium';
      case 'warning':
        return 'text-amber-500 font-medium';
      case 'low':
        return 'text-gray-500';
      default:
        return 'text-gray-300';
    }
  };

  const getVerificationBadge = (status) => {
    switch (status) {
      case 'Verified':
        return <span className="text-green-400 bg-green-400/10 px-2 py-1 rounded text-xs">Verified</span>;
      case 'Pending':
        return <span className="text-amber-400 bg-amber-400/10 px-2 py-1 rounded text-xs animate-pulse">Pending</span>;
      case 'Dismissed':
        return <span className="text-gray-500 bg-gray-500/10 px-2 py-1 rounded text-xs">Dismissed</span>;
      default:
        return status;
    }
  };

  return (
    <div className="bg-[#161b22] p-5 rounded-lg border border-gray-800 flex-1 flex flex-col h-full font-poppins">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-sm font-medium text-gray-400">Recent Anomalies Log</h2>
        <button className="text-xs text-teal-500 hover:text-teal-400 transition-colors">
          Export Log ↓
        </button>
      </div>

      {/* Table Container with overflow for scrolling */}
      <div className="overflow-auto flex-1 pr-2">
        <table className="w-full text-left text-sm text-gray-300">
          <thead className="text-xs text-gray-500 uppercase bg-[#0d1117] sticky top-0 z-10">
            <tr>
              <th className="px-4 py-3 font-medium rounded-tl-md">Timestamp</th>
              <th className="px-4 py-3 font-medium">Location</th>
              <th className="px-4 py-3 font-medium">Inmate ID</th>
              <th className="px-4 py-3 font-medium">Anomaly Type</th>
              <th className="px-4 py-3 font-medium">Duration</th>
              <th className="px-4 py-3 font-medium rounded-tr-md">Verification</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {anomalies.map((evt) => (
              <tr 
                key={evt.id} 
                className="hover:bg-[#1c2128] transition-colors group cursor-pointer"
              >
                <td className="px-4 py-3 whitespace-nowrap text-gray-400">{evt.timestamp}</td>
                <td className="px-4 py-3">{evt.location}</td>
                <td className="px-4 py-3 font-mono text-gray-400">{evt.inmateId}</td>
                <td className={`px-4 py-3 ${getSeverityColor(evt.severity)}`}>
                  {evt.anomalyType}
                </td>
                <td className="px-4 py-3">{evt.duration}</td>
                <td className="px-4 py-3">
                  {getVerificationBadge(evt.verification)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        
        {anomalies.length === 0 && (
          <div className="text-center py-8 text-gray-500 text-sm">
            No anomalies detected in the current timeframe.
          </div>
        )}
      </div>
    </div>
  );
}