import { useState } from 'react';
import { useAppSelector, useAppDispatch } from '../store';
import { updateVerification } from '../store/dashboardSlice';
import type { Severity, VerificationStatus } from '../types';

type FilterState = {
  search: string;
  severity: Severity | 'all';
  verification: VerificationStatus | 'all';
};

export const IncidentReports = () => {
  const dispatch = useAppDispatch();
  const allAnomalies = useAppSelector((state) => state.dashboard.recentAnomalies);
  const [filters, setFilters] = useState<FilterState>({
    search: '',
    severity: 'all',
    verification: 'all',
  });

  const filtered = allAnomalies.filter(a => {
    if (filters.search) {
      const q = filters.search.toLowerCase();
      if (!a.anomalyType.toLowerCase().includes(q) &&
          !a.inmateId.toLowerCase().includes(q) &&
          !a.location.toLowerCase().includes(q)) {
        return false;
      }
    }
    if (filters.severity !== 'all' && a.severity !== filters.severity) return false;
    if (filters.verification !== 'all' && a.verification !== filters.verification) return false;
    return true;
  });

  const severityCounts = {
    all: allAnomalies.length,
    critical: allAnomalies.filter(a => a.severity === 'critical').length,
    warning: allAnomalies.filter(a => a.severity === 'warning').length,
    low: allAnomalies.filter(a => a.severity === 'low').length,
  };

  const getSeverityColor = (severity: Severity): string => {
    switch (severity) {
      case 'critical': return 'text-red-500 font-semibold bg-red-500/10 px-2 py-1 rounded-md';
      case 'warning': return 'text-amber-500 font-medium';
      case 'low': return 'text-gray-500';
      default: return 'text-gray-300';
    }
  };

  const getVerificationBadge = (status: VerificationStatus, id: string) => {
    const base = "px-3 py-1 rounded-full text-xs font-medium cursor-pointer transition-all hover:opacity-80";
    switch (status) {
      case 'Verified':
        return <span onClick={() => dispatch(updateVerification({ id, status: 'Pending' }))} className={`${base} text-green-400 bg-green-400/10 border border-green-400/20`}>Verified</span>;
      case 'Pending':
        return <span onClick={() => dispatch(updateVerification({ id, status: 'Verified' }))} className={`${base} text-amber-400 bg-amber-400/10 border border-amber-400/20 animate-pulse`}>Pending</span>;
      default:
        return <span className={`${base} text-gray-500 bg-gray-500/10 border border-gray-500/20`}>{status}</span>;
    }
  };

  const chipClass = (active: boolean) =>
    `px-3 py-1.5 rounded text-xs font-medium transition-colors ${
      active ? 'bg-teal-600 text-white' : 'bg-[#0d1117] text-gray-400 border border-gray-700 hover:text-white'
    }`;

  return (
    <div className="flex flex-col h-full font-poppins gap-6">
      {/* Header + filters */}
      <div className="flex flex-col gap-4">
        <div className="flex justify-between items-end">
          <div>
            <h1 className="text-2xl font-semibold text-white tracking-wide">
              Incident Reports <span className="text-gray-500 font-light text-lg">| Master Log</span>
            </h1>
            <p className="text-sm text-gray-400 mt-1">Complete record of all detected anomalies. Click verification badges to update status.</p>
          </div>
          <div className="w-64">
            <input
              type="text"
              placeholder="Search by ID, type, or location..."
              value={filters.search}
              onChange={(e) => setFilters({ ...filters, search: e.target.value })}
              className="w-full px-4 py-2 bg-[#161b22] border border-gray-700 rounded-lg text-gray-300 focus:outline-none focus:border-teal-500 transition-colors text-sm"
            />
          </div>
        </div>

        {/* Filter chips */}
        <div className="flex items-center gap-3 flex-wrap">
          <span className="text-xs text-gray-500 uppercase tracking-wider">Severity:</span>
          {(['all', 'critical', 'warning', 'low'] as const).map(s => (
            <button
              key={s}
              onClick={() => setFilters({ ...filters, severity: s })}
              className={chipClass(filters.severity === s)}
            >
              {s === 'all' ? 'All' : s.charAt(0).toUpperCase() + s.slice(1)}
              <span className="ml-1.5 opacity-60">{severityCounts[s]}</span>
            </button>
          ))}

          <span className="mx-2 text-gray-700">|</span>

          <span className="text-xs text-gray-500 uppercase tracking-wider">Status:</span>
          {(['all', 'Pending', 'Verified'] as const).map(v => (
            <button
              key={v}
              onClick={() => setFilters({ ...filters, verification: v })}
              className={chipClass(filters.verification === v)}
            >
              {v === 'all' ? 'All' : v}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
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
              {filtered.map((evt) => (
                <tr key={evt.id} className="hover:bg-[#1c2128] transition-colors group">
                  <td className="px-6 py-4 font-mono text-gray-500 text-xs">{evt.id}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-gray-400 font-mono text-xs">{evt.timestamp}</td>
                  <td className="px-6 py-4 text-gray-200">{evt.location}</td>
                  <td className="px-6 py-4 font-mono text-gray-400 text-xs">{evt.inmateId}</td>
                  <td className="px-6 py-4"><span className={getSeverityColor(evt.severity)}>{evt.anomalyType}</span></td>
                  <td className="px-6 py-4">{getVerificationBadge(evt.verification, evt.id)}</td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={6} className="text-center py-12 text-gray-500">No incidents match the current filters.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="px-6 py-3 border-t border-gray-800 bg-[#0d1117] text-xs text-gray-500 flex justify-between">
          <span>Showing {filtered.length} of {allAnomalies.length} incidents</span>
          <span>{allAnomalies.filter(a => a.verification === 'Pending').length} pending review</span>
        </div>
      </div>
    </div>
  );
};