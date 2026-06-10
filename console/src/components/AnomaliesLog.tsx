import { useAppDispatch, useAppSelector } from '../store';
import type { VerificationStatus } from '../types';
import { updateVerification } from '../store/dashboardSlice';

export default function RecentAnomaliesLog() {
  const dispatch = useAppDispatch();
  const anomalies = useAppSelector((state) => state.dashboard.recentAnomalies);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-500 font-semibold bg-red-500/10 px-2 py-1 rounded-md';
      case 'warning': return 'text-amber-500 font-medium';
      case 'low': return 'text-gray-500';
      default: return 'text-gray-300';
    }
  };

  const getVerificationBadge = (status: VerificationStatus, id: string) => {
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
    <div className="bg-[#161b22] p-5 rounded-xl border border-gray-800/60 shadow-lg flex-1 flex flex-col min-h-[300px] font-poppins relative overflow-hidden">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-sm font-semibold text-gray-300 tracking-wide uppercase">Recent Anomalies Log</h2>
        <button className="text-xs font-medium text-teal-500 hover:text-teal-400 transition-colors flex items-center gap-1 bg-teal-500/10 px-3 py-1.5 rounded-md">
          Export Log
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
        </button>
      </div>

      <div className="overflow-auto flex-1 pr-2 custom-scrollbar">
        <table className="w-full text-left text-sm text-gray-300 border-collapse">
          <thead className="text-xs text-gray-500 uppercase bg-[#0d1117] sticky top-0 z-10 shadow-sm border-b border-gray-800">
            <tr>
              <th className="px-4 py-4 font-medium rounded-tl-lg">Timestamp</th>
              <th className="px-4 py-4 font-medium">Location</th>
              <th className="px-4 py-4 font-medium">Inmate ID</th>
              <th className="px-4 py-4 font-medium">Anomaly Type</th>
              <th className="px-4 py-4 font-medium">Duration</th>
              <th className="px-4 py-4 font-medium rounded-tr-lg">Verification</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/50">
            {anomalies.map((evt, index) => (
              <tr 
                key={evt.id} 
                className={`hover:bg-[#1c2128] transition-all group cursor-default animate-[fadeIn_0.4s_ease-out_forwards]`}
                style={{ animationDelay: `${index * 0.05}s`, opacity: 0 }}
              >
                <td className="px-4 py-4 whitespace-nowrap text-gray-400 font-mono text-xs">{evt.timestamp}</td>
                <td className="px-4 py-4 text-gray-200">{evt.location}</td>
                <td className="px-4 py-4 font-mono text-gray-400 text-xs bg-[#0d1117]/50 rounded my-2 inline-block ml-4 px-2">{evt.inmateId}</td>
                <td className="px-4 py-4">
                  <span className={getSeverityColor(evt.severity)}>{evt.anomalyType}</span>
                </td>
                <td className="px-4 py-4 font-mono text-gray-400 text-xs">{evt.duration}</td>
                <td className="px-4 py-4">
                  {getVerificationBadge(evt.verification, evt.id)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}