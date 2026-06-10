import { useEffect } from 'react';
import { tickTimer } from '../store/dashboardSlice';
import { useAppDispatch, useAppSelector } from '../store';

type StatRowProps = {
  label: string;
  value: number | string;
  trend?: 'up' | 'down';
  isPercentage?: boolean;
};

const StatRow = ({ label, value, trend, isPercentage = false }: StatRowProps) => (
  <div className="flex justify-between items-center py-2 border-b border-gray-800/50 last:border-0 hover:bg-[#1c2128]/50 px-2 rounded transition-colors">
    <span className="text-gray-300 text-sm">{label}</span>
    <div className="flex items-center gap-2">
      <span className="text-white font-medium text-lg">
        {value}{isPercentage ? '%' : ''}
      </span>
      {trend === 'up' && <span className="text-green-500 text-xs font-medium">▲</span>}
      {trend === 'down' && <span className="text-red-500 text-xs font-medium">▼</span>}
    </div>
  </div>
);

const BlockStat = ({ label, value }: { label: string; value: number | string }) => (
  <div className="bg-gradient-to-br from-[#1c2128] to-[#161b22] p-4 rounded border border-gray-800 flex flex-col items-center justify-center text-center shadow-inner">
    <span className="text-xs text-gray-400 mb-1">{label}</span>
    <span className="text-2xl font-semibold text-white tracking-tight">{value.toLocaleString()}</span>
  </div>
);

export function TopMetrics() {
  const dispatch = useAppDispatch();
  const { metrics, breakdown, currentAnomalyTimer } = useAppSelector((state) => state.dashboard);

  useEffect(() => {
    const interval = setInterval(() => dispatch(tickTimer()), 1000);
    return () => clearInterval(interval);
  }, [dispatch]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6 font-poppins">
      
      {/* 1. Performance Overview */}
      <div className="bg-[#161b22] p-5 rounded-xl border border-gray-800/60 shadow-lg flex flex-col">
        <h2 className="text-sm font-medium text-gray-400 mb-4 flex justify-between items-center">
          Performance Metrics <span className="text-gray-600 cursor-pointer hover:text-gray-400 transition-colors">•••</span>
        </h2>
        <div className="flex-1 flex flex-col justify-center gap-1">
          <StatRow label="Accurate Detections" value={metrics.accurateDetections} isPercentage={true} trend="up" />
          <StatRow label="False Positives" value={metrics.falsePositives} isPercentage={true} trend="down" />
          <StatRow label="False Negatives" value={metrics.falseNegatives} isPercentage={true} trend="down" />
        </div>
      </div>

      {/* 2. Timing & Totals */}
      <div className="bg-[#161b22] p-5 rounded-xl border border-gray-800/60 shadow-lg flex flex-col gap-4">
        <div className="flex justify-between items-center bg-[#1c2128] p-4 rounded-lg border border-gray-700/50 shadow-inner">
          <span className="text-sm text-gray-400 font-medium">Current Anomaly:</span>
          <div className="flex items-baseline gap-1 text-red-400">
            <span className="text-3xl font-bold animate-pulse">{currentAnomalyTimer}</span>
            <span className="text-sm font-medium">sec</span>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4 flex-1">
          <BlockStat label="Total Events" value={metrics.totalEvents} />
          <BlockStat label="Anomalies" value={metrics.anomaliesIdentified} />
        </div>
      </div>

      {/* 3. Detailed Breakdown */}
      <div className="bg-[#161b22] p-5 rounded-xl border border-gray-800/60 shadow-lg flex flex-col">
        <h2 className="text-sm font-medium text-gray-400 mb-4 flex justify-between items-center">
          Detailed Breakdown <span className="text-gray-600 cursor-pointer hover:text-gray-400 transition-colors">•••</span>
        </h2>
        <div className="flex-1 flex flex-col justify-center gap-1">
          <StatRow label="Loitering" value={breakdown.loitering} />
          <StatRow label="Unusual Aggregation" value={breakdown.unusualAggregation} />
          <StatRow label="Boundary Breach" value={breakdown.boundaryBreach} />
          <StatRow label="System Offline" value={breakdown.systemOffline} />
        </div>
      </div>

    </div>
  );
}