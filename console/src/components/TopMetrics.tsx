import { useEffect, useState } from 'react';
import { tickTimer } from '../store/dashboardSlice';
import { useAppDispatch, useAppSelector } from '../store';

type ModelMetrics = {
  evaluation: {
    auc_roc: number | null;
    pr_auc: number | null;
    eer: number | null;
    optimal_threshold: number | null;
  };
  confusion_matrix: {
    tn: number; fp: number; fn: number; tp: number;
  } | null;
  per_category: Record<string, {
    count: number;
    mean_score: number;
    label: string;
  }>;
};

const StatRow = ({ label, value, trend, isPercentage = false }: {
  label: string; value: number | string; trend?: 'up' | 'down'; isPercentage?: boolean;
}) => (
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

function formatTimeAgo(seconds: number): string {
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s ago`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m ago`;
}

export function TopMetrics() {
  const dispatch = useAppDispatch();
  const { metrics, currentAnomalyTimer, recentAnomalies } = useAppSelector((state) => state.dashboard);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);

  useEffect(() => {
    fetch('/model-metrics.json')
      .then(r => r.ok ? r.json() : null)
      .then(setModelMetrics)
      .catch(() => {});
  }, []);

  useEffect(() => {
    const interval = setInterval(() => dispatch(tickTimer()), 1000);
    return () => clearInterval(interval);
  }, [dispatch]);

  const cm = modelMetrics?.confusion_matrix;
  const precision = cm ? cm.tp / (cm.tp + cm.fp) : metrics.accurateDetections / 100;
  const recall = cm ? cm.tp / (cm.tp + cm.fn) : (100 - metrics.falseNegatives) / 100;
  const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
  const fpr = cm ? cm.fp / (cm.fp + cm.tn) : metrics.falsePositives / 100;
  const fnr = cm ? cm.fn / (cm.fn + cm.tp) : metrics.falseNegatives / 100;

  const categories = modelMetrics?.per_category ?? {};
  const anomalousCategories = Object.entries(categories)
    .filter(([, d]) => d.label === 'anomalous')
    .sort(([, a], [, b]) => b.count - a.count)
    .slice(0, 4);

  const latest = recentAnomalies[0] ?? null;
  const isClear = !latest || currentAnomalyTimer >= 300;

  const severityStyles: Record<string, { border: string; bg: string; text: string; label: string }> = {
    critical: { border: 'border-red-500/40', bg: 'bg-red-500/10', text: 'text-red-400', label: 'CRITICAL' },
    warning:  { border: 'border-amber-500/40', bg: 'bg-amber-500/10', text: 'text-amber-400', label: 'WARNING' },
    low:      { border: 'border-gray-500/40', bg: 'bg-gray-500/10', text: 'text-gray-400', label: 'LOW' },
  };
  const sev = latest ? (severityStyles[latest.severity] ?? severityStyles.low) : severityStyles.low;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6 font-poppins">

      {/* 1. Model Performance */}
      <div className="bg-[#161b22] p-5 rounded-xl border border-gray-800/60 shadow-lg flex flex-col">
        <h2 className="text-sm font-medium text-gray-400 mb-4 flex justify-between items-center">
          Model Performance <span className="text-gray-600 cursor-pointer hover:text-gray-400 transition-colors">•••</span>
        </h2>
        <div className="flex-1 flex flex-col justify-center gap-1">
          <StatRow label="Precision" value={(precision * 100).toFixed(1)} isPercentage={true} trend="up" />
          <StatRow label="Recall" value={(recall * 100).toFixed(1)} isPercentage={true} trend="up" />
          <StatRow label="F1 Score" value={f1.toFixed(3)} />
          <StatRow label="False Positive Rate" value={(fpr * 100).toFixed(1)} isPercentage={true} trend="down" />
          <StatRow label="False Negative Rate" value={(fnr * 100).toFixed(1)} isPercentage={true} trend="down" />
        </div>
      </div>

      {/* 2. Latest Incident Spotlight */}
      <div className={`bg-[#161b22] p-5 rounded-xl border shadow-lg flex flex-col gap-4 ${isClear ? 'border-green-500/30' : sev.border}`}>
        <h2 className="text-sm font-medium text-gray-400 flex justify-between items-center">
          Latest Incident
          {isClear ? (
            <span className="px-2.5 py-1 rounded-full text-[10px] font-bold tracking-wider bg-green-500/10 text-green-400">
              ALL CLEAR
            </span>
          ) : latest && (
            <span className={`px-2.5 py-1 rounded-full text-[10px] font-bold tracking-wider ${sev.bg} ${sev.text}`}>
              {sev.label}
            </span>
          )}
        </h2>

        {isClear ? (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 py-4">
            <div className="w-14 h-14 rounded-full bg-green-500/10 border border-green-500/20 flex items-center justify-center">
              <svg className="w-7 h-7 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <p className="text-green-400 font-medium text-sm">Normal Behaviour Detected</p>
            <p className="text-xs text-gray-500 text-center">
              {latest
                ? `No anomalies in the last ${formatTimeAgo(currentAnomalyTimer).replace(' ago', '')}. Last incident was ${latest.anomalyType} at ${latest.location}.`
                : 'No incidents have been recorded this session.'}
            </p>
          </div>
        ) : latest && (
          <>
            <div className={`p-4 rounded-lg border ${sev.border} ${sev.bg}`}>
              <p className={`text-lg font-semibold ${sev.text}`}>{latest.anomalyType}</p>
              <p className="text-sm text-gray-300 mt-1">{latest.location}</p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="bg-[#0d1117] p-3 rounded-lg border border-gray-800/50">
                <span className="text-[10px] text-gray-500 uppercase tracking-wider block mb-1">Time Elapsed</span>
                <span className="text-xl font-bold text-white">{formatTimeAgo(currentAnomalyTimer)}</span>
              </div>
              <div className="bg-[#0d1117] p-3 rounded-lg border border-gray-800/50">
                <span className="text-[10px] text-gray-500 uppercase tracking-wider block mb-1">Reported At</span>
                <span className="text-xl font-bold text-white">{latest.timestamp}</span>
              </div>
            </div>

            <div className="flex justify-between items-center text-xs">
              <span className="text-gray-500">Inmate {latest.inmateId} · Duration {latest.duration}</span>
              <span className={`px-2 py-0.5 rounded-full ${
                latest.verification === 'Verified'
                  ? 'text-green-400 bg-green-400/10'
                  : 'text-amber-400 bg-amber-400/10 animate-pulse'
              }`}>
                {latest.verification}
              </span>
            </div>
          </>
        )}
      </div>

      {/* 3. Category Breakdown */}
      <div className="bg-[#161b22] p-5 rounded-xl border border-gray-800/60 shadow-lg flex flex-col">
        <h2 className="text-sm font-medium text-gray-400 mb-4 flex justify-between items-center">
          Detection Categories <span className="text-gray-600 cursor-pointer hover:text-gray-400 transition-colors">•••</span>
        </h2>
        <div className="flex-1 flex flex-col justify-center gap-1">
          {anomalousCategories.length > 0 ? (
            anomalousCategories.map(([name, data]) => (
              <StatRow
                key={name}
                label={name.replace(/_/g, ' ')}
                value={data.count}
              />
            ))
          ) : (
            <>
              <StatRow label="Session Anomalies" value={recentAnomalies.filter(a => a.severity === 'critical').length} />
              <StatRow label="Pending Review" value={recentAnomalies.filter(a => a.verification === 'Pending').length} />
              <StatRow label="Verified" value={recentAnomalies.filter(a => a.verification === 'Verified').length} />
            </>
          )}
        </div>

        {modelMetrics?.evaluation.auc_roc && (
          <div className="mt-4 bg-[#1c2128] p-3 rounded-lg border border-gray-700/50 flex justify-between items-center">
            <span className="text-xs text-gray-400">AUC-ROC</span>
            <span className="text-lg font-semibold text-teal-400">{modelMetrics.evaluation.auc_roc.toFixed(4)}</span>
          </div>
        )}
      </div>

    </div>
  );
}