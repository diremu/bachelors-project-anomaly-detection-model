import React, { useState, useEffect } from 'react';

// Reusable component for the list-style metric rows
const StatRow = ({ label, value, trend, isPercentage = false }) => (
  <div className="flex justify-between items-center py-2 border-b border-gray-800/50 last:border-0">
    <span className="text-gray-300 text-sm">{label}</span>
    <div className="flex items-center gap-2">
      <span className="text-white font-medium text-lg">
        {value}{isPercentage ? '%' : ''}
      </span>
      {trend === 'up' && <span className="text-green-500 text-xs">▲</span>}
      {trend === 'down' && <span className="text-red-500 text-xs">▼</span>}
    </div>
  </div>
);

// Reusable component for the block-style total cards
const BlockStat = ({ label, value, subtext }) => (
  <div className="bg-[#1c2128] p-4 rounded border border-gray-800 flex flex-col items-center justify-center text-center">
    <span className="text-xs text-gray-400 mb-1">{label}</span>
    <span className="text-2xl font-semibold text-white">{value}</span>
    {subtext && <span className="text-xs text-gray-500 mt-1">{subtext}</span>}
  </div>
);

export function TopMetricsBoard() {
  // A simple mock timer for the "Current Anomaly" to make the UI feel alive
  const [anomalyTimer, setAnomalyTimer] = useState(23);

  useEffect(() => {
    const interval = setInterval(() => setAnomalyTimer(t => t + 1), 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6 font-poppins">
      
      {/* 1. Performance Overview */}
      <div className="bg-[#161b22] p-5 rounded-lg border border-gray-800 flex flex-col">
        <h2 className="text-sm font-medium text-gray-400 mb-4 flex justify-between">
          Performance Overview <span className="text-gray-600">•••</span>
        </h2>
        <div className="flex-1 flex flex-col justify-center">
          <StatRow label="Accurate Detections" value="96.5" isPercentage={true} trend="up" />
          <StatRow label="False Positives" value="2.1" isPercentage={true} trend="down" />
          <StatRow label="False Negatives" value="1.4" isPercentage={true} trend="down" />
        </div>
      </div>

      {/* 2. Timing & Totals */}
      <div className="bg-[#161b22] p-5 rounded-lg border border-gray-800 flex flex-col gap-4">
        <div className="flex justify-between items-center bg-[#1c2128] p-4 rounded border border-gray-800">
          <span className="text-sm text-gray-400">Current Anomaly:</span>
          <span className="text-2xl font-bold text-red-400 animate-pulse">{anomalyTimer}s</span>
        </div>
        <div className="grid grid-cols-2 gap-4 flex-1">
          <BlockStat label="Total Events Logged" value="12,345" />
          <BlockStat label="Anomalies Identified" value="215" />
        </div>
      </div>

      {/* 3. Detailed Anomaly Breakdown */}
      <div className="bg-[#161b22] p-5 rounded-lg border border-gray-800 flex flex-col">
        <h2 className="text-sm font-medium text-gray-400 mb-4 flex justify-between">
          Detailed Breakdown <span className="text-gray-600">•••</span>
        </h2>
        <div className="flex-1 flex flex-col justify-center">
          <StatRow label="Loitering" value="88" />
          <StatRow label="Unusual Aggregation" value="62" />
          <StatRow label="Boundary Breach" value="45" />
          <StatRow label="System Offline" value="20" />
        </div>
      </div>

    </div>
  );
}