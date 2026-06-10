import React from 'react';

export default function SystemOverview() {
  return (
    <div className="grid grid-cols-3 gap-6 h-full">
      <div className="col-span-1 flex flex-col gap-6">
        <div className="bg-[#161b22] p-5 rounded-lg border border-gray-800 flex-1">
          <h2 className="text-sm font-medium text-gray-400 mb-4">Performance Overview</h2>
        </div>
        <div className="bg-[#161b22] p-5 rounded-lg border border-gray-800 flex-1">
          <h2 className="text-sm font-medium text-gray-400 mb-4">Anomaly Type Distribution</h2>
        </div>
      </div>

      <div className="col-span-2 flex flex-col gap-6">
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-[#161b22] p-5 rounded-lg border border-gray-800">
             <h2 className="text-sm font-medium text-gray-400 mb-2">Anomaly Timing</h2>
             <p className="text-2xl font-semibold">Current Anomaly: <span className="text-white">23s</span></p>
          </div>
          <div className="bg-[#161b22] p-5 rounded-lg border border-gray-800">
             <h2 className="text-sm font-medium text-gray-400 mb-2">Detailed Anomaly Breakdown</h2>
          </div>
        </div>
        
        <div className="bg-[#161b22] p-5 rounded-lg border border-gray-800 flex-1">
          <h2 className="text-sm font-medium text-gray-400 mb-4">Recent Anomalies Log</h2>
        </div>
      </div>
    </div>
  );
}