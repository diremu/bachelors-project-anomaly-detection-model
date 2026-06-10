import React, { useState } from 'react';

export const LiveFeeds: React.FC = () => {
  const [feedDirectory, setFeedDirectory] = useState('/mnt/surveillance/cam_group_alpha');
  const [isConnecting, setIsConnecting] = useState(false);

  const handleConnect = () => {
    setIsConnecting(true);
    setTimeout(() => setIsConnecting(false), 1500); // Mock connection delay
  };

  return (
    <div className="flex flex-col h-full font-poppins gap-6">
      <div>
        <h1 className="text-2xl font-semibold text-white tracking-wide">Real-Time Feeds <span className="text-gray-500 font-light text-lg">| Model Ingest</span></h1>
        <p className="text-sm text-gray-400 mt-1">Configure the directory paths for the live video anomaly detection stream.</p>
      </div>

      <div className="bg-[#161b22] p-6 rounded-xl border border-gray-800/60 shadow-lg max-w-2xl">
        <h2 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
          <span className="text-teal-500">🔌</span> Feed Directory Configuration
        </h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1 uppercase tracking-wider">Active Stream Path</label>
            <div className="flex gap-3">
              <input 
                type="text" 
                value={feedDirectory}
                onChange={(e) => setFeedDirectory(e.target.value)}
                className="flex-1 px-4 py-2 bg-[#0d1117] border border-gray-700 rounded text-gray-300 focus:outline-none focus:border-teal-500 transition-colors font-mono text-sm"
                placeholder="/path/to/video/stream"
              />
              <button 
                onClick={handleConnect}
                disabled={isConnecting}
                className="px-6 py-2 bg-teal-600/20 text-teal-500 hover:bg-teal-600 hover:text-white border border-teal-600/30 rounded font-medium transition-all text-sm disabled:opacity-50"
              >
                {isConnecting ? 'Connecting...' : 'Mount Directory'}
              </button>
            </div>
          </div>
        </div>

        <div className="mt-8 p-4 bg-[#0d1117]/50 border border-gray-800 rounded-lg">
          <h3 className="text-xs font-medium text-gray-400 mb-2">Connection Status</h3>
          <div className="flex items-center gap-3 text-sm text-gray-500">
            <span className={`w-2 h-2 rounded-full ${isConnecting ? 'bg-amber-500 animate-pulse' : 'bg-gray-600'}`}></span>
            {isConnecting ? 'Establishing secure pipeline...' : 'Awaiting directory mount. No active streams processing.'}
          </div>
        </div>
      </div>
    </div>
  );
};