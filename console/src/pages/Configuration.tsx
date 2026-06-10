import React, { useState } from 'react';

export const Configuration: React.FC = () => {
  const [config, setConfig] = useState({
    trainDir: '',
    testDir: '',
    retentionDays: '30'
  });

  const [saved, setSaved] = useState(false);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  const handleTrainDirSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setConfig({...config, trainDir: files[0].webkitRelativePath || files[0].name});
    }
  };

  const handleTestDirSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setConfig({...config, testDir: files[0].webkitRelativePath || files[0].name});
    }
  };

  return (
    <div className="flex flex-col h-full font-poppins gap-6 max-w-3xl">
      <div>
        <h1 className="text-2xl font-semibold text-white tracking-wide">Configuration <span className="text-gray-500 font-light text-lg">| Storage</span></h1>
        <p className="text-sm text-gray-400 mt-1">Manage machine learning dataset locations and system parameters.</p>
      </div>

      <form onSubmit={handleSave} className="bg-[#161b22] p-6 rounded-xl border border-gray-800/60 shadow-lg space-y-6">
        
        <div className="space-y-4">
          <h2 className="text-sm font-medium text-gray-300 border-b border-gray-800 pb-2">Dataset Directories</h2>
          
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1 uppercase tracking-wider">Training Directory</label>
            <input 
              type="file" 
              onChange={handleTrainDirSelect}
              className="w-full px-4 py-2 bg-[#0d1117] border border-gray-700 rounded text-gray-300 focus:outline-none focus:border-teal-500 transition-colors font-mono text-sm"
              {...({ webkitdirectory: true } as any)}
            />
            {config.trainDir && <p className="text-xs text-gray-500 mt-1">Selected: {config.trainDir}</p>}
            <p className="text-xs text-gray-500 mt-1">Path to historical video segments used for retraining the anomaly detection weights.</p>
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1 uppercase tracking-wider">Testing Directory</label>
            <input 
              type="file" 
              onChange={handleTestDirSelect}
              className="w-full px-4 py-2 bg-[#0d1117] border border-gray-700 rounded text-gray-300 focus:outline-none focus:border-teal-500 transition-colors font-mono text-sm"
              {...({ webkitdirectory: true } as any)}
            />
            {config.testDir && <p className="text-xs text-gray-500 mt-1">Selected: {config.testDir}</p>}
          </div>
        </div>

        <div className="space-y-4 pt-4 border-t border-gray-800">
          <h2 className="text-sm font-medium text-gray-300 border-b border-gray-800 pb-2">System Parameters</h2>
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-1 uppercase tracking-wider">Log Retention (Days)</label>
            <input 
              type="number" 
              value={config.retentionDays}
              onChange={(e) => setConfig({...config, retentionDays: e.target.value})}
              className="w-32 px-4 py-2 bg-[#0d1117] border border-gray-700 rounded text-gray-300 focus:outline-none focus:border-teal-500 transition-colors font-mono text-sm"
            />
          </div>
        </div>

        <div className="pt-4 flex items-center gap-4">
          <button 
            type="submit"
            className="px-6 py-2 bg-teal-600 hover:bg-teal-500 text-white rounded font-medium transition-colors text-sm"
          >
            Save Configuration
          </button>
          {saved && <span className="text-sm text-green-400">Settings saved successfully.</span>}
        </div>
      </form>
    </div>
  );
};