import { useState, useEffect } from 'react';

type ModelInfo = {
  model: {
    name: string;
    lstm_hidden: number;
    lstm_layers: number;
    unfreeze_layers: number;
    temporal_weight: number;
    training_epochs: number;
    best_val_loss: number;
  };
  evaluation: {
    auc_roc: number | null;
    optimal_threshold: number | null;
    eer: number | null;
    pr_auc: number | null;
  };
} | null;

const ParamRow = ({ label, value }: { label: string; value: string | number }) => (
  <div className="flex justify-between items-center py-2.5 border-b border-gray-800/40 last:border-0">
    <span className="text-sm text-gray-400">{label}</span>
    <span className="text-sm text-gray-200 font-mono">{String(value)}</span>
  </div>
);

export const Configuration = () => {
  const [modelInfo, setModelInfo] = useState<ModelInfo>(null);
  const [config, setConfig] = useState({
    trainDir: '',
    testDir: '',
    retentionDays: '30',
    threshold: '0.004218',
    smoothingWindow: '5',
  });
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    fetch('/model-metrics.json')
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data) {
          setModelInfo(data);
          if (data.evaluation?.optimal_threshold) {
            setConfig(c => ({ ...c, threshold: data.evaluation.optimal_threshold.toFixed(6) }));
          }
        }
      })
      .catch(() => {});
  }, []);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  return (
    <div className="flex flex-col h-full font-poppins gap-6 overflow-y-auto pb-8">
      <div>
        <h1 className="text-2xl font-semibold text-white tracking-wide">
          Configuration <span className="text-gray-500 font-light text-lg">| System Settings</span>
        </h1>
        <p className="text-sm text-gray-400 mt-1">Model parameters, detection thresholds, and storage settings.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Model Architecture */}
        <div className="bg-[#161b22] p-6 rounded-xl border border-gray-800/60 shadow-lg">
          <h2 className="text-sm font-semibold text-gray-300 tracking-wide uppercase mb-4">Model Architecture</h2>
          {modelInfo ? (
            <div className="space-y-0">
              <ParamRow label="Architecture" value={modelInfo.model.name} />
              <ParamRow label="LSTM Hidden Dimension" value={modelInfo.model.lstm_hidden} />
              <ParamRow label="LSTM Layers" value={modelInfo.model.lstm_layers} />
              <ParamRow label="Unfrozen ResNet Layers" value={modelInfo.model.unfreeze_layers} />
              <ParamRow label="Temporal Smoothness Weight" value={modelInfo.model.temporal_weight} />
              <ParamRow label="Training Epochs" value={modelInfo.model.training_epochs} />
              <ParamRow label="Best Validation Loss" value={modelInfo.model.best_val_loss.toFixed(6)} />
            </div>
          ) : (
            <p className="text-sm text-gray-600">No model metrics loaded. Place model-metrics.json in public/.</p>
          )}
        </div>

        {/* Evaluation Metrics */}
        <div className="bg-[#161b22] p-6 rounded-xl border border-gray-800/60 shadow-lg">
          <h2 className="text-sm font-semibold text-gray-300 tracking-wide uppercase mb-4">Evaluation Summary</h2>
          {modelInfo?.evaluation ? (
            <div className="space-y-0">
              <ParamRow label="AUC-ROC" value={modelInfo.evaluation.auc_roc?.toFixed(4) ?? '—'} />
              <ParamRow label="PR-AUC" value={modelInfo.evaluation.pr_auc?.toFixed(4) ?? '—'} />
              <ParamRow label="Equal Error Rate" value={modelInfo.evaluation.eer ? `${(modelInfo.evaluation.eer * 100).toFixed(1)}%` : '—'} />
              <ParamRow label="Optimal Threshold" value={modelInfo.evaluation.optimal_threshold?.toFixed(6) ?? '—'} />
            </div>
          ) : (
            <p className="text-sm text-gray-600">No evaluation data available.</p>
          )}
        </div>
      </div>

      {/* Detection Settings */}
      <form onSubmit={handleSave} className="bg-[#161b22] p-6 rounded-xl border border-gray-800/60 shadow-lg space-y-6">
        <h2 className="text-sm font-semibold text-gray-300 tracking-wide uppercase">Detection Settings</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-2 uppercase tracking-wider">Anomaly Threshold</label>
            <input
              type="number"
              step="0.000001"
              value={config.threshold}
              onChange={(e) => setConfig({ ...config, threshold: e.target.value })}
              className="w-full px-4 py-2 bg-[#0d1117] border border-gray-700 rounded text-gray-300 focus:outline-none focus:border-teal-500 transition-colors font-mono text-sm"
            />
            <p className="text-xs text-gray-600 mt-1">Clips scoring above this value are flagged as anomalous. Lower = more sensitive.</p>
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-500 mb-2 uppercase tracking-wider">Score Smoothing Window</label>
            <input
              type="number"
              min="1"
              max="20"
              value={config.smoothingWindow}
              onChange={(e) => setConfig({ ...config, smoothingWindow: e.target.value })}
              className="w-full px-4 py-2 bg-[#0d1117] border border-gray-700 rounded text-gray-300 focus:outline-none focus:border-teal-500 transition-colors font-mono text-sm"
            />
            <p className="text-xs text-gray-600 mt-1">Number of consecutive clip scores averaged for the smoothed output.</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-gray-800">
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-2 uppercase tracking-wider">Log Retention (Days)</label>
            <input
              type="number"
              value={config.retentionDays}
              onChange={(e) => setConfig({ ...config, retentionDays: e.target.value })}
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