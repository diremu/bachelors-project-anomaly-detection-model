import { useEffect, useState, type ReactNode } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Cell, Area, AreaChart,
  ReferenceLine,
} from 'recharts';

type MetricsData = {
  model: {
    name: string;
    lstm_hidden: number;
    lstm_layers: number;
    unfreeze_layers: number;
    temporal_weight: number;
    training_epochs: number;
    best_val_loss: number;
  };
  training_history: {
    train_losses: number[];
    val_losses: number[];
  };
  evaluation: {
    auc_roc: number | null;
    optimal_threshold: number | null;
    eer: number | null;
    pr_auc: number | null;
  };
  confusion_matrix: {
    tn: number; fp: number; fn: number; tp: number;
  } | null;
  roc_curve: {
    fpr: number[];
    tpr: number[];
  } | null;
  per_category: Record<string, {
    count: number;
    mean_score: number;
    label: string;
    auc?: number;
  }>;
};

const MetricCard = ({ label, value, sub, accent }: {
  label: string; value: string; sub?: string; accent?: boolean;
}) => (
  <div className="bg-[#161b22] p-5 rounded-xl border border-gray-800/60 shadow-lg flex flex-col gap-1">
    <span className="text-xs text-gray-500 uppercase tracking-wider font-medium">{label}</span>
    <span className={`text-3xl font-bold tracking-tight ${accent ? 'text-teal-400' : 'text-white'}`}>{value}</span>
    {sub && <span className="text-xs text-gray-500">{sub}</span>}
  </div>
);

const SectionTitle = ({ children }: { children: React.ReactNode }) => (
  <h2 className="text-sm font-semibold text-gray-300 tracking-wide uppercase mb-4">{children}</h2>
);

export const ModelPerformance: React.FC = () => {
  const [data, setData] = useState<MetricsData | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch('/model-metrics.json')
      .then(r => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch(() => setError('Could not load model-metrics.json. Run the notebook export cell and place the file in public/.'));
  }, []);

  if (error) {
    return (
      <div className="flex flex-col h-full font-poppins items-center justify-center gap-4">
        <div className="bg-[#161b22] p-8 rounded-xl border border-gray-800/60 max-w-lg text-center">
          <p className="text-gray-400 text-sm">{error}</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex h-full items-center justify-center">
        <span className="text-gray-500 text-sm animate-pulse">Loading metrics…</span>
      </div>
    );
  }

  const { model, training_history, evaluation, confusion_matrix, roc_curve, per_category } = data;

  // Derived data
  const trainingData = training_history.train_losses.map((tl, i) => ({
    epoch: i + 1,
    train: tl,
    val: training_history.val_losses[i],
  }));

  const rocData = roc_curve
    ? roc_curve.fpr.map((f, i) => ({ fpr: f, tpr: roc_curve.tpr[i] }))
    : [];

  const categoryData = Object.entries(per_category)
    .map(([name, d]) => ({ name: name.replace(/_/g, ' '), ...d }))
    .sort((a, b) => b.mean_score - a.mean_score);

  const cm = confusion_matrix;
  const precision = cm ? cm.tp / (cm.tp + cm.fp) : 0;
  const recall = cm ? cm.tp / (cm.tp + cm.fn) : 0;
  const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
  const accuracy = cm ? (cm.tp + cm.tn) / (cm.tp + cm.tn + cm.fp + cm.fn) : 0;

  const cmTotal = cm ? cm.tp + cm.tn + cm.fp + cm.fn : 1;

  const formatTooltipValue = (value: string | number | readonly (string | number)[] | undefined, digits = 4) =>
    typeof value === 'number' ? value.toFixed(digits) : String(value ?? '');

  const tooltipFormatter = (
    value: string | number | readonly (string | number)[] | undefined,
    _name: string | number | undefined,
    _item: unknown,
    _index: number,
    _payload: readonly unknown[],
  ): ReactNode => formatTooltipValue(value, 4);

  const tooltipFormatter6 = (
    value: string | number | readonly (string | number)[] | undefined,
    _name: string | number | undefined,
    _item: unknown,
    _index: number,
    _payload: readonly unknown[],
  ): ReactNode => formatTooltipValue(value, 6);

  const tooltipLabelFormatter = (label: ReactNode, _payload: readonly unknown[]): ReactNode =>
    String(label ?? '');

  const tooltipStyle = {
    contentStyle: { background: '#161b22', border: '1px solid #30363d', borderRadius: 8, fontSize: 12 },
    labelStyle: { color: '#8b949e' },
  };

  return (
    <div className="flex flex-col h-full font-poppins gap-6 overflow-y-auto pb-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold text-white tracking-wide">
          Model Performance <span className="text-gray-500 font-light text-lg">| Evaluation Report</span>
        </h1>
        <p className="text-sm text-gray-400 mt-1">
          {model.name} — {model.training_epochs} epochs, hidden={model.lstm_hidden}, layers={model.lstm_layers}
        </p>
      </div>

      {/* Top metric cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="AUC-ROC"
          value={evaluation.auc_roc?.toFixed(4) ?? '—'}
          sub="Area Under ROC Curve"
          accent
        />
        <MetricCard
          label="PR-AUC"
          value={evaluation.pr_auc?.toFixed(4) ?? '—'}
          sub="Precision-Recall AUC"
        />
        <MetricCard
          label="Equal Error Rate"
          value={evaluation.eer ? `${(evaluation.eer * 100).toFixed(1)}%` : '—'}
          sub="FPR = FNR operating point"
        />
        <MetricCard
          label="F1 Score"
          value={f1.toFixed(3)}
          sub={`Precision ${precision.toFixed(3)} · Recall ${recall.toFixed(3)}`}
        />
      </div>

      {/* ROC Curve + Training Loss */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ROC Curve */}
        <div className="bg-[#161b22] p-5 rounded-xl border border-gray-800/60 shadow-lg">
          <SectionTitle>ROC Curve</SectionTitle>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={rocData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
              <XAxis
                dataKey="fpr" type="number" domain={[0, 1]}
                tick={{ fill: '#8b949e', fontSize: 11 }}
                label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -2, fill: '#8b949e', fontSize: 11 }}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fill: '#8b949e', fontSize: 11 }}
                label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fill: '#8b949e', fontSize: 11 }}
              />
              <Tooltip
                {...tooltipStyle}
                formatter={tooltipFormatter}
              />
              <ReferenceLine
                segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                stroke="#30363d" strokeDasharray="4 4"
              />
              <Area
                type="monotone" dataKey="tpr"
                stroke="#2dd4bf" fill="#2dd4bf" fillOpacity={0.08} strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-500 text-center mt-2">
            AUC = {evaluation.auc_roc?.toFixed(4) ?? '—'} · Threshold = {evaluation.optimal_threshold?.toFixed(6) ?? '—'}
          </p>
        </div>

        {/* Training Loss */}
        <div className="bg-[#161b22] p-5 rounded-xl border border-gray-800/60 shadow-lg">
          <SectionTitle>Training History</SectionTitle>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={trainingData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
              <XAxis
                dataKey="epoch"
                tick={{ fill: '#8b949e', fontSize: 11 }}
                label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#8b949e', fontSize: 11 }}
              />
              <YAxis
                tick={{ fill: '#8b949e', fontSize: 11 }}
                label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: '#8b949e', fontSize: 11 }}
              />
              <Tooltip
                {...tooltipStyle}
                formatter={tooltipFormatter6}
              />
              <Line type="monotone" dataKey="train" stroke="#3b82f6" strokeWidth={2} dot={false} name="Train" />
              <Line type="monotone" dataKey="val" stroke="#ef4444" strokeWidth={2} dot={false} name="Val" />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex justify-center gap-6 mt-2 text-xs text-gray-500">
            <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-blue-500 inline-block rounded" /> Train</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-red-500 inline-block rounded" /> Validation</span>
          </div>
        </div>
      </div>

      {/* Confusion Matrix + Per-Category Scores */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Confusion Matrix */}
        {cm && (
          <div className="bg-[#161b22] p-5 rounded-xl border border-gray-800/60 shadow-lg">
            <SectionTitle>Confusion Matrix</SectionTitle>
            <div className="flex items-center justify-center">
              <div className="relative">
                {/* Axis labels */}
                <div className="absolute -left-16 top-1/2 -translate-y-1/2 -rotate-90 text-xs text-gray-500 whitespace-nowrap tracking-wider">
                  ACTUAL
                </div>
                <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-xs text-gray-500 tracking-wider">
                  PREDICTED
                </div>

                {/* Matrix grid */}
                <div className="grid grid-cols-[auto_1fr_1fr] grid-rows-[auto_1fr_1fr] gap-0.5 ml-4">
                  {/* Header row */}
                  <div />
                  <div className="text-center text-xs text-gray-500 pb-2 px-4">Normal</div>
                  <div className="text-center text-xs text-gray-500 pb-2 px-4">Anomalous</div>

                  {/* Row 1: Actual Normal */}
                  <div className="text-xs text-gray-500 pr-3 flex items-center justify-end">Normal</div>
                  <div
                    className="w-28 h-24 rounded-lg flex flex-col items-center justify-center gap-1 transition-colors"
                    style={{ backgroundColor: `rgba(45, 212, 191, ${0.1 + (cm.tn / cmTotal) * 0.6})` }}
                  >
                    <span className="text-2xl font-bold text-white">{cm.tn}</span>
                    <span className="text-[10px] text-gray-400">TN</span>
                  </div>
                  <div  
                    className="w-28 h-24 rounded-lg flex flex-col items-center justify-center gap-1 transition-colors"
                    style={{ backgroundColor: `rgba(239, 68, 68, ${0.1 + (cm.fp / cmTotal) * 0.6})` }}
                  >
                    <span className="text-2xl font-bold text-white">{cm.fp}</span>
                    <span className="text-[10px] text-gray-400">FP</span>
                  </div>

                  {/* Row 2: Actual Anomalous */}
                  <div className="text-xs text-gray-500 pr-3 flex items-center justify-end">Anomalous</div>
                  <div
                    className="w-28 h-24 rounded-lg flex flex-col items-center justify-center gap-1 transition-colors"
                    style={{ backgroundColor: `rgba(239, 68, 68, ${0.1 + (cm.fn / cmTotal) * 0.6})` }}
                  >
                    <span className="text-2xl font-bold text-white">{cm.fn}</span>
                    <span className="text-[10px] text-gray-400">FN</span>
                  </div>
                  <div
                    className="w-28 h-24 rounded-lg flex flex-col items-center justify-center gap-1 transition-colors"
                    style={{ backgroundColor: `rgba(45, 212, 191, ${0.1 + (cm.tp / cmTotal) * 0.6})` }}
                  >
                    <span className="text-2xl font-bold text-white">{cm.tp}</span>
                    <span className="text-[10px] text-gray-400">TP</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Secondary metrics below matrix */}
            <div className="grid grid-cols-4 gap-3 mt-8">
              {[
                ['Accuracy', accuracy],
                ['Precision', precision],
                ['Recall', recall],
                ['F1', f1],
              ].map(([label, val]) => (
                <div key={label as string} className="text-center">
                  <div className="text-lg font-semibold text-white">{(val as number).toFixed(3)}</div>
                  <div className="text-[10px] text-gray-500 uppercase tracking-wider">{label as string}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Per-Category Scores */}
        <div className="bg-[#161b22] p-5 rounded-xl border border-gray-800/60 shadow-lg">
          <SectionTitle>Mean Anomaly Score by Category</SectionTitle>
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={categoryData} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#21262d" horizontal={false} />
              <XAxis
                type="number"
                tick={{ fill: '#8b949e', fontSize: 10 }}
                tickFormatter={(v: any) => String(v ?? '')}
              />
              <YAxis
                dataKey="name" type="category"
                tick={{ fill: '#8b949e', fontSize: 10 }}
                width={130}
              />
              <Tooltip
                {...tooltipStyle}
                formatter={tooltipFormatter6}
                labelFormatter={tooltipLabelFormatter}
              />
              {evaluation.optimal_threshold && (
                <ReferenceLine
                  x={evaluation.optimal_threshold}
                  stroke="#ef4444" strokeDasharray="4 4"
                  label={{ value: 'Threshold', fill: '#ef4444', fontSize: 10, position: 'top' }}
                />
              )}
              <Bar dataKey="mean_score" radius={[0, 4, 4, 0]} maxBarSize={20}>
                {categoryData.map((entry, idx) => (
                  <Cell
                    key={idx}
                    fill={entry.label === 'normal' ? '#2dd4bf' : '#f87171'}
                    fillOpacity={0.7}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="flex justify-center gap-6 mt-2 text-xs text-gray-500">
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 bg-teal-400 rounded-sm inline-block opacity-70" /> Normal</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 bg-red-400 rounded-sm inline-block opacity-70" /> Anomalous</span>
          </div>
        </div>
      </div>

      {/* Model Configuration */}
      <div className="bg-[#161b22] p-5 rounded-xl border border-gray-800/60 shadow-lg">
        <SectionTitle>Model Configuration</SectionTitle>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
          {[
            ['Architecture', model.name.split('+')[1]?.trim() ?? model.name],
            ['LSTM Hidden', model.lstm_hidden],
            ['LSTM Layers', model.lstm_layers],
            ['Unfreeze Layers', model.unfreeze_layers],
            ['Temporal Weight', model.temporal_weight],
            ['Epochs Trained', model.training_epochs],
            ['Best Val Loss', model.best_val_loss.toFixed(6)],
          ].map(([label, val]) => (
            <div key={label as string} className="bg-[#0d1117] p-3 rounded-lg border border-gray-800/50">
              <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{label as string}</div>
              <div className="text-sm font-medium text-gray-200 font-mono">{String(val)}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Data notice */}
      <p className="text-xs text-gray-600 text-center">
        Metrics loaded from model-metrics.json · Replace with notebook export after training
      </p>
    </div>
  );
};