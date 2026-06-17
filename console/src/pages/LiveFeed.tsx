import { useState, useRef, useCallback, useEffect, type ReactNode } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { useAppDispatch } from '../store';
import { receiveNewAnomaly } from '../store/dashboardSlice';

const MOCK_THRESHOLD = 0.004218;

type ScorePoint = {
  idx: number;
  score: number;
  threshold: number;
  anomaly: boolean;
};

type AlertEntry = {
  frame: number;
  score: number;
  time: string;
};

/** Generate a plausible anomaly score.
 *  Normal clips hover around 0.001-0.003; anomalous clips spike to 0.005-0.012.
 *  The `anomalyChance` controls how often spikes occur.
 */
function mockScore(prev: number, anomalyChance = 0.12): number {
  const isSpike = Math.random() < anomalyChance;
  if (isSpike) {
    return 0.005 + Math.random() * 0.007;            // 0.005 – 0.012
  }
  // Brownian-ish walk around normal baseline
  const drift = (Math.random() - 0.52) * 0.0008;
  return Math.max(0.0005, Math.min(0.004, prev + drift));
}

/** Smooth over a sliding window. */
function smooth(buf: number[], n = 5): number {
  const tail = buf.slice(-n);
  return tail.reduce((a, b) => a + b, 0) / tail.length;
}

export const LiveFeeds = () => {
  const dispatch = useAppDispatch();

  // State
  const [mode, setMode] = useState<'stream' | 'upload'>('upload');
  const [isStreaming, setIsStreaming] = useState(false);
  const [scores, setScores] = useState<ScorePoint[]>([]);
  const [alerts, setAlerts] = useState<AlertEntry[]>([]);
  const [videoSrc, setVideoSrc] = useState<string | null>(null);

  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const rawScoresRef = useRef<number[]>([]);
  const idxRef = useRef(0);

  // ── Video file selection ──
  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setVideoSrc(URL.createObjectURL(file));
    setScores([]);
    setAlerts([]);
    rawScoresRef.current = [];
    idxRef.current = 0;
  };

  // ── Mock inference tick ──
  const tick = useCallback(() => {
    const prev = rawScoresRef.current.length > 0
      ? rawScoresRef.current[rawScoresRef.current.length - 1]
      : 0.002;

    const raw = mockScore(prev);
    rawScoresRef.current.push(raw);
    const smoothed = smooth(rawScoresRef.current);
    const isAnomaly = smoothed > MOCK_THRESHOLD;
    const idx = idxRef.current++;

    const pt: ScorePoint = { idx, score: smoothed, threshold: MOCK_THRESHOLD, anomaly: isAnomaly };
    setScores(s => [...s.slice(-200), pt]);

    if (isAnomaly) {
      const alert: AlertEntry = {
        frame: idx * 8,    // stride = 8
        score: smoothed,
        time: new Date().toLocaleTimeString(),
      };
      setAlerts(a => [alert, ...a].slice(0, 50));
      dispatch(receiveNewAnomaly({
        id: `live_${Date.now()}`,
        timestamp: alert.time,
        location: 'Live Feed',
        inmateId: '—',
        anomalyType: 'Anomalous Activity',
        duration: `frame ${alert.frame}`,
        verification: 'Pending',
        severity: smoothed > MOCK_THRESHOLD * 2 ? 'critical' : 'warning',
      }));
    }
  }, [dispatch]);

  // ── Start / stop ──
  const startStream = useCallback(() => {
    videoRef.current?.play();
    intervalRef.current = setInterval(tick, 250);  // ~4 scores/sec for visual pacing
    setIsStreaming(true);
  }, [tick]);

  const stopStream = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
    videoRef.current?.pause();
    setIsStreaming(false);
  }, []);

  // ── Upload mode: instant batch results ──
  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    rawScoresRef.current = [];
    idxRef.current = 0;

    // Simulate 40-80 clips worth of scores
    const nClips = 40 + Math.floor(Math.random() * 40);
    const pts: ScorePoint[] = [];
    const newAlerts: AlertEntry[] = [];
    let prev = 0.002;

    for (let i = 0; i < nClips; i++) {
      const raw = mockScore(prev);
      rawScoresRef.current.push(raw);
      const smoothed = smooth(rawScoresRef.current);
      const isAnomaly = smoothed > MOCK_THRESHOLD;
      prev = raw;

      pts.push({ idx: i, score: smoothed, threshold: MOCK_THRESHOLD, anomaly: isAnomaly });

      if (isAnomaly) {
        newAlerts.push({
          frame: i * 8,
          score: smoothed,
          time: new Date().toLocaleTimeString(),
        });
      }
    }

    setScores(pts);
    setAlerts(newAlerts.slice(0, 50));

    // Dispatch top 5 to Redux
    newAlerts.slice(0, 5).forEach(a => {
      dispatch(receiveNewAnomaly({
        id: `upload_${Date.now()}_${a.frame}`,
        timestamp: a.time,
        location: 'Uploaded Video',
        inmateId: '—',
        anomalyType: 'Anomalous Activity',
        duration: `frame ${a.frame}`,
        verification: 'Pending',
        severity: a.score > MOCK_THRESHOLD * 2 ? 'critical' : 'warning',
      }));
    });
  };

  // Cleanup
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const tooltipFormatter = (
    value: string | number | readonly (string | number)[] | undefined,
    _name: string | number | undefined,
    _item: unknown,
    _index: number,
    _payload: readonly unknown[],
  ): ReactNode => typeof value === 'number' ? value.toFixed(6) : String(value ?? '');

  const tooltipStyle = {
    contentStyle: { background: '#161b22', border: '1px solid #30363d', borderRadius: 8, fontSize: 12 },
    labelStyle: { color: '#8b949e' },
  };

  return (
    <div className="flex flex-col h-full font-poppins gap-6 overflow-y-auto pb-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold text-white tracking-wide">
          Real-Time Feeds <span className="text-gray-500 font-light text-lg">| Model Inference</span>
        </h1>
        <p className="text-sm text-gray-400 mt-1">
          Stream video frames through the anomaly detection model or upload a clip for batch scoring.
        </p>
      </div>

      {/* Control bar */}
      <div className="bg-[#161b22] p-4 rounded-xl border border-gray-800/60 shadow-lg flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <span className={`w-2.5 h-2.5 rounded-full ${isStreaming ? 'bg-green-500 animate-pulse' : 'bg-gray-600'}`} />
          <span className="text-xs text-gray-400 uppercase tracking-wider">
            {isStreaming ? 'Streaming' : 'Idle'}
          </span>
        </div>

        <div className="flex-1" />

        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span>Threshold:</span>
          <span className="font-mono text-gray-300">{MOCK_THRESHOLD.toFixed(6)}</span>
        </div>

        {/* Mode toggle */}
        <div className="flex bg-[#0d1117] rounded border border-gray-700 overflow-hidden">
          <button
            onClick={() => { stopStream(); setMode('upload'); }}
            className={`px-3 py-2 text-xs font-medium transition-colors ${mode === 'upload' ? 'bg-teal-600 text-white' : 'text-gray-400 hover:text-white'}`}
          >
            Upload
          </button>
          <button
            onClick={() => setMode('stream')}
            className={`px-3 py-2 text-xs font-medium transition-colors ${mode === 'stream' ? 'bg-teal-600 text-white' : 'text-gray-400 hover:text-white'}`}
          >
            Stream
          </button>
        </div>
      </div>

      {/* Main content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-0">

        {/* Left: Video / Upload */}
        <div className="bg-[#161b22] p-5 rounded-xl border border-gray-800/60 shadow-lg flex flex-col gap-4">
          {mode === 'stream' ? (
            <>
              <h2 className="text-sm font-semibold text-gray-300 tracking-wide uppercase">Video Stream</h2>
              <div className="bg-[#0d1117] rounded-lg border border-gray-800 overflow-hidden aspect-video flex items-center justify-center">
                {videoSrc ? (
                  <video ref={videoRef} src={videoSrc} className="w-full h-full object-cover" muted playsInline />
                ) : (
                  <span className="text-xs text-gray-600">Select a video to begin</span>
                )}
              </div>

              <div className="space-y-3">
                <div>
                  <label className="block text-xs text-gray-500 uppercase tracking-wider mb-1">Source Video</label>
                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleFile}
                    className="w-full text-sm text-gray-400 file:mr-3 file:py-1.5 file:px-3 file:rounded file:border-0 file:text-xs file:font-medium file:bg-teal-600/20 file:text-teal-400 hover:file:bg-teal-600/30"
                  />
                </div>

                <button
                  onClick={isStreaming ? stopStream : startStream}
                  disabled={!videoSrc}
                  className={`w-full py-2 rounded text-sm font-medium transition-colors disabled:opacity-30 disabled:cursor-not-allowed ${
                    isStreaming
                      ? 'bg-red-600/20 text-red-400 border border-red-600/30 hover:bg-red-600 hover:text-white'
                      : 'bg-teal-600/20 text-teal-500 border border-teal-600/30 hover:bg-teal-600 hover:text-white'
                  }`}
                >
                  {isStreaming ? 'Stop Streaming' : 'Start Streaming'}
                </button>
              </div>
            </>
          ) : (
            <>
              <h2 className="text-sm font-semibold text-gray-300 tracking-wide uppercase">Upload Video</h2>
              <div className="flex-1 flex flex-col items-center justify-center gap-4 bg-[#0d1117] rounded-lg border border-dashed border-gray-700 p-6">
                <svg className="w-10 h-10 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="text-sm text-gray-500">Upload a video clip for batch scoring</p>
                <input
                  type="file"
                  accept="video/*"
                  onChange={handleUpload}
                  className="text-sm text-gray-400 file:mr-3 file:py-1.5 file:px-3 file:rounded file:border-0 file:text-xs file:font-medium file:bg-teal-600/20 file:text-teal-400 hover:file:bg-teal-600/30"
                />
              </div>

              {scores.length > 0 && (
                <div className="bg-[#0d1117] p-3 rounded-lg border border-gray-800 space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-500">Clips scored</span>
                    <span className="text-gray-300">{scores.length}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-500">Anomalous</span>
                    <span className={alerts.length > 0 ? 'text-red-400 font-medium' : 'text-green-400'}>
                      {alerts.length}
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-500">Peak score</span>
                    <span className="text-gray-300 font-mono">{Math.max(...scores.map(s => s.score)).toFixed(6)}</span>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Right: Score chart + alerts */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          {/* Score timeline */}
          <div className="bg-[#161b22] p-5 rounded-xl border border-gray-800/60 shadow-lg flex-1 min-h-[300px]">
            <h2 className="text-sm font-semibold text-gray-300 tracking-wide uppercase mb-4">
              Anomaly Score Timeline
              {scores.length > 0 && (
                <span className="text-gray-600 font-normal ml-2">({scores.length} clips)</span>
              )}
            </h2>

            {scores.length === 0 ? (
              <div className="flex items-center justify-center h-[240px] text-gray-600 text-sm">
                {mode === 'stream' ? 'Start streaming to see scores' : 'Upload a video to see scores'}
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={scores}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                  <XAxis
                    dataKey="idx"
                    tick={{ fill: '#8b949e', fontSize: 10 }}
                    label={{ value: 'Clip', position: 'insideBottom', offset: -2, fill: '#8b949e', fontSize: 11 }}
                  />
                  <YAxis
                    tick={{ fill: '#8b949e', fontSize: 10 }}
                    tickFormatter={(v: number) => v.toFixed(4)}
                  />
                  <Tooltip
                    {...tooltipStyle}
                    formatter={tooltipFormatter}
                  />
                  <ReferenceLine
                    y={MOCK_THRESHOLD}
                    stroke="#ef4444" strokeDasharray="4 4"
                    label={{ value: 'Threshold', fill: '#ef4444', fontSize: 10, position: 'right' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="score"
                    stroke="#2dd4bf"
                    strokeWidth={1.5}
                    dot={false}
                    isAnimationActive={mode === 'upload'}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Alerts */}
          <div className="bg-[#161b22] p-5 rounded-xl border border-gray-800/60 shadow-lg max-h-[240px] overflow-y-auto">
            <h2 className="text-sm font-semibold text-gray-300 tracking-wide uppercase mb-3">
              Session Alerts
              {alerts.length > 0 && (
                <span className="ml-2 px-2 py-0.5 bg-red-500/10 text-red-400 text-xs rounded-full">{alerts.length}</span>
              )}
            </h2>

            {alerts.length === 0 ? (
              <p className="text-xs text-gray-600">No anomalies detected yet.</p>
            ) : (
              <div className="space-y-2">
                {alerts.map((a, i) => (
                  <div key={i} className="flex items-center justify-between bg-[#0d1117] px-3 py-2 rounded border border-red-900/20">
                    <div className="flex items-center gap-3">
                      <span className="w-2 h-2 bg-red-500 rounded-full" />
                      <span className="text-xs text-gray-400 font-mono">{a.time}</span>
                      <span className="text-xs text-gray-300">Frame {a.frame}</span>
                    </div>
                    <span className="text-xs text-red-400 font-mono">{a.score.toFixed(6)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};