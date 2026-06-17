import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import type { DashboardState, Anomaly, VerificationStatus, AnomalyBreakdown } from '../types';

const initialState: DashboardState = {
  metrics: {
    accurateDetections: 96.5,
    falsePositives: 2.1,
    falseNegatives: 1.4,
    totalEvents: 12345,
    anomaliesIdentified: 215,
  },
  breakdown: {
    loitering: 88,
    unusualAggregation: 62,
    boundaryBreach: 45,
    systemOffline: 20,
  },
  currentAnomalyTimer: 23,
  isTimerActive: true,
  recentAnomalies: [
    { id: 'evt_001', timestamp: '10:14:29 AM', location: 'Block A - Exercise Yard', inmateId: '#451', anomalyType: 'Fighting', duration: '15s', verification: 'Verified', severity: 'critical' },
    { id: 'evt_002', timestamp: '10:13:42 AM', location: 'Block B - Corridor 2', inmateId: '#203', anomalyType: 'Assault', duration: '8s', verification: 'Pending', severity: 'critical' },
    { id: 'evt_003', timestamp: '10:12:05 AM', location: 'Block C - Mess Hall', inmateId: '#551', anomalyType: 'Stealing', duration: '45s', verification: 'Verified', severity: 'warning' },
    { id: 'evt_004', timestamp: '10:08:11 AM', location: 'Block A - Cell Wing 3', inmateId: '#118', anomalyType: 'Abuse', duration: '22s', verification: 'Pending', severity: 'critical' },
    { id: 'evt_005', timestamp: '10:04:53 AM', location: 'Perimeter - Gate 2', inmateId: '#309', anomalyType: 'Arrest', duration: '12s', verification: 'Verified', severity: 'warning' },
  ]
};

const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    receiveNewAnomaly: (state, action: PayloadAction<Anomaly>) => {
      state.recentAnomalies.unshift(action.payload);
      if (state.recentAnomalies.length > 50) state.recentAnomalies.pop();
      state.metrics.totalEvents += 1;
      state.metrics.anomaliesIdentified += 1;
      state.currentAnomalyTimer = 0;
      state.isTimerActive = true;

      const typeMap: Record<string, keyof AnomalyBreakdown> = {
        'Loitering': 'loitering',
        'Abnormal Aggregation': 'unusualAggregation',
        'Boundary Breach': 'boundaryBreach'
      };
      const key = typeMap[action.payload.anomalyType] as keyof AnomalyBreakdown | undefined;
      if (key) state.breakdown[key] += 1;
    },
    tickTimer: (state) => {
      if (state.isTimerActive) {
        state.currentAnomalyTimer += 1;
      }
    },
    resolveActiveAnomaly: (state) => {
      state.isTimerActive = false;
      state.currentAnomalyTimer = 0;
    },
    updateVerification: (state, action: PayloadAction<{ id: string; status: VerificationStatus }>) => {
      const { id, status } = action.payload;
      const anomaly = state.recentAnomalies.find(a => a.id === id);
      if (anomaly) anomaly.verification = status;
    }
  }
});

export const { receiveNewAnomaly, tickTimer, resolveActiveAnomaly, updateVerification } = dashboardSlice.actions;
export default dashboardSlice.reducer;