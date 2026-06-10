import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { DashboardState, Anomaly, VerificationStatus } from '../types';

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
  recentAnomalies: [
    { id: 'evt_001', timestamp: '10:14:29 AM', location: 'Block A - Perimeter Fence 3', inmateId: '#451', anomalyType: 'Loitering', duration: '15s', verification: 'Verified', severity: 'warning' },
    { id: 'evt_002', timestamp: '10:13:42 AM', location: 'Yard 1 - Blind Spot', inmateId: '#203', anomalyType: 'Trajectory Deviation', duration: '8s', verification: 'Pending', severity: 'warning' },
    { id: 'evt_003', timestamp: '10:12:05 AM', location: 'Block B - Access Corridor', inmateId: '#551', anomalyType: 'Boundary Breach', duration: '45s', verification: 'Verified', severity: 'critical' },
  ]
};

const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    // Fired by your WebSocket when a new VAD inference arrives
    receiveNewAnomaly: (state, action) => {
      // 1. Add to the top of the log
      state.recentAnomalies.unshift(action.payload);
      // Keep log bounded to 50 items to prevent DOM bloat
      if (state.recentAnomalies.length > 50) state.recentAnomalies.pop();

      // 2. Update global metrics
      state.metrics.totalEvents += 1;
      state.metrics.anomaliesIdentified += 1;
      
      // 3. Update specific breakdown
      const typeMap = {
        'Loitering': 'loitering',
        'Abnormal Aggregation': 'unusualAggregation',
        'Boundary Breach': 'boundaryBreach'
      };
      const key = typeMap[action.payload.anomalyType];
      if (key) state.breakdown[key] += 1;

      // 4. Reset the live timer
      state.currentAnomalyTimer = 0;
    },
    tickTimer: (state) => {
      state.currentAnomalyTimer += 1;
    },
    updateVerification: (state, action) => {
      const { id, status } = action.payload;
      const anomaly = state.recentAnomalies.find(a => a.id === id);
      if (anomaly) anomaly.verification = status;
    }
  }
});

export const { receiveNewAnomaly, tickTimer, updateVerification } = dashboardSlice.actions;
export default dashboardSlice.reducer;