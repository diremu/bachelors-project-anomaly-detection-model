export type Severity = 'critical' | 'warning' | 'low';
export type VerificationStatus = 'Verified' | 'Pending' | 'Dismissed';

export interface Anomaly {
  id: string;
  timestamp: string;
  location: string;
  inmateId: string;
  anomalyType: string;
  duration: string;
  verification: VerificationStatus;
  severity: Severity;
}

export interface DashboardMetrics {
  accurateDetections: number;
  falsePositives: number;
  falseNegatives: number;
  totalEvents: number;
  anomaliesIdentified: number;
}

export interface AnomalyBreakdown {
  loitering: number;
  unusualAggregation: number;
  boundaryBreach: number;
  systemOffline: number;
}

export interface DashboardState {
  metrics: DashboardMetrics;
  breakdown: AnomalyBreakdown;
  currentAnomalyTimer: number;
  recentAnomalies: Anomaly[];
  isTimerActive: boolean;
}