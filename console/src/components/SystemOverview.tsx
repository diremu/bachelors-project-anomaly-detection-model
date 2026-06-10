import { TopMetrics } from './TopMetrics';
import RecentAnomaliesLog from './AnomaliesLog';

export default function SystemOverview() {
  return (
    <div className="flex flex-col h-full gap-6">
      <TopMetrics />
      <RecentAnomaliesLog />
    </div>
  );
}