import React from 'react';
import type { ExperimentInfo, ResultsStatus } from '../../types';

interface ExperimentDashboardProps {
  experiments: ExperimentInfo[];
  selectedExperiment: string;
  onSelectExperiment: (expName: string) => void;
  resultsStatus: ResultsStatus;
  onSelectRun: (runId: string) => void;
  onFetchAblationLogs: (category: string) => void;
  onStopTask: (runId: string) => void;
  onRestartTask: (runId: string) => void;
  onFetchLogs: (runId: string) => void;
}

export const ExperimentDashboard: React.FC<ExperimentDashboardProps> = ({
  experiments,
  selectedExperiment,
  onSelectExperiment,
  resultsStatus,
  onSelectRun,
  onFetchAblationLogs,
  onStopTask,
  onRestartTask,
  onFetchLogs
}) => {
  return (
    <div className="experiment-dashboard">
      <h2>Experiment Dashboard</h2>
      <select 
        value={selectedExperiment} 
        onChange={(e) => onSelectExperiment(e.target.value)}
      >
        {experiments.map(exp => (
          <option key={exp.exp_name} value={exp.exp_name}>{exp.exp_name}</option>
        ))}
      </select>

      {selectedExperiment && (
        <div>
          <h3>Runs for {selectedExperiment}</h3>
          {resultsStatus.run_statuses && Object.keys(resultsStatus.run_statuses)
            .filter(runId => runId.startsWith(selectedExperiment))
            .map(runId => (
              <div key={runId}>
                <span>{runId}</span>
                <button onClick={() => onSelectRun(runId)}>View</button>
                <button onClick={() => onFetchAblationLogs(runId)}>Ablation Logs</button>
                <button onClick={() => onStopTask(runId)}>Stop</button>
                <button onClick={() => onRestartTask(runId)}>Restart</button>
                <button onClick={() => onFetchLogs(runId)}>Logs</button>
              </div>
            ))}
        </div>
      )}
    </div>
  );
};