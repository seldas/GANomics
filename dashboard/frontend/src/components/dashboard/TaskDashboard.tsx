import React from 'react';
import { ArrowLeft, RotateCcw, ChevronRight, Info, RefreshCw } from 'lucide-react';
import { StatusButton } from '../common/UIComponents';
import type { RunStatus, Project, LogResponse } from '../../types';

import { LogViewer } from '../analysis/LogViewer';
import { SyncStatusDetails } from '../analysis/SyncStatusDetails';
import { ComparativeAnalysis } from '../analysis/ComparativeAnalysis';
import { DegAnalysis } from '../analysis/DegAnalysis';
import { PathwayAnalysis } from '../analysis/PathwayAnalysis';
import { PredictionAnalysis } from '../analysis/PredictionAnalysis';

interface TaskDashboardProps {
  selectedRunId: string;
  selectedExtId: string | null;
  runStatus: RunStatus | undefined;
  status: any;
  isSizeTask: boolean;
  currentProj: Project | undefined;
  taskView: string;
  onBack: () => void;
  onSetTaskView: (view: any) => void;
  onSetSelectedExtId: (id: string | null) => void;
  onShowSyncModal: () => void;
  onRestartTask: (id: string) => void;
  onRunStep: (step: number) => void;
  fetchLogs: (id: string) => void;
  fetchSyncStatus: (id: string) => void;
  fetchComparativeMetrics: (id: string) => void;
  fetchDegMetrics: (id: string) => void;
  fetchPathwayMetrics: (id: string) => void;
  fetchPredictionMetrics: (id: string) => void;
  logData: LogResponse | null;
  runSyncData: any | null;
  runComparativeData: any[] | null;
  runDegData: any | null;
  runPathwayData: any | null;
  runPredictionData: any | null;
}

export const TaskDashboard: React.FC<TaskDashboardProps> = ({
  selectedRunId, selectedExtId, runStatus, status, isSizeTask, currentProj, taskView,
  onBack, onSetTaskView, onSetSelectedExtId, onShowSyncModal, onRestartTask, onRunStep,
  fetchLogs, fetchSyncStatus, fetchComparativeMetrics, fetchDegMetrics, fetchPathwayMetrics, fetchPredictionMetrics,
  logData, runSyncData, runComparativeData, runDegData, runPathwayData, runPredictionData
}) => {

  if (taskView === 'training') return <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}><button className="chip" onClick={() => onSetTaskView('overview')}><ArrowLeft size={18} /></button><LogViewer logData={logData} runId={selectedRunId} /></div>;
  if (taskView === 'sync') return <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}><button className="chip" onClick={() => onSetTaskView('overview')}><ArrowLeft size={18} /></button><SyncStatusDetails data={runSyncData} /></div>;
  if (taskView === 'comparative') return <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}><button className="chip" onClick={() => onSetTaskView('overview')}><ArrowLeft size={18} /></button><ComparativeAnalysis data={runComparativeData} /></div>;
  if (taskView === 'deg') return <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}><button className="chip" onClick={() => onSetTaskView('overview')}><ArrowLeft size={18} /></button><DegAnalysis data={runDegData} /></div>;
  if (taskView === 'pathway') return <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}><button className="chip" onClick={() => onSetTaskView('overview')}><ArrowLeft size={18} /></button><PathwayAnalysis data={runPathwayData} /></div>;
  if (taskView === 'prediction') return <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}><button className="chip" onClick={() => onSetTaskView('overview')}><ArrowLeft size={18} /></button><PredictionAnalysis data={runPredictionData} /></div>;

  const WorkflowArrow = () => <div style={{ display: 'flex', alignItems: 'center', padding: '0 0.5rem', color: '#cbd5e1' }}><ChevronRight size={32} /></div>;
  const WorkflowStep = ({ title, num, statusLabel, stepStatus, children, isActive = true }: any) => (
    <section className="card" style={{ flex: 1, textAlign: 'center', opacity: isActive ? 1 : 0.5 }}>
      <div style={{ width: '24px', height: '24px', borderRadius: '50%', background: '#eee', margin: '0 auto 0.5rem auto', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.7rem', fontWeight: 'bold' }}>{num}</div>
      <div style={{ fontSize: '0.9rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>{title}</div>
      <StatusButton label={statusLabel} status={stepStatus} />
      <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center', marginTop: '1rem' }}>{children}</div>
    </section>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}><button className="chip" onClick={onBack}><ArrowLeft size={18} /></button><div><h2 style={{ margin: 0 }}>{selectedRunId}</h2><div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Workflow Pipeline</div></div></div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', backgroundColor: '#f1f5f9', padding: '0.5rem 1rem', borderRadius: '10px' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 'bold' }}>DATASET BRANCH:</div>
          <select className="chip" style={{ border: 'none' }} value={selectedExtId || 'main'} onChange={(e) => { if (e.target.value === 'ADD_NEW') onShowSyncModal(); else onSetSelectedExtId(e.target.value === 'main' ? null : e.target.value); }}>
            <option value="main">Standard Internal Test</option>
            {runStatus?.ext_ids?.map((id: string) => (<option key={id} value={id}>{id} (External)</option>))}
            <option value="ADD_NEW">+ Create New Test Set</option>
          </select>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'stretch' }}>
        <WorkflowStep title="Training" num="1" statusLabel={status?.training} stepStatus={status?.training}>
          <button className="chip" style={{ fontSize: '0.75rem' }} onClick={() => { onSetTaskView('training'); fetchLogs(selectedRunId); }}>Logs</button>
          {status?.training !== 'running' && (
            <button className="chip selected" style={{ fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '4px' }} onClick={() => onRestartTask(selectedRunId)}>
              <RotateCcw size={12} /> Re-run
            </button>
          )}
        </WorkflowStep>
        <WorkflowArrow />
        <WorkflowStep title="Sync Data" num="2" statusLabel={status?.sync ? 'Generated' : 'Pending'} stepStatus={status?.sync}>
          <button className="chip" disabled={!status?.sync} onClick={() => fetchSyncStatus(selectedRunId)}>Details</button>
          {!status?.sync && <button className="chip selected" style={{ opacity: status?.training !== 'completed' ? 0.5 : 1 }} onClick={() => status?.training === 'completed' && onRunStep(2)}>Sync</button>}
        </WorkflowStep>
        <WorkflowArrow />
        <WorkflowStep title="Comparative" num="3" statusLabel={isSizeTask ? (status?.comparative ? 'Done' : 'Pending') : 'Unavailable'} stepStatus={isSizeTask ? status?.comparative : 'unavailable'} isActive={isSizeTask}>
          <button className="chip" disabled={!status?.comparative} onClick={() => fetchComparativeMetrics(selectedRunId)}>Results</button>
          {isSizeTask && !status?.comparative && <button className="chip selected" style={{ opacity: !status?.sync ? 0.5 : 1 }} onClick={() => status?.sync && onRunStep(3)}>Start</button>}
        </WorkflowStep>
        <WorkflowArrow />
        <WorkflowStep title="Bio-markers" num="4" statusLabel={isSizeTask ? (status?.deg ? 'Done' : 'Pending') : 'Unavailable'} stepStatus={isSizeTask ? status?.deg : 'unavailable'} isActive={isSizeTask}>
          <button className="chip" disabled={!status?.deg} onClick={() => { fetchDegMetrics(selectedRunId); fetchPathwayMetrics(selectedRunId); fetchPredictionMetrics(selectedRunId); }}>Results</button>
          {isSizeTask && !status?.deg && <button className="chip selected" style={{ opacity: !status?.comparative ? 0.5 : 1 }} onClick={() => status?.comparative && onRunStep(4)}>Start</button>}
        </WorkflowStep>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={{ padding: '1.25rem', backgroundColor: '#fff', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <div><div style={{ fontSize: '0.7rem', fontWeight: 'bold', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Dataset Description</div><div style={{ fontSize: '0.95rem', fontWeight: '600' }}>{status?.metadata?.description || "No description."}</div>{status?.metadata?.note && <div style={{ fontSize: '0.8rem', color: 'var(--primary-color)', marginTop: '4px' }}>ℹ️ {status.metadata.note}</div>}</div>
            <div style={{ display: 'flex', gap: '1.5rem', textAlign: 'right' }}><div><div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Samples</div><div style={{ fontSize: '1rem', fontWeight: '700' }}>{status?.metadata?.samples || 0}</div></div><div><div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Genes</div><div style={{ fontSize: '1rem', fontWeight: '700' }}>{status?.metadata?.genes || 0}</div></div></div>
          </div>
        </div>
        <div style={{ padding: '1.5rem', backgroundColor: '#f1f5f9', borderRadius: '12px' }}><div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 'bold' }}><Info size={16} /> WORKFLOW AUTOMATION</div><p style={{ fontSize: '0.85rem' }}>Sequential pipeline tracking for reproducible cross-platform alignment and analysis.</p></div>
      </div>
    </div>
  );
};

