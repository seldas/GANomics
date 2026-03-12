import React, { useState } from 'react';
import { ArrowLeft, RotateCcw, ChevronRight, Info, RefreshCw, Download, Settings, Sliders, Check, HelpCircle } from 'lucide-react';
import { StatusButton, MetaPanel } from '../common/UIComponents';
import type { RunStatus, Project, LogResponse } from '../../types';
import { API_BASE } from '../../constants';

import { LogViewer } from '../analysis/LogViewer';
import { ComparativeAnalysis } from '../analysis/ComparativeAnalysis';
import { DegAnalysis } from '../analysis/DegAnalysis';
import { PathwayAnalysis } from '../analysis/PathwayAnalysis';
import { PredictionAnalysis } from '../analysis/PredictionAnalysis';
import { TsneVisualization } from '../analysis/TsneVisualization';
import { SyncStatusDetails } from '../analysis/SyncStatusDetails';

interface TaskDashboardProps {
  selectedRunId: string;
  selectedExtId: string | null;
  runStatus: RunStatus | undefined;
  status: any;
  isSizeTask: boolean;
  currentProj: Project | undefined;
  taskView: 'overview' | 'training' | 'sync' | 'comparative' | 'deg' | 'pathway' | 'prediction';
  onBack: () => void;
  onSetTaskView: (view: 'overview' | 'training' | 'sync' | 'comparative' | 'deg' | 'pathway' | 'prediction') => void;
  onSetSelectedExtId: (id: string | null) => void;
  onShowSyncModal: () => void;
  onRestartTask: (id: string) => void;
  onRunStep: (step: number, params?: any) => void;
  fetchLogs: (id: string) => void;
  fetchSyncStatus: (id: string) => void;
  fetchComparativeMetrics: (id: string) => void;
  fetchDegMetrics: (id: string) => void;
  fetchPathwayMetrics: (id: string) => void;
  fetchPredictionMetrics: (id: string) => void;
  fetchTsneCoords: (id: string) => void;
  logData: LogResponse | null;
  runSyncData: any | null;
  runComparativeData: any[] | null;
  runDegData: any | null;
  runPathwayData: any | null;
  runPredictionData: any | null;
  runTsneData: any[] | null;
}

export const TaskDashboard: React.FC<TaskDashboardProps> = ({
  selectedRunId, selectedExtId, runStatus, status, isSizeTask, currentProj, taskView,
  onBack, onSetTaskView, onSetSelectedExtId, onShowSyncModal, onRestartTask, onRunStep,
  fetchLogs, fetchSyncStatus, fetchComparativeMetrics, fetchDegMetrics, fetchPathwayMetrics, fetchPredictionMetrics, fetchTsneCoords,
  logData, runSyncData, runComparativeData, runDegData, runPathwayData, runPredictionData, runTsneData
}) => {
  const [showStepSettings, setShowStepSettings] = useState(false);
  const [pathwayFilter, setPathwayFilter] = useState(true);
  const [selectedLibraries, setSelectedLibraries] = useState<string[]>(['KEGG_2021_Human', 'GO_Biological_Process_2021']);
  const [selectedAlgos, setSelectedAlgos] = useState<string[]>(['combat', 'yugene', 'cublock', 'tdm', 'qn']);

  const libraries = ['KEGG_2021_Human', 'GO_Biological_Process_2021', 'MSigDB_Hallmark_2020', 'Reactome_2022'];
  const baselines = [
    { id: 'combat', name: 'ComBat' },
    { id: 'yugene', name: 'YuGene' },
    { id: 'cublock', name: 'CuBlock' },
    { id: 'tdm', name: 'TDM' },
    { id: 'qn', name: 'Quantile' }
  ];

  const handleRunStepWithParams = (step: number) => {
    const params: any = {};
    if (step === 3) params.algorithms = selectedAlgos;
    if (step === 5) {
      params.filter_pathways = pathwayFilter;
      params.libraries = selectedLibraries;
    }
    onRunStep(step, params);
  };

  const StepSettingsModal = () => {
    if (!showStepSettings) return null;
    return (
      <div className="modal-overlay" onClick={() => setShowStepSettings(false)}>
        <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: '550px' }}>
          <div className="modal-header">
            <h3><Sliders size={20} style={{ marginRight: '8px', verticalAlign: 'middle' }} /> Pipeline Parameters</h3>
            <button className="icon-button" onClick={() => setShowStepSettings(false)}><ArrowLeft size={20} /></button>
          </div>
          <div style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <section>
              <h4 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--text-muted)', borderBottom: '1px solid #f1f5f9', paddingBottom: '0.5rem' }}>STEP 3: COMPARATIVE ANALYSIS</h4>
              <div style={{ fontSize: '0.85rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Baseline Algorithms:</div>
              <div className="chip-grid" style={{ gap: '0.5rem' }}>
                {baselines.map(algo => (
                  <div key={algo.id} className={`chip ${selectedAlgos.includes(algo.id) ? 'selected' : ''}`} onClick={() => {
                    if (selectedAlgos.includes(algo.id)) setSelectedAlgos(selectedAlgos.filter(a => a !== algo.id));
                    else setSelectedAlgos([...selectedAlgos, algo.id]);
                  }} style={{ fontSize: '0.75rem' }}>
                    {selectedAlgos.includes(algo.id) && <Check size={12} style={{ marginRight: '4px' }} />}
                    {algo.name}
                  </div>
                ))}
              </div>
            </section>
            <section>
              <h4 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--text-muted)', borderBottom: '1px solid #f1f5f9', paddingBottom: '0.5rem' }}>STEP 5: PATHWAY SETTINGS</h4>
              <label style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer', marginBottom: '1rem' }}>
                <input type="checkbox" checked={pathwayFilter} onChange={(e) => setPathwayFilter(e.target.checked)} style={{ width: '18px', height: '18px' }} />
                <span style={{ fontSize: '0.85rem' }}>Apply Size Filter (15-500 genes)</span>
              </label>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Libraries:</div>
              <div className="chip-grid" style={{ gap: '0.5rem' }}>
                {libraries.map(lib => (
                  <div key={lib} className={`chip ${selectedLibraries.includes(lib) ? 'selected' : ''}`} onClick={() => {
                    if (selectedLibraries.includes(lib)) setSelectedLibraries(selectedLibraries.filter(l => l !== lib));
                    else setSelectedLibraries([...selectedLibraries, lib]);
                  }} style={{ fontSize: '0.75rem' }}>
                    {selectedLibraries.includes(lib) && <Check size={12} style={{ marginRight: '4px' }} />}
                    {lib.split('_')[0]}
                  </div>
                ))}
              </div>
            </section>
            <button className="chip selected" style={{ width: '100%', padding: '0.75rem' }} onClick={() => setShowStepSettings(false)}>Save & Close</button>
          </div>
        </div>
      </div>
    );
  };

  const handleDownloadResults = (filename: string) => {
    const url = `${API_BASE}/runs/${selectedRunId}/sync/download?filename=${filename}${selectedExtId ? `&ext_id=${selectedExtId}` : ''}`;
    window.open(url);
  };

  const DownloadBox = () => (
    <div className="card" style={{ padding: '1.5rem', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
      <h3 style={{ fontSize: '1rem', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
        <Download size={18} /> Export Synchronized Datasets
      </h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
        {['microarray_real.csv', 'microarray_fake.csv', 'rnaseq_real.csv', 'rnaseq_fake.csv'].map(f => (
          <button key={f} className="chip" style={{ justifyContent: 'space-between', padding: '0.75rem 1rem', backgroundColor: '#fff' }} onClick={() => handleDownloadResults(f)}>
            <span>{f.replace('_', ' ').replace('.csv', '')}</span><Download size={14} />
          </button>
        ))}
      </div>
    </div>
  );

  if (taskView === 'training') return <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}><div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}><button className="chip" onClick={() => onSetTaskView('overview')}><ArrowLeft size={18} /></button><h2 style={{ margin: 0 }}>Training Performance</h2></div><LogViewer logData={logData} runId={selectedRunId} /></div>;
  if (taskView === 'sync') return <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}><div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}><button className="chip" onClick={() => onSetTaskView('overview')}><ArrowLeft size={18} /></button><h2 style={{ margin: 0 }}>Sync Analysis</h2></div><TsneVisualization data={runTsneData} /><DownloadBox /></div>;
  if (taskView === 'comparative') return <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}><div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}><button className="chip" onClick={() => onSetTaskView('overview')}><ArrowLeft size={18} /></button><h2 style={{ margin: 0 }}>Comparative Analysis</h2></div><ComparativeAnalysis data={runComparativeData} /></div>;
  if (taskView === 'deg') return <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}><div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}><button className="chip" onClick={() => onSetTaskView('overview')}><ArrowLeft size={18} /></button><h2 style={{ margin: 0 }}>DEG Analysis</h2></div><DegAnalysis data={runDegData} /></div>;
  if (taskView === 'pathway') return <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}><div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}><button className="chip" onClick={() => onSetTaskView('overview')}><ArrowLeft size={18} /></button><h2 style={{ margin: 0 }}>Pathway Concordance</h2></div><PathwayAnalysis data={runPathwayData} /></div>;
  if (taskView === 'prediction') return <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}><div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}><button className="chip" onClick={() => onSetTaskView('overview')}><ArrowLeft size={18} /></button><h2 style={{ margin: 0 }}>Prediction Modeling</h2></div><PredictionAnalysis data={runPredictionData} /></div>;

  const WorkflowArrow = () => <div style={{ display: 'flex', alignItems: 'center', color: '#cbd5e1' }}><ChevronRight size={20} /></div>;
  const WorkflowStep = ({ title, num, statusLabel, stepStatus, children }: any) => (
    <section className="card" style={{ flex: 1, textAlign: 'center', padding: '1rem 0.5rem' }}>
      <div style={{ width: '20px', height: '20px', borderRadius: '50%', background: '#eee', margin: '0 auto 0.5rem auto', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.6rem', fontWeight: 'bold' }}>{num}</div>
      <div style={{ fontSize: '0.75rem', fontWeight: 'bold', marginBottom: '0.5rem', whiteSpace: 'nowrap' }}>{title}</div>
      <StatusButton label={statusLabel} status={stepStatus} />
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem', justifyContent: 'center', marginTop: '1rem' }}>{children}</div>
    </section>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <StepSettingsModal />
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <button className="chip" onClick={onBack}><ArrowLeft size={18} /></button>
          <div>
            <h2 style={{ margin: 0 }}>{selectedRunId}</h2>
            <button className="chip" style={{ padding: '2px 8px', fontSize: '0.7rem', marginTop: '4px' }} onClick={() => setShowStepSettings(true)}><Settings size={12} /> Parameters</button>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', backgroundColor: '#f1f5f9', padding: '0.5rem 1rem', borderRadius: '10px' }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 'bold' }}>BRANCH:</div>
          <select className="chip" style={{ border: 'none' }} value={selectedExtId || 'main'} onChange={(e) => { if (e.target.value === 'ADD_NEW') onShowSyncModal(); else onSetSelectedExtId(e.target.value === 'main' ? null : e.target.value); }}>
            <option value="main">Internal</option>
            {runStatus?.ext_ids?.map((id: string) => (<option key={id} value={id}>{id}</option>))}
            <option value="ADD_NEW">+ New</option>
          </select>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'stretch', gap: '2px' }}>
        <WorkflowStep title="Training" num="1" statusLabel={status?.training} stepStatus={status?.training}>
          <button className="chip" style={{ fontSize: '0.65rem' }} onClick={() => { onSetTaskView('training'); fetchLogs(selectedRunId); }}>Logs</button>
        </WorkflowStep>
        <WorkflowArrow />
        <WorkflowStep title="Sync" num="2" statusLabel={status?.sync ? 'Done' : 'Pending'} stepStatus={status?.sync}>
          <button className="chip" style={{ fontSize: '0.65rem' }} disabled={!status?.sync} onClick={() => fetchSyncStatus(selectedRunId)}>Results</button>
          {!status?.sync && <button className="chip selected" style={{ fontSize: '0.65rem' }} onClick={() => onRunStep(2)}>Run</button>}
        </WorkflowStep>
        <WorkflowArrow />
        <WorkflowStep title="Comp." num="3" statusLabel={status?.comparative ? 'Done' : 'Pending'} stepStatus={status?.comparative}>
          <button className="chip" style={{ fontSize: '0.65rem' }} disabled={!status?.comparative} onClick={() => fetchComparativeMetrics(selectedRunId)}>Results</button>
          {!status?.comparative && <button className="chip selected" style={{ fontSize: '0.65rem' }} onClick={() => handleRunStepWithParams(3)}>Run</button>}
        </WorkflowStep>
        <WorkflowArrow />
        <WorkflowStep title="DEG" num="4" statusLabel={status?.deg ? 'Done' : 'Pending'} stepStatus={status?.deg}>
          <button className="chip" style={{ fontSize: '0.65rem' }} disabled={!status?.deg} onClick={() => fetchDegMetrics(selectedRunId)}>Results</button>
          {!status?.deg && <button className="chip selected" style={{ fontSize: '0.65rem' }} onClick={() => onRunStep(4)}>Run</button>}
        </WorkflowStep>
        <WorkflowArrow />
        <WorkflowStep title="Path" num="5" statusLabel={status?.pathway ? 'Done' : 'Pending'} stepStatus={status?.pathway}>
          <button className="chip" style={{ fontSize: '0.65rem' }} disabled={!status?.pathway} onClick={() => fetchPathwayMetrics(selectedRunId)}>Results</button>
          {!status?.pathway && <button className="chip selected" style={{ fontSize: '0.65rem' }} onClick={() => handleRunStepWithParams(5)}>Run</button>}
        </WorkflowStep>
        <WorkflowArrow />
        <WorkflowStep title="Pred" num="6" statusLabel={status?.pred_model ? 'Done' : 'Pending'} stepStatus={status?.pred_model}>
          <button className="chip" style={{ fontSize: '0.65rem' }} disabled={!status?.pred_model} onClick={() => fetchPredictionMetrics(selectedRunId)}>Results</button>
          {!status?.pred_model && <button className="chip selected" style={{ fontSize: '0.65rem' }} onClick={() => onRunStep(6)}>Run</button>}
        </WorkflowStep>
      </div>

      <MetaPanel description={status?.metadata?.description} samples={status?.metadata?.samples} genes={status?.metadata?.genes} note={status?.metadata?.note} />
    </div>
  );
};
