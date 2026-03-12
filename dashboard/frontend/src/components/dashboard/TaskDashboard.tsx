import React, { useState } from 'react';
import { ArrowLeft, RotateCcw, ChevronRight, Info, RefreshCw, Download, Settings, Sliders, Check, HelpCircle } from 'lucide-react';
import { StatusButton } from '../common/UIComponents';
import type { RunStatus, Project, LogResponse } from '../../types';
import { API_BASE } from '../../constants';

// ... (previous imports)

interface TaskDashboardProps {
// ... (props)
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
  const [modelType, setModelType] = useState('rf');

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
    if (step === 3) {
      params.algorithms = selectedAlgos;
    }
    if (step === 4) {
      params.filter_pathways = pathwayFilter;
      params.libraries = selectedLibraries;
      // modelType is passed but backend biomarker.py doesn't use it yet (fixed to RF)
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
              <h4 style={{ fontSize: '0.9rem', marginBottom: '1rem', color: 'var(--text-muted)', borderBottom: '1px solid #f1f5f9', paddingBottom: '0.5rem' }}>STEP 4: BIOMARKER ANALYSIS</h4>
              
              <div style={{ marginBottom: '1.25rem' }}>
                <div style={{ fontSize: '0.85rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Prediction Model:</div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <div className="chip selected" style={{ fontSize: '0.75rem', opacity: 0.9, cursor: 'default', pointerEvents: 'none' }}>
                    <Check size={12} style={{ marginRight: '4px' }} /> Random Forest
                  </div>
                  <div className="chip disabled" style={{ fontSize: '0.75rem', title: 'Coming soon' }}>
                    XGBoost (Soon)
                  </div>
                </div>
              </div>

              <div style={{ fontSize: '0.85rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Pathway Settings:</div>
              <label style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer', marginBottom: '1rem' }}>
                <input type="checkbox" checked={pathwayFilter} onChange={(e) => setPathwayFilter(e.target.checked)} style={{ width: '18px', height: '18px' }} />
                <span style={{ fontSize: '0.85rem' }}>Apply Size Filter (15-500 genes)</span>
                <HelpCircle size={14} style={{ color: 'var(--text-muted)' }} title="Recommended to filter out overly generic pathways." />
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
            
            <div style={{ padding: '1rem', backgroundColor: '#f1f5f9', borderRadius: '8px', fontSize: '0.8rem', color: '#475569' }}>
              <Info size={14} style={{ marginRight: '6px' }} /> 
              Settings are saved for your current session and applied when you trigger a workflow step.
            </div>
            
            <button className="chip selected" style={{ width: '100%', padding: '0.75rem' }} onClick={() => setShowStepSettings(false)}>Save & Close</button>
          </div>
        </div>
      </div>
    );
  };

  const renderLogViewer = () => <LogViewer logData={logData} runId={selectedRunId || ''} />;
// ...
  const renderSyncStatus = () => <SyncStatusDetails data={runSyncData} />;
  const renderComparativeAnalysis = () => <ComparativeAnalysis data={runComparativeData} />;
  const renderDegAnalysis = () => <DegAnalysis data={runDegData} />;
  const renderPathwayAnalysis = () => <PathwayAnalysis data={runPathwayData} />;
  const renderPredictionAnalysis = () => <PredictionAnalysis data={runPredictionData} />;

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
        <button className="chip" style={{ justifyContent: 'space-between', padding: '0.75rem 1rem', backgroundColor: '#fff' }} onClick={() => handleDownloadResults('microarray_real.csv')}>
          <span>Microarray (Real)</span><Download size={14} />
        </button>
        <button className="chip" style={{ justifyContent: 'space-between', padding: '0.75rem 1rem', backgroundColor: '#fff' }} onClick={() => handleDownloadResults('microarray_fake.csv')}>
          <span>Microarray (Synthetic)</span><Download size={14} />
        </button>
        <button className="chip" style={{ justifyContent: 'space-between', padding: '0.75rem 1rem', backgroundColor: '#fff' }} onClick={() => handleDownloadResults('rnaseq_real.csv')}>
          <span>RNA-Seq (Real)</span><Download size={14} />
        </button>
        <button className="chip" style={{ justifyContent: 'space-between', padding: '0.75rem 1rem', backgroundColor: '#fff' }} onClick={() => handleDownloadResults('rnaseq_fake.csv')}>
          <span>RNA-Seq (Synthetic)</span><Download size={14} />
        </button>
      </div>
    </div>
  );

  if (taskView === 'training') return <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}><div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}><button className="chip" onClick={() => onSetTaskView('overview')}><ArrowLeft size={18} /></button><h2 style={{ margin: 0 }}>Training Performance: {selectedRunId}</h2></div><LogViewer logData={logData} runId={selectedRunId} /></div>;
  
  if (taskView === 'sync') {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <button className="chip" onClick={() => onSetTaskView('overview')}><ArrowLeft size={18} /></button>
          <h2 style={{ margin: 0 }}>Sync Data Analysis: {selectedRunId}</h2>
        </div>
        <TsneVisualization data={runTsneData} />
        <DownloadBox />
      </div>
    );
  }

  if (taskView === 'comparative') return <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}><div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}><button className="chip" onClick={() => onSetTaskView('overview')}><ArrowLeft size={18} /></button><h2 style={{ margin: 0 }}>Comparative Analysis: {selectedRunId}</h2></div><ComparativeAnalysis data={runComparativeData} /></div>;
  
  if (['deg', 'pathway', 'prediction'].includes(taskView)) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <button className="chip" onClick={() => onSetTaskView('overview')} style={{ padding: '0.5rem' }}>
              <ArrowLeft size={18} />
            </button>
            <h2 style={{ margin: 0 }}>Bio-marker Analysis: {selectedRunId}</h2>
          </div>
          <div style={{ display: 'flex', gap: '0.25rem', backgroundColor: '#f3f4f6', padding: '0.25rem', borderRadius: '8px' }}>
            <button 
              className={`chip ${taskView === 'deg' ? 'selected' : ''}`} 
              style={{ border: 'none' }}
              onClick={() => fetchDegMetrics(selectedRunId)}
            >DEG</button>
            <button 
              className={`chip ${taskView === 'pathway' ? 'selected' : ''}`} 
              style={{ border: 'none' }}
              onClick={() => fetchPathwayMetrics(selectedRunId)}
            >Pathway</button>
            <button 
              className={`chip ${taskView === 'prediction' ? 'selected' : ''}`} 
              style={{ border: 'none' }}
              onClick={() => fetchPredictionMetrics(selectedRunId)}
            >Prediction</button>
          </div>
        </div>
        {taskView === 'deg' && <DegAnalysis data={runDegData} />}
        {taskView === 'pathway' && <PathwayAnalysis data={runPathwayData} />}
        {taskView === 'prediction' && <PredictionAnalysis data={runPredictionData} />}
      </div>
    );
  }

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
      <StepSettingsModal />
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <button className="chip" onClick={onBack}><ArrowLeft size={18} /></button>
          <div>
            <h2 style={{ margin: 0 }}>{selectedRunId}</h2>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Workflow Pipeline</div>
              <button 
                className="chip" 
                style={{ padding: '2px 8px', fontSize: '0.7rem', display: 'flex', alignItems: 'center', gap: '4px', backgroundColor: '#f8fafc' }}
                onClick={() => setShowStepSettings(true)}
              >
                <Settings size={12} /> Parameters
              </button>
            </div>
          </div>
        </div>
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
          {!status?.sync && (
            <button 
              className={`chip selected ${status?.training !== 'completed' ? 'disabled' : ''}`} 
              disabled={status?.training !== 'completed'}
              style={{ opacity: status?.training !== 'completed' ? 0.5 : 1 }} 
              onClick={() => status?.training === 'completed' && onRunStep(2)}
            >
              Sync
            </button>
          )}
        </WorkflowStep>
        <WorkflowArrow />
        <WorkflowStep title="Comparative" num="3" statusLabel={status?.comparative ? 'Done' : 'Pending'} stepStatus={status?.comparative || 'pending'}>
          <button className="chip" disabled={!status?.comparative} onClick={() => fetchComparativeMetrics(selectedRunId)}>Results</button>
          {!status?.comparative && (
            <button 
              className={`chip selected ${!status?.sync ? 'disabled' : ''}`} 
              disabled={!status?.sync}
              style={{ opacity: !status?.sync ? 0.5 : 1 }} 
              onClick={() => status?.sync && handleRunStepWithParams(3)}
            >
              Start
            </button>
          )}
        </WorkflowStep>
        <WorkflowArrow />
        <WorkflowStep title="Bio-markers" num="4" statusLabel={status?.deg ? 'Done' : 'Pending'} stepStatus={status?.deg || 'pending'}>
          <button className="chip" disabled={!status?.deg} onClick={() => { fetchDegMetrics(selectedRunId); fetchPathwayMetrics(selectedRunId); fetchPredictionMetrics(selectedRunId); }}>Results</button>
          {!status?.deg && (
            <button 
              className={`chip selected ${!status?.comparative ? 'disabled' : ''}`} 
              disabled={!status?.comparative}
              style={{ opacity: !status?.comparative ? 0.5 : 1 }} 
              onClick={() => status?.comparative && handleRunStepWithParams(4)}
            >
              Start
            </button>
          )}
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
