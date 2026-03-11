import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { 
  LayoutDashboard, Activity, X, Loader2, ChevronsLeft, ChevronsRight, ArrowLeft, RotateCcw, 
  ChevronRight, Info, Plus, Download, Upload, LineChart as LineChartIcon, Table as TableIcon,
  Settings
} from 'lucide-react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

import type { Project, RunStatus, ResultsStatus, LogResponse } from './types';
import { API_BASE, SIZES, BETAS, LAMBDAS } from './constants';
import { StatusButton, StepItem, FileUploadBox, MetaPanel } from './components/common/UIComponents';
import { SettingsModal } from './components/modals/SettingsModal';
import { SyncExternalModal } from './components/modals/SyncExternalModal';
import { ProjectDashboard } from './components/dashboard/ProjectDashboard';
import { TaskDashboard } from './components/dashboard/TaskDashboard';
import { NewSessionPanel } from './components/dashboard/NewSessionPanel';
import { AblationCharts } from './components/analysis/AblationCharts';
import { AblationAnalyticsModal } from './components/modals/AblationAnalyticsModal';
import { LogViewer } from './components/analysis/LogViewer';
import { SyncStatusDetails } from './components/analysis/SyncStatusDetails';
import { ComparativeAnalysis } from './components/analysis/ComparativeAnalysis';
import { DegAnalysis } from './components/analysis/DegAnalysis';
import { PathwayAnalysis } from './components/analysis/PathwayAnalysis';
import { PredictionAnalysis } from './components/analysis/PredictionAnalysis';

import './App.css';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'train' | 'analysis' | 'new-session'>('train');
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<string>('');
  const [selectedSizes, setSelectedSizes] = useState<number[]>([50]);
  const [selectedBetas, setSelectedBetas] = useState<number[]>([10.0]);
  const [selectedLambdas, setSelectedLambdas] = useState<number[]>([10.0]);
  const [selectedRepeats, setSelectedRepeats] = useState<number>(1);
  const [selectedEpochs, setSelectedEpochs] = useState<number | 'custom'>(500);
  const [customEpochs, setCustomEpochs] = useState<number>(500);
  const [useGpu, setUseGpu] = useState<boolean>(true);
  const [ablationType, setAblationType] = useState<'size' | 'beta' | 'lambda'>('size');
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [selectedExtId, setSelectedExtId] = useState<string | null>(null);
  const [taskView, setTaskView] = useState<'overview' | 'training' | 'sync' | 'comparative' | 'deg' | 'pathway' | 'prediction'>('overview');
  
  const [resultsStatus, setResultsStatus] = useState<ResultsStatus>({ checkpoints: [], logs: [] });
  const [runComparativeData, setRunComparativeData] = useState<any[] | null>(null);
  const [runSyncData, setRunSyncData] = useState<any | null>(null);
  const [runDegData, setRunDegData] = useState<any | null>(null);
  const [runPredictionData, setRunPredictionData] = useState<any | null>(null);
  const [runPathwayData, setRunPathwayData] = useState<any | null>(null);
  const [runTsneData, setRunTsneData] = useState<any[] | null>(null);
  const [ablationData, setAblationData] = useState<any[]>([]);
  const [sensitivityType, setSensitivityType] = useState<'beta' | 'lambda'>('beta');

  // Modals
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [showSyncExternalModal, setShowSyncExternalModal] = useState(false);
  const [viewingAblationCategory, setViewingAblationCategory] = useState<string | null>(null);
  const [ablationLogs, setAblationLogs] = useState<any[] | null>(null);
  
  // External Sync state
  const [extAgFile, setExtAgFile] = useState<File | null>(null);
  const [extRsFile, setExtRsFile] = useState<File | null>(null);
  const [extDescription, setExtDescription] = useState('External Testing dataset');
  const [extCustomSuffix, setExtCustomSuffix] = useState('1');
  const [isRunningExtSync, setIsRunningExtSync] = useState(false);
  const [extSyncResult, setExtSyncResult] = useState<any>(null);

  // Project creation state
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDescription, setNewProjectDescription] = useState('');
  const [agFile, setAgFile] = useState<File | null>(null);
  const [rsFile, setRsFile] = useState<File | null>(null);
  const [labelFile, setLabelFile] = useState<File | null>(null);
  const [isCreatingProject, setIsCreatingProject] = useState(false);

  // Log viewer state
  const [viewingLog, setViewingLog] = useState<string | null>(null);
  const [logData, setLogData] = useState<LogResponse | null>(null);
  const [logMode, setLogMode] = useState<'structured' | 'chart'>('chart');
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 20;

  const [loading, setLoading] = useState(true);

  // Helper: extCustomId derived from suffix
  const extCustomId = `ext_${extCustomSuffix}`;

  // Fetch Projects
  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const projRes = await axios.get(`${API_BASE}/projects`);
        setProjects(projRes.data);
        if (projRes.data.length > 0) setSelectedProject(projRes.data[0].id);
      } catch (err) { console.error(err); }
      finally { setLoading(false); }
    };
    fetchProjects();
  }, []);

  // Fetch Results Status
const fetchStatus = async () => {
  try {
    const resStatus = await axios.get(`${API_BASE}/results`);
    console.log('Results Status:', resStatus.data); // Add this line for debugging
    setResultsStatus(resStatus.data);
  } catch (err) { console.error(err); }
};

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  // Fetch Ablation Data
  useEffect(() => {
    if (selectedProject) {
      axios.get(`${API_BASE}/projects/${selectedProject}/ablation`)
        .then(res => setAblationData(res.data))
        .catch(err => console.error(err));
    }
  }, [selectedProject, resultsStatus]);

  // Handlers
  const handleStartSession = async () => {
    const project = projects.find(p => p.id === selectedProject);
    if (!project) return;
    const payload = {
      config_path: project.config_path,
      sizes: ablationType === 'size' ? selectedSizes : [],
      betas: ablationType === 'beta' ? selectedBetas : [],
      lambdas: ablationType === 'lambda' ? selectedLambdas : [],
      repeats: selectedRepeats,
      epochs: selectedEpochs === 'custom' ? customEpochs : selectedEpochs,
      use_gpu: useGpu
    };
    try {
      await axios.post(`${API_BASE}/train`, payload);
      alert("Training session started!");
      setActiveTab('train');
    } catch (err) { alert("Failed to start session"); }
  };

  const handleRunStep = async (step: number) => {
    if (!selectedRunId) return;
    const project = projects.find(p => selectedRunId.startsWith(p.id));
    try {
      await axios.post(`${API_BASE}/runs/${selectedRunId}/run_step`, null, {
        params: { step, config_path: project?.config_path, ext_id: selectedExtId }
      });
      alert(`Step ${step} started.`);
    } catch (err) { alert("Failed to start step"); }
  };

  const handleStopTask = async (runId: string) => {
    try {
      await axios.post(`${API_BASE}/runs/${runId}/stop`);
      fetchStatus();
    } catch (err) { alert("Failed to stop task"); }
  };

  const handleRestartTask = async (runId: string) => {
    try {
      await axios.post(`${API_BASE}/runs/${runId}/restart`);
      fetchStatus();
    } catch (err) { alert("Failed to restart task"); }
  };

  const handleCreateProject = async () => {
    if (!newProjectName || !agFile || !rsFile) { alert("Required fields missing."); return; }
    const formData = new FormData();
    formData.append('project_name', newProjectName);
    formData.append('description', newProjectDescription);
    formData.append('df_ag', agFile);
    formData.append('df_rs', rsFile);
    if (labelFile) formData.append('label', labelFile);
    setIsCreatingProject(true);
    try {
      await axios.post(`${API_BASE}/projects/create`, formData);
      alert("Project created!");
      setShowSettingsModal(false);
      const projRes = await axios.get(`${API_BASE}/projects`);
      setProjects(projRes.data);
    } catch (err) { alert("Failed to create project"); }
    finally { setIsCreatingProject(false); }
  };

  const handleRunExtSync = async () => {
    if (!selectedRunId || (!extAgFile && !extRsFile)) { alert("Files missing."); return; }
    const formData = new FormData();
    if (extAgFile) formData.append('test_ag', extAgFile);
    if (extRsFile) formData.append('test_rs', extRsFile);
    formData.append('ext_id', extCustomId);
    formData.append('description', extDescription);
    setIsRunningExtSync(true);
    try {
      const res = await axios.post(`${API_BASE}/runs/${selectedRunId}/sync_external`, formData);
      setExtSyncResult(res.data);
      fetchStatus();
    } catch (err) { alert("External sync failed"); }
    finally { setIsRunningExtSync(false); }
  };

  const handleDownloadGenelist = () => {
    if (!selectedProject) return;
    window.open(`${API_BASE}/projects/${selectedProject}/genelist/download`);
  };

  const fetchLogs = (runId: string) => {
    setTaskView('training');
    setViewingLog(runId);
    setLogData(null);
    axios.get(`${API_BASE}/runs/${runId}/logs`).then(res => setLogData(res.data));
  };

  const fetchSyncStatus = (runId: string) => {
    setTaskView('sync');
    setRunSyncData(null);
    axios.get(`${API_BASE}/runs/${runId}/sync`, { params: { ext_id: selectedExtId } }).then(res => setRunSyncData(res.data));
    fetchTsneCoords(runId);
  };

  const fetchComparativeMetrics = (runId: string) => {
    setTaskView('comparative');
    setRunComparativeData(null);
    axios.get(`${API_BASE}/runs/${runId}/comparative`, { params: { ext_id: selectedExtId } }).then(res => setRunComparativeData(res.data));
  };

  const fetchDegMetrics = (runId: string) => {
    setTaskView('deg');
    setRunDegData(null);
    axios.get(`${API_BASE}/runs/${runId}/deg`, { params: { ext_id: selectedExtId } }).then(res => setRunDegData(res.data));
  };

  const fetchPathwayMetrics = (runId: string) => {
    setTaskView('pathway');
    setRunPathwayData(null);
    axios.get(`${API_BASE}/runs/${runId}/pathway`, { params: { ext_id: selectedExtId } }).then(res => setRunPathwayData(res.data));
  };

  const fetchPredictionMetrics = (runId: string) => {
    setTaskView('prediction');
    setRunPredictionData(null);
    axios.get(`${API_BASE}/runs/${runId}/prediction`, { params: { ext_id: selectedExtId } }).then(res => setRunPredictionData(res.data));
  };

  const fetchTsneCoords = (runId: string) => {
    setRunTsneData(null);
    axios.get(`${API_BASE}/runs/${runId}/tsne`, { params: { ext_id: selectedExtId } })
      .then(res => setRunTsneData(res.data))
      .catch(() => setRunTsneData([]));
  };

  const fetchAblationLogs = (category: string) => {
    setViewingAblationCategory(category);
    setAblationLogs(null);
    axios.get(`${API_BASE}/projects/${selectedProject}/ablation_logs`, { params: { category } })
      .then(res => setAblationLogs(res.data))
      .catch(err => console.error(err));
  };

  // Renderers for analysis
  const renderLogViewer = () => <LogViewer logData={logData} runId={selectedRunId || ''} />;
  const renderSyncStatus = () => <SyncStatusDetails data={runSyncData} />;
  const renderComparativeAnalysis = () => <ComparativeAnalysis data={runComparativeData} />;
  const renderDegAnalysis = () => <DegAnalysis data={runDegData} />;
  const renderPathwayAnalysis = () => <PathwayAnalysis data={runPathwayData} />;
  const renderPredictionAnalysis = () => <PredictionAnalysis data={runPredictionData} />;

  const toggleSelection = (val: any, list: any[], setter: (val: any[]) => void) => {
    if (list.includes(val)) setter(list.filter(item => item !== val));
    else setter([...list, val]);
  };

  // Logic for status selection in TaskDashboard
  const runStatus = selectedRunId ? resultsStatus.run_statuses?.[selectedRunId] : undefined;
  const status = (selectedExtId && runStatus?.ext_statuses?.[selectedExtId]) 
    ? { ...runStatus, ...runStatus.ext_statuses[selectedExtId] }
    : runStatus;
  const isSizeTask = selectedRunId ? (selectedRunId.includes("Size") && !selectedRunId.includes("Architecture")) : false;
  const currentProj = projects.find(p => p.id === selectedProject);

  return (
    <div className="dashboard-container">
      <aside className="sidebar">
        <div className="sidebar-header"><Activity size={24} /><span>GANomics Dashboard</span></div>
        <nav className="nav-menu">
          <div style={{ padding: '1rem' }}><button className="chip selected" style={{ width: '100%', padding: '0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', fontWeight: 'bold' }} onClick={() => setActiveTab('new-session')}><Plus size={18} /> New Experiment</button></div>
          <div style={{ padding: '0.5rem 1rem', fontSize: '0.65rem', fontWeight: 'bold', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Menu</div>
          <a className={`nav-item ${activeTab === 'train' && !selectedRunId ? 'active' : ''}`} onClick={() => { setActiveTab('train'); setSelectedRunId(null); }}><LayoutDashboard size={18} /> Project Dashboard</a>
          {selectedRunId && (
            <><div style={{ padding: '1.5rem 1rem 0.5rem 1rem', fontSize: '0.65rem', fontWeight: 'bold', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Task Pipeline</div>
              <StepItem num="1" label="Training" active={taskView === 'training'} status={status?.training} onClick={() => setTaskView('training')} />
              <StepItem num="2" label="Sync Data" active={taskView === 'sync'} status={status?.sync} onClick={() => setTaskView('sync')} />
              <StepItem num="3" label="Comparative" active={taskView === 'comparative'} status={status?.comparative} onClick={() => setTaskView('comparative')} />
              <StepItem num="4" label="Bio-markers" active={['deg','pathway','prediction'].includes(taskView)} status={status?.deg} onClick={() => setTaskView('deg')} />
            </>
          )}
          <div style={{ marginTop: 'auto', padding: '1rem' }}><div className="nav-item" onClick={() => setShowSettingsModal(true)}><Settings size={18} /> Create Project</div></div>
        </nav>
      </aside>

      <main className="main-content">
        <SettingsModal 
          show={showSettingsModal} onClose={() => setShowSettingsModal(false)}
          newProjectName={newProjectName} setNewProjectName={setNewProjectName}
          newProjectDescription={newProjectDescription} setNewProjectDescription={setNewProjectDescription}
          agFile={agFile} setAgFile={setAgFile} rsFile={rsFile} setRsFile={setRsFile}
          labelFile={labelFile} setLabelFile={setLabelFile} isCreating={isCreatingProject} onCreateProject={handleCreateProject}
        />
        <SyncExternalModal 
          show={showSyncExternalModal} onClose={() => setShowSyncExternalModal(false)}
          extAgFile={extAgFile} setExtAgFile={setExtAgFile} extRsFile={extRsFile} setExtRsFile={setExtRsFile}
          extDescription={extDescription} setExtDescription={setExtDescription}
          extCustomSuffix={extCustomSuffix} setExtCustomSuffix={setExtCustomSuffix}
          isRunning={isRunningExtSync} result={extSyncResult} onRunSync={handleRunExtSync}
          onDownloadGenelist={handleDownloadGenelist} onSwitchBranch={(id) => { setSelectedExtId(id); setShowSyncExternalModal(false); }}
        />
        <AblationAnalyticsModal 
          category={viewingAblationCategory} onClose={() => setViewingAblationCategory(null)}
          projectName={currentProj?.name || ''} ablationLogs={ablationLogs}
          sensitivityType={sensitivityType} onSetSensitivityType={setSensitivityType}
        />

        {activeTab === 'new-session' ? (
          <NewSessionPanel 
            projects={projects} selectedProject={selectedProject} onSelectProject={setSelectedProject}
            ablationType={ablationType} onSetAblationType={setAblationType}
            selectedSizes={selectedSizes} setSelectedSizes={setSelectedSizes}
            selectedBetas={selectedBetas} setSelectedBetas={setSelectedBetas}
            selectedLambdas={selectedLambdas} setSelectedLambdas={setSelectedLambdas}
            selectedRepeats={selectedRepeats} setSelectedRepeats={setSelectedRepeats}
            selectedEpochs={selectedEpochs} setSelectedEpochs={setSelectedEpochs}
            customEpochs={customEpochs} setCustomEpochs={setCustomEpochs}
            useGpu={useGpu} setUseGpu={setUseGpu} onStartSession={handleStartSession}
            onBack={() => setActiveTab('train')} toggleSelection={toggleSelection}
          />
        ) : selectedRunId ? (
          <TaskDashboard 
            selectedRunId={selectedRunId} selectedExtId={selectedExtId} runStatus={runStatus} status={status}
            isSizeTask={isSizeTask} currentProj={currentProj} taskView={taskView}
            onBack={() => setSelectedRunId(null)} onSetTaskView={setTaskView} onSetSelectedExtId={setSelectedExtId}
            onShowSyncModal={() => setShowSyncExternalModal(true)} onRestartTask={handleRestartTask} onRunStep={handleRunStep}
            fetchLogs={fetchLogs} fetchSyncStatus={fetchSyncStatus} fetchComparativeMetrics={fetchComparativeMetrics} 
            fetchDegMetrics={fetchDegMetrics} fetchPathwayMetrics={fetchPathwayMetrics} fetchPredictionMetrics={fetchPredictionMetrics}
            fetchTsneCoords={fetchTsneCoords}
            logData={logData} runSyncData={runSyncData} runComparativeData={runComparativeData} 
            runDegData={runDegData} runPathwayData={runPathwayData} runPredictionData={runPredictionData}
            runTsneData={runTsneData}
            renderLogViewer={renderLogViewer} renderSyncStatus={renderSyncStatus} renderComparativeAnalysis={renderComparativeAnalysis}
            renderDegAnalysis={renderDegAnalysis} renderPathwayAnalysis={renderPathwayAnalysis} renderPredictionAnalysis={renderPredictionAnalysis}
          />
        ) : (
          <ProjectDashboard 
            projects={projects} selectedProject={selectedProject} onSelectProject={setSelectedProject}
            resultsStatus={resultsStatus} onSelectRun={(id) => setSelectedRunId(id)}
            onFetchAblationLogs={fetchAblationLogs}
            onStopTask={handleStopTask} onRestartTask={handleRestartTask} onFetchLogs={fetchLogs}
            ablationData={ablationData} sensitivityType={sensitivityType} onSetSensitivityType={setSensitivityType}
          />
        )}
      </main>
    </div>
  );
};

export default App;
