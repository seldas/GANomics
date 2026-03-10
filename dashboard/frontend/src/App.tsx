import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { 
  LayoutDashboard, 
  Settings, 
  Play, 
  Activity, 
  Database,
  X,
  Table as TableIcon,
  BarChart3,
  LineChart as LineChartIcon,
  Loader2,
  Eye,
  EyeOff,
  Search,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  ChevronsLeft,
  ChevronsRight,
  ArrowLeft
} from 'lucide-react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import './App.css';

interface Project {
  id: string;
  name: string;
  genes: number;
  samples: number;
  config_path: string;
  config?: any;
}

interface LogResponse {
  run_id: string;
  summary: any;
  structured: any[];
  total_lines: number;
}

const API_BASE = "http://localhost:8832/api";

const LOSS_METRICS = [
  { key: 'G_A', color: '#007bff', label: 'Gen A' },
  { key: 'G_B', color: '#0056b3', label: 'Gen B' },
  { key: 'D_A', color: '#dc3545', label: 'Disc A' },
  { key: 'D_B', color: '#a71d2a', label: 'Disc B' },
  { key: 'Cycle', color: '#ffc107', label: 'Cycle' },
  { key: 'Feedback', color: '#28a745', label: 'Feedback' },
  { key: 'IDT', color: '#6f42c1', label: 'Identity' },
];

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'train' | 'analysis'>('train');
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<string>('');
  const [projectAblationTab, setProjectAblationTab] = useState<'size' | 'architecture' | 'sensitivity'>('size');
  const [selectedSizes, setSelectedSizes] = useState<number[]>([50]);
  const [selectedBetas, setSelectedBetas] = useState<number[]>([10.0]);
  const [selectedLambdas, setSelectedLambdas] = useState<number[]>([10.0]);
  const [selectedRepeats, setSelectedRepeats] = useState<number>(1);
  const [selectedEpochs, setSelectedEpochs] = useState<number | 'custom'>(500);
  const [customEpochs, setCustomEpochs] = useState<number>(500);
  const [ablationType, setAblationType] = useState<'size' | 'beta' | 'lambda'>('size');
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [taskView, setTaskView] = useState<'overview' | 'training' | 'comparative'>('overview');
  const [runComparativeData, setRunComparativeData] = useState<any[] | null>(null);
  const [runSyncData, setRunSyncData] = useState<any | null>(null);
  const [runDegData, setRunDegData] = useState<any | null>(null);
  const [runPredictionData, setRunPredictionData] = useState<any | null>(null);
  const [runPathwayData, setRunPathwayData] = useState<any | null>(null);
  const [selectedPathwayLibrary, setSelectedPathwayLibrary] = useState<string | null>(null);
  const [ablationData, setAblationData] = useState<any[]>([]);
  const [ablationLogs, setAblationLogs] = useState<any[] | null>(null);
  const [viewingAblationCategory, setViewingAblationCategory] = useState<string | null>(null);
  const [corrGroup, setCorrGroup] = useState<'MA' | 'RS'>('MA');
  
  const [resultsStatus, setResultsStatus] = useState<{
    checkpoints: string[], 
    logs: string[],
    run_statuses?: Record<string, {
      training: 'running' | 'completed' | 'idle',
      sync: boolean,
      comparative: boolean,
      deg: boolean,
      pathway: boolean,
      pred_model: boolean
    }>
  }>({checkpoints: [], logs: []});

  const [metrics, setMetrics] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Log Viewer State
  const [viewingLog, setViewingLog] = useState<string | null>(null);
  const [logData, setLogData] = useState<LogResponse | null>(null);
  const [previouslySelected, setPreviouslySelected] = useState<string[]>([]);

  useEffect(() => {
    if (selectedRunId && !previouslySelected.includes(selectedRunId)) {
      setPreviouslySelected(prev => [selectedRunId, ...prev].slice(0, 5));
    }
  }, [selectedRunId]);

  const [logMode, setLogMode] = useState<'structured' | 'chart'>('chart');
  const [visibleMetrics, setVisibleMetrics] = useState<string[]>(['G_A', 'G_B', 'D_A', 'D_B', 'Cycle', 'Feedback']);
  
  // Pagination & Search State
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 20;

  const sizes = [10, 20, 50, 100, 200];
  const betas = [0.0, 1.0, 5.0, 10.0, 50.0];
  const lambdas = [0.0, 1.0, 5.0, 10.0, 50.0];

  const handleStartSession = async () => {
    const project = projects.find(p => p.id === selectedProject);
    if (!project) return;
    
    // In one-dimensional ablation, only the active type has multiple values
    const payload = {
      config_path: project.config_path,
      sizes: ablationType === 'size' ? selectedSizes : [50],
      betas: ablationType === 'beta' ? selectedBetas : [10.0],
      lambdas: ablationType === 'lambda' ? selectedLambdas : [10.0],
      repeats: selectedRepeats,
      epochs: selectedEpochs === 'custom' ? customEpochs : selectedEpochs
    };

    try {
      await axios.post(`${API_BASE}/train`, payload);
      alert("Training session started!");
    } catch (err) {
      console.error(err);
      alert("Failed to start session");
    }
  };

  const handleRunStep = async (step: number) => {
    if (!selectedRunId) return;
    const project = projects.find(p => selectedRunId.startsWith(p.id));
    
    try {
      await axios.post(`${API_BASE}/runs/${selectedRunId}/run_step`, null, {
        params: { 
          step,
          config_path: project?.config_path
        }
      });
      alert(`Step ${step} started in a new console.`);
    } catch (err) {
      console.error(err);
      alert(`Failed to start step ${step}`);
    }
  };

  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const projRes = await axios.get(`${API_BASE}/projects`);
        setProjects(projRes.data);
        if (projRes.data.length > 0 && !selectedProject) setSelectedProject(projRes.data[0].id);
      } catch (err) {
        console.error("Failed to fetch projects", err);
      } finally {
        setLoading(false);
      }
    };
    fetchProjects();
  }, []); // Only fetch projects once on mount

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const resStatus = await axios.get(`${API_BASE}/results`);
        setResultsStatus(resStatus.data);
      } catch (err) {
        console.error("Failed to fetch results status", err);
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000); // Poll results every 30 seconds
    return () => clearInterval(interval);
  }, []); // Poll status globally

  useEffect(() => {
    if (selectedProject) {
      axios.get(`${API_BASE}/projects/${selectedProject}/ablation`)
        .then(res => setAblationData(res.data))
        .catch(err => console.error(err));
    }
  }, [selectedProject, resultsStatus]);


  const toggleSelection = (val: any, list: any[], setter: (val: any[]) => void) => {
    if (list.includes(val)) {
      setter(list.filter(item => item !== val));
    } else {
      setter([...list, val]);
    }
  };

  const fetchLogs = (runId: string) => {
    setViewingLog(runId);
    setLogData(null);
    setSearchTerm('');
    setCurrentPage(1);
    axios.get(`${API_BASE}/runs/${runId}/logs`)
      .then(res => setLogData(res.data))
      .catch(() => setLogData({ run_id: runId, summary: {}, structured: [], total_lines: 0 }));
  };

  const chartData = useMemo(() => logData?.structured.map((d, i) => ({
    name: `E${d.epoch}`,
    index: i,
    G_A: d.G_A || 0, G_B: d.G_B || 0, D_A: d.D_A || 0, D_B: d.D_B || 0,
    Cycle: (d.cycle_A || 0) + (d.cycle_B || 0),
    Feedback: (d.feedback_A || 0) + (d.feedback_B || 0),
    IDT: (d.idt_A || 0) + (d.idt_B || 0),
  })) || [], [logData]);

  const filteredData = useMemo(() => {
    if (!logData?.structured) return [];
    if (!searchTerm) return logData.structured;
    const term = searchTerm.toLowerCase();
    return logData.structured.filter(d => 
      d.epoch.toString().includes(term) || 
      d.iters.toString().includes(term)
    );
  }, [logData, searchTerm]);

  const tableData = useMemo(() => {
    const reversed = [...filteredData].reverse();
    const start = (currentPage - 1) * pageSize;
    return reversed.slice(start, start + pageSize);
  }, [filteredData, currentPage]);

  const totalPages = Math.ceil(filteredData.length / pageSize);

  const StatusButton = ({ label, status }: { label: string, status: 'running' | 'completed' | 'idle' | 'unavailable' | boolean }) => {
    let className = "status-badge ";
    let style: React.CSSProperties = {
      fontSize: '0.65rem', padding: '2px 6px', borderRadius: '4px', whiteSpace: 'nowrap',
    };
    if (status === 'running') {
      className += "status-running";
      style = { ...style, background: '#1890ff', color: 'white', animation: 'pulse 1.5s infinite' };
    } else if (status === 'completed' || status === true) {
      className += "status-success";
      style = { ...style, background: '#52c41a', color: 'white' };
    } else if (status === 'unavailable') {
      className += "status-disabled";
      style = { ...style, background: '#f5f5f5', color: '#bfbfbf', border: '1px solid #d9d9d9', opacity: 0.8 };
    } else {
      className += "status-error";
      style = { ...style, border: '1px solid #ff4d4f', color: '#ff4d4f', opacity: 0.6 };
    }
    return <div className={className} style={style}>{label}</div>;
  };

  const fetchComparativeMetrics = (runId: string) => {
    setTaskView('comparative');
    setRunComparativeData(null);
    axios.get(`${API_BASE}/runs/${runId}/comparative`)
      .then(res => setRunComparativeData(res.data))
      .catch(() => setRunComparativeData([]));
  };

  const fetchSyncStatus = (runId: string) => {
    setTaskView('sync');
    setRunSyncData(null);
    axios.get(`${API_BASE}/runs/${runId}/sync`)
      .then(res => setRunSyncData(res.data))
      .catch(() => setRunSyncData({ exists: false }));
  };

  const renderSyncStatus = () => {
    if (!runSyncData) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;
    if (!runSyncData.exists) return <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>Sync data not found for this run.</div>;

    const details = runSyncData.details;
    const platforms = ["Microarray", "RNA-Seq"];
    const CheckIcon = ({ checked }: { checked: boolean }) => (
      <div style={{ 
        color: checked ? 'var(--success-color)' : '#ff4d4f', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        fontWeight: 'bold'
      }}>
        {checked ? "✓" : "✗"}
      </div>
    );

    return (
      <div className="card" style={{ padding: '2rem' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid var(--border-color)', textAlign: 'left' }}>
              <th style={{ padding: '1rem' }}>Category / Row</th>
              {platforms.map(p => <th key={p} style={{ padding: '1rem', textAlign: 'center' }}>{p}</th>)}
            </tr>
          </thead>
          <tbody>
            <tr style={{ backgroundColor: '#f9fafb', fontWeight: 'bold' }}>
              <td colSpan={3} style={{ padding: '0.75rem 1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>GANOMICS (TRAINING)</td>
            </tr>
            {["Real", "Fake"].map(type => (
              <tr key={`train-${type}`} style={{ borderBottom: '1px solid var(--border-color)' }}>
                <td style={{ padding: '0.75rem 1rem', paddingLeft: '2rem' }}>{type}</td>
                {platforms.map(p => (
                  <td key={p} style={{ textAlign: 'center' }}>
                    <CheckIcon checked={details.train[p]?.[type]} />
                  </td>
                ))}
              </tr>
            ))}

            <tr style={{ backgroundColor: '#f9fafb', fontWeight: 'bold' }}>
              <td colSpan={3} style={{ padding: '0.75rem 1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>GANOMICS (TESTING)</td>
            </tr>
            {["Real", "Fake"].map(type => (
              <tr key={`test-${type}`} style={{ borderBottom: '1px solid var(--border-color)' }}>
                <td style={{ padding: '0.75rem 1rem', paddingLeft: '2rem' }}>{type}</td>
                {platforms.map(p => (
                  <td key={p} style={{ textAlign: 'center' }}>
                    <CheckIcon checked={details.test[p]?.[type]} />
                  </td>
                ))}
              </tr>
            ))}

            <tr style={{ backgroundColor: '#f9fafb', fontWeight: 'bold' }}>
              <td colSpan={3} style={{ padding: '0.75rem 1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>BASELINE ALGORITHMS (FAKE ONLY)</td>
            </tr>
            {["ComBat", "CuBlock", "QN", "TDM", "YuGene"].map(algo => (
              <tr key={algo} style={{ borderBottom: '1px solid var(--border-color)' }}>
                <td style={{ padding: '0.75rem 1rem', paddingLeft: '2rem' }}>{algo}</td>
                {platforms.map(p => (
                  <td key={p} style={{ textAlign: 'center' }}>
                    <CheckIcon checked={details.algorithms[p]?.[algo]} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const fetchDegMetrics = (runId: string) => {
    setTaskView('deg');
    setRunDegData(null);
    axios.get(`${API_BASE}/runs/${runId}/deg`)
      .then(res => setRunDegData(res.data))
      .catch(() => setRunDegData({}));
  };

  const fetchPredictionMetrics = (runId: string) => {
    setTaskView('prediction');
    setRunPredictionData(null);
    axios.get(`${API_BASE}/runs/${runId}/prediction`)
      .then(res => setRunPredictionData(res.data))
      .catch(() => setRunPredictionData({}));
  };

  const fetchPathwayMetrics = (runId: string) => {
    setTaskView('pathway');
    setRunPathwayData(null);
    setSelectedPathwayLibrary(null);
    axios.get(`${API_BASE}/runs/${runId}/pathway`)
      .then(res => {
        setRunPathwayData(res.data);
        const libraries = Object.keys(res.data);
        if (libraries.length > 0) setSelectedPathwayLibrary(libraries[0]);
      })
      .catch(() => setRunPathwayData({}));
  };

  const renderPathwayAnalysis = () => {
    if (!runPathwayData) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;
    const libraries = Object.keys(runPathwayData);
    if (libraries.length === 0) 
      return <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>No pathway results found for this run.</div>;

    const currentLib = selectedPathwayLibrary || libraries[0];
    const libData = runPathwayData[currentLib];
    
    if (!libData || !libData.concordance) return null;

    const algos = Object.keys(libData.concordance);
    const chartData = algos.map(algo => ({
      name: algo,
      rho: libData.concordance[algo].Spearman_Rho,
      p: libData.concordance[algo].P_Value
    })).sort((a,b) => b.rho - a.rho);

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
        <section className="card" style={{ padding: '2rem' }}>
          <div style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h3 style={{ margin: 0 }}>Pathway Concordance (Spearman ρ)</h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                Measures the correlation of pathway enrichment scores between real and synthetic data.
              </p>
            </div>
            
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {libraries.map(lib => (
                <button 
                  key={lib} 
                  className={`chip ${selectedPathwayLibrary === lib ? 'selected' : ''}`}
                  onClick={() => setSelectedPathwayLibrary(lib)}
                >
                  {lib.replace(/_/g, ' ')}
                </button>
              ))}
            </div>
          </div>

          <div style={{ height: '400px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" angle={-45} textAnchor="end" interval={0} fontSize={11} />
                <YAxis domain={[0, 1]} label={{ value: 'Spearman Rho', angle: -90, position: 'insideLeft', fontSize: 12 }} />
                <Tooltip 
                  contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 8px 24px rgba(0,0,0,0.12)' }}
                  formatter={(val: number) => val.toFixed(4)}
                />
                <Bar dataKey="rho" fill="var(--primary-color)" radius={[4, 4, 0, 0]} barSize={40} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={{ marginTop: '2rem' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ textAlign: 'left', backgroundColor: '#f9fafb', borderBottom: '2px solid var(--border-color)' }}>
                  <th style={{ padding: '1rem' }}>Algorithm</th>
                  <th style={{ padding: '1rem' }}>Spearman Rho (ρ)</th>
                  <th style={{ padding: '1rem' }}>P-Value</th>
                  <th style={{ padding: '1rem' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {chartData.map((d) => (
                  <tr key={d.name} style={{ borderBottom: '1px solid #f3f4f6' }} className="hover-row">
                    <td style={{ padding: '0.75rem 1rem', fontWeight: '600' }}>{d.name}</td>
                    <td style={{ padding: '0.75rem 1rem' }}>{d.rho.toFixed(4)}</td>
                    <td style={{ padding: '0.75rem 1rem' }}>{d.p.toFixed(4)}</td>
                    <td style={{ padding: '0.75rem 1rem' }}>
                      <span style={{ 
                        padding: '0.2rem 0.5rem', 
                        borderRadius: '4px', 
                        fontSize: '0.7rem',
                        backgroundColor: d.p < 0.05 ? 'var(--success-color-bg)' : '#fff1f0',
                        color: d.p < 0.05 ? 'var(--success-color)' : '#ff4d4f',
                        fontWeight: 'bold'
                      }}>
                        {d.p < 0.05 ? 'Significant' : 'Not Significant'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    );
  };

  const renderPredictionAnalysis = () => {
    if (!runPredictionData) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;
    if (Object.keys(runPredictionData).length === 0) return <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>No prediction results found for this run.</div>;

    const scenarios = ["Real->Real", "Real->Syn", "Syn->Real", "Syn->Syn"];
    const algos = Object.keys(runPredictionData);
    
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
        <section className="card">
          <div style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <h3 style={{ margin: 0 }}>Classifier Performance (RandomForest, 100 Trees)</h3>
              <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Evaluates how well a classifier trained on one domain performs on another. 
                Focus on <b>Syn-&gt;Real</b> to measure synthetic data utility.
              </p>
            </div>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ textAlign: 'left', backgroundColor: '#f9fafb', borderBottom: '2px solid var(--border-color)' }}>
                  <th style={{ padding: '1rem' }}>Algorithm</th>
                  <th style={{ padding: '1rem' }}>Scenario</th>
                  <th style={{ padding: '1rem' }}>MCC</th>
                  <th style={{ padding: '1rem' }}>Accuracy</th>
                  <th style={{ padding: '1rem' }}>Precision</th>
                  <th style={{ padding: '1rem' }}>Recall</th>
                  <th style={{ padding: '1rem' }}>F1-Score</th>
                </tr>
              </thead>
              <tbody>
                {algos.sort((a,b) => a === 'GANomics' ? -1 : 1).map((algo) => (
                  <React.Fragment key={algo}>
                    {scenarios.map((scenario, idx) => {
                      const data = runPredictionData[algo].find((d: any) => d.Scenario === scenario);
                      if (!data) return null;
                      return (
                        <tr key={`${algo}-${scenario}`} style={{ 
                          borderBottom: idx === 3 ? '2px solid var(--border-color)' : '1px solid #f3f4f6',
                          backgroundColor: algo === 'GANomics' ? 'rgba(24, 144, 255, 0.02)' : 'transparent'
                        }} className="hover-row">
                          <td style={{ padding: '0.75rem 1rem', fontWeight: idx === 0 ? '700' : '400', color: idx === 0 ? 'inherit' : 'transparent' }}>
                            {algo}
                          </td>
                          <td style={{ padding: '0.75rem 1rem', fontWeight: '500', color: 'var(--text-muted)' }}>{scenario}</td>
                          <td style={{ padding: '0.75rem 1rem', fontWeight: scenario === 'Syn->Real' ? '700' : '400', color: scenario === 'Syn->Real' && data.MCC > 0.7 ? 'var(--success-color)' : 'inherit' }}>{data.MCC.toFixed(4)}</td>
                          <td style={{ padding: '0.75rem 1rem' }}>{data.Accuracy.toFixed(4)}</td>
                          <td style={{ padding: '0.75rem 1rem' }}>{data.Precision.toFixed(4)}</td>
                          <td style={{ padding: '0.75rem 1rem' }}>{data.Recall.toFixed(4)}</td>
                          <td style={{ padding: '0.75rem 1rem' }}>{data.F1.toFixed(4)}</td>
                        </tr>
                      );
                    })}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    );
  };

  const renderDegAnalysis = () => {
    if (!runDegData) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;
    if (Object.keys(runDegData).length === 0) return <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>No DEG results found for this run.</div>;

    const algos = Object.keys(runDegData);
    const colors = {
      GANomics: 'var(--primary-color)',
      ComBat: '#818cf8',
      CuBlock: '#ec4899',
      QN: '#f59e0b',
      TDM: '#10b981',
      YuGene: '#6366f1'
    };

    // Prepare data for Recharts: array of { threshold, GANomics, ComBat, ... }
    // First, find all unique thresholds
    const allThresholds = Array.from(new Set(
      Object.values(runDegData).flatMap((points: any) => points.map((p: any) => p.threshold))
    )).sort((a: any, b: any) => a - b);

    const chartData = allThresholds.map(t => {
      const entry: any = { threshold: t, label: `p<${t}` };
      algos.forEach(algo => {
        const point = runDegData[algo].find((p: any) => p.threshold === t);
        if (point) entry[algo] = point.jaccard;
      });
      return entry;
    });

    return (
      <div className="card" style={{ padding: '2rem' }}>
        <div style={{ marginBottom: '2rem' }}>
          <h3 style={{ margin: 0 }}>Jaccard Similarity (Bio-Marker Preservation)</h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            Measures the overlap between differentially expressed genes (DEGs) in real vs. fake data. 
            Higher jaccard index indicates better preservation of biological signals.
          </p>
        </div>

        <div style={{ height: '500px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="label" fontSize={11} />
              <YAxis domain={[0, 1]} fontSize={11} label={{ value: 'Jaccard Index', angle: -90, position: 'insideLeft', fontSize: 12 }} />
              <Tooltip 
                contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 8px 24px rgba(0,0,0,0.12)' }}
                formatter={(val: number) => val.toFixed(4)}
              />
              <Legend verticalAlign="top" align="right" height={36} iconType="circle" />
              {algos.map(algo => (
                <Line 
                  key={algo} 
                  type="monotone" 
                  dataKey={algo} 
                  stroke={colors[algo as keyof typeof colors] || '#94a3b8'} 
                  strokeWidth={algo === 'GANomics' ? 3 : 2}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  const renderComparativeAnalysis = () => {
    if (!runComparativeData) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;
    if (runComparativeData.length === 0) return <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>No comparative data found for this run.</div>;

    const parseVal = (str: string) => {
      if (!str || typeof str !== 'string') return 0;
      const match = str.match(/([-\d.]+)/);
      return match ? parseFloat(match[1]) : 0;
    };

    const processed = runComparativeData.map(d => ({
      ...d,
      pearsonNum: parseVal(d.Pearson),
      spearmanNum: parseVal(d.Spearman),
      maeNum: parseVal(d.L1 || d.MAE),
      isExtreme: parseVal(d.L1 || d.MAE) > 100000 
    }));

    // Filter for Correlation Section
    const allCorrData = processed.filter(d => 
      d.Algorithm.includes(`(${corrGroup})`) || d.Algorithm.toLowerCase().includes('baseline')
    );

    const baselineEntry = allCorrData.find(d => d.Algorithm.toLowerCase().includes('baseline'));
    const chartData = allCorrData.filter(d => !d.Algorithm.toLowerCase().includes('baseline'));
    
    const baselinePearson = baselineEntry?.pearsonNum || 0;
    const baselineSpearman = baselineEntry?.spearmanNum || 0;

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
        {/* SECTION 1: Correlation Performance */}
        <section className="card">
          <div style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <h3 style={{ margin: 0 }}>1. Correlation Performance (Alignment Strength)</h3>
              <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.8rem', color: 'var(--text-muted)' }}>How well the trends match (Range: -1 to 1, Higher is Better)</p>
            </div>
            <div className="chip-grid" style={{ gap: '0.5rem' }}>
              <button className={`chip ${corrGroup === 'MA' ? 'selected' : ''}`} onClick={() => setCorrGroup('MA')}>Microarray (MA)</button>
              <button className={`chip ${corrGroup === 'RS' ? 'selected' : ''}`} onClick={() => setCorrGroup('RS')}>RNA-Seq (RS)</button>
            </div>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '2rem', alignItems: 'start' }}>
            {/* Correlation Chart */}
            <div style={{ height: '450px', backgroundColor: '#fcfcfc', borderRadius: '8px', padding: '1rem' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 20, right: 10, left: 0, bottom: 90 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
                  <XAxis 
                    dataKey="Algorithm" 
                    angle={-45} 
                    textAnchor="end" 
                    interval={0} 
                    fontSize={11} 
                    tick={{ fill: 'var(--text-main)' }}
                    height={100}
                  />
                  <YAxis 
                    domain={[0, 1]} 
                    fontSize={11}
                    label={{ value: 'Correlation Coefficient', angle: -90, position: 'insideLeft', offset: 10, fontSize: 12, fontWeight: 500 }} 
                  />
                  <Tooltip 
                    cursor={{ fill: 'rgba(0,0,0,0.04)' }} 
                    contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 8px 24px rgba(0,0,0,0.12)' }} 
                  />
                  <Legend 
                    verticalAlign="top" 
                    align="right" 
                    wrapperStyle={{ paddingBottom: '20px', fontSize: '11px' }} 
                    payload={[
                      { value: 'Pearson (r)', type: 'rect', color: 'var(--primary-color)' },
                      { value: 'Spearman (ρ)', type: 'rect', color: '#818cf8' },
                      { value: `Baseline (r): ${baselinePearson.toFixed(3)}`, type: 'plainline', color: 'var(--primary-color)' },
                      { value: `Baseline (ρ): ${baselineSpearman.toFixed(3)}`, type: 'plainline', color: '#818cf8' },
                    ]}
                  />
                  
                  {/* Baseline Reference Lines */}
                  <ReferenceLine 
                    y={baselinePearson} 
                    stroke="var(--primary-color)" 
                    strokeDasharray="5 5" 
                    strokeWidth={2} 
                  />
                  <ReferenceLine 
                    y={baselineSpearman} 
                    stroke="#818cf8" 
                    strokeDasharray="5 5" 
                    strokeWidth={2} 
                  />

                  <Bar dataKey="pearsonNum" name="Pearson (r)" fill="var(--primary-color)" radius={[4, 4, 0, 0]} barSize={24} />
                  <Bar dataKey="spearmanNum" name="Spearman (ρ)" fill="#818cf8" radius={[4, 4, 0, 0]} barSize={24} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Correlation Table */}
            <div style={{ overflowX: 'auto', border: '1px solid var(--border-color)', borderRadius: '8px' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                <thead>
                  <tr style={{ textAlign: 'left', backgroundColor: '#f9fafb', borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '0.75rem' }}>Algorithm</th>
                    <th style={{ padding: '0.75rem' }}>Pearson (r)</th>
                    <th style={{ padding: '0.75rem' }}>Spearman (ρ)</th>
                  </tr>
                </thead>
                <tbody>
                  {allCorrData.map((m, i) => {
                    const isBaseline = m.Algorithm.toLowerCase().includes('baseline');
                    return (
                      <tr key={i} style={{ 
                        borderBottom: '1px solid #f3f4f6',
                        backgroundColor: isBaseline ? '#f0f9ff' : 'transparent',
                        fontWeight: isBaseline ? 'bold' : 'normal'
                      }} className="hover-row">
                        <td style={{ padding: '0.75rem', fontWeight: '600' }}>{m.Algorithm}</td>
                        <td style={{ padding: '0.75rem', color: m.pearsonNum > 0.9 ? 'var(--success-color)' : 'inherit', fontWeight: m.pearsonNum > 0.9 ? '700' : '400' }}>{m.Pearson}</td>
                        <td style={{ padding: '0.75rem', color: m.spearmanNum > 0.9 ? 'var(--success-color)' : 'inherit', fontWeight: m.spearmanNum > 0.9 ? '700' : '400' }}>{m.Spearman}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* SECTION 2: Error Performance */}
        <section className="card">
          <div style={{ marginBottom: '1.5rem' }}>
            <h3 style={{ margin: 0 }}>2. Error Performance (Deviation)</h3>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.8rem', color: 'var(--text-muted)' }}>Average distance between actual and predicted values (Lower is Better)</p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', alignItems: 'start' }}>
            {/* Error Chart (Log Scale) */}
            <div style={{ height: '300px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={processed} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
                  <XAxis dataKey="Algorithm" angle={-30} textAnchor="end" interval={0} fontSize={10} />
                  <YAxis 
                    scale="log" 
                    domain={[1, 'auto']} 
                    label={{ value: 'Error (log)', angle: -90, position: 'insideLeft', fontSize: 10 }}
                    tickFormatter={(val) => val.toExponential(0)}
                  />
                  <Tooltip 
                    cursor={{ fill: '#f8f9fa' }}
                    formatter={(value: any) => [parseFloat(value).toExponential(2), "Error"]}
                    contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                  />
                  <Bar dataKey="maeNum" name="MAE / L1" fill="#ec4899" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Error Table */}
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                <thead>
                  <tr style={{ textAlign: 'left', borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '0.5rem' }}>Algorithm</th>
                    <th style={{ padding: '0.5rem' }}>MAE / L1</th>
                  </tr>
                </thead>
                <tbody>
                  {processed.map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f9fafb' }}>
                      <td style={{ padding: '0.5rem', fontWeight: '600' }}>{m.Algorithm}</td>
                      <td style={{ padding: '0.5rem', color: m.isExtreme ? '#ef4444' : 'inherit' }}>
                        {m.isExtreme ? m.maeNum.toExponential(4) : (m.L1 || m.MAE)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </div>
    );
  };

  const renderLogViewer = () => {
    if (!logData) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;
    
    // Meta-panel calculation
    const first = logData.structured[0] || {};
    const last = logData.structured[logData.structured.length - 1] || {};
    const sampleSizeMatch = viewingLog?.match(/Size_(\d+)/);
    const sampleSize = sampleSizeMatch ? sampleSizeMatch[1] : "N/A";

    const getLossValue = (row: any, key: string) => {
      if (['G_A', 'G_B', 'D_A', 'D_B'].includes(key)) return row[key];
      if (key === 'Cycle') return (row.cycle_A || 0) + (row.cycle_B || 0);
      if (key === 'Feedback') return (row.feedback_A || 0) + (row.feedback_B || 0);
      if (key === 'IDT') return (row.idt_A || 0) + (row.idt_B || 0);
      return 0;
    };

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        {/* Meta Panel */}
        <section className="card" style={{ padding: '1rem' }}>
          <h3 style={{ fontSize: '0.9rem', marginBottom: '1rem' }}>Training Process Summary</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem' }}>
            <div style={{ padding: '0.75rem', backgroundColor: '#f8f9fa', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 'bold' }}>SAMPLES (N)</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>{sampleSize}</div>
            </div>
            <div style={{ padding: '0.75rem', backgroundColor: '#f8f9fa', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 'bold' }}>TOTAL EPOCHS</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>{last.epoch || 0}</div>
            </div>
            <div style={{ padding: '0.75rem', backgroundColor: '#f8f9fa', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 'bold' }}>ITERATIONS</div>
              <div style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>{logData.total_lines}</div>
            </div>
          </div>
          
          <div style={{ marginTop: '1rem', overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: '0.75rem', textAlign: 'left', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                  <th style={{ padding: '0.5rem' }}>Loss Metric</th>
                  <th style={{ padding: '0.5rem' }}>Initial (E{first.epoch})</th>
                  <th style={{ padding: '0.5rem' }}>Final (E{last.epoch})</th>
                  <th style={{ padding: '0.5rem' }}>Improvement</th>
                </tr>
              </thead>
              <tbody>
                {LOSS_METRICS.map(m => {
                  const vInit = getLossValue(first, m.key);
                  const vFinal = getLossValue(last, m.key);
                  const diff = vInit - vFinal;
                  const pct = vInit !== 0 ? ((diff / vInit) * 100).toFixed(1) : "0.0";
                  return (
                    <tr key={m.key} style={{ borderBottom: '1px solid #f0f0f0' }}>
                      <td style={{ padding: '0.5rem', fontWeight: '600' }}>{m.label}</td>
                      <td style={{ padding: '0.5rem' }}>{vInit?.toFixed(4)}</td>
                      <td style={{ padding: '0.5rem', fontWeight: 'bold', color: m.color }}>{vFinal?.toFixed(4)}</td>
                      <td style={{ padding: '0.5rem', color: diff >= 0 ? 'var(--success-color)' : '#dc3545' }}>
                        {diff >= 0 ? '↓' : '↑'} {Math.abs(diff).toFixed(4)} ({pct}%)
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        {/* Existing Charts/Tables */}
        <section className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div className="chip-grid">
              <div className={`chip ${logMode === 'chart' ? 'selected' : ''}`} style={{ padding: '0.2rem 0.6rem', fontSize: '0.8rem' }} onClick={() => setLogMode('chart')}><LineChartIcon size={12} /> Figure</div>
              <div className={`chip ${logMode === 'structured' ? 'selected' : ''}`} style={{ padding: '0.2rem 0.6rem', fontSize: '0.8rem' }} onClick={() => setLogMode('structured')}><TableIcon size={12} /> Table</div>
            </div>
          </div>

          <div style={{ borderRadius: '8px', border: '1px solid var(--border-color)', backgroundColor: '#fff', display: 'flex', flexDirection: 'column' }}>
            {logMode === 'chart' ? (
              <div style={{ width: '100%', height: '400px', padding: '1.5rem' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" interval="preserveStartEnd" minTickGap={50} fontSize={10} />
                    <YAxis domain={['auto', 'auto']} fontSize={10} />
                    <Tooltip />
                    <Legend iconSize={10} wrapperStyle={{fontSize: '10px'}} />
                    {LOSS_METRICS.map(m => visibleMetrics.includes(m.key) && (<Line key={m.key} type="monotone" dataKey={m.key} stroke={m.color} dot={false} strokeWidth={1.5} animationDuration={300} isAnimationActive={chartData.length < 1000} />))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <>
                <div style={{ padding: '0.75rem', borderBottom: '1px solid var(--border-color)', display: 'flex', gap: '1rem', alignItems: 'center', backgroundColor: '#fdfdfd' }}>
                  <div style={{ position: 'relative', flex: 1, maxWidth: '300px' }}>
                    <Search size={14} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                    <input 
                      type="text" placeholder="Search epoch or iteration..." 
                      style={{ padding: '0.4rem 0.75rem 0.4rem 2rem', width: '100%', borderRadius: '6px', border: '1px solid var(--border-color)', fontSize: '0.85rem' }} 
                      value={searchTerm} onChange={(e) => {setSearchTerm(e.target.value); setCurrentPage(1);}}
                    />
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Page {currentPage} of {totalPages || 1}</div>
                </div>
                <div style={{ flex: 1, overflowY: 'auto', maxHeight: '400px' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                    <thead style={{ position: 'sticky', top: 0, backgroundColor: '#fff', zIndex: 1 }}><tr style={{ textAlign: 'left', borderBottom: '2px solid var(--border-color)' }}><th style={{ padding: '0.4rem' }}>Epoch</th><th style={{ padding: '0.4rem' }}>Iters</th><th style={{ padding: '0.4rem' }}>G_A</th><th style={{ padding: '0.4rem' }}>G_B</th><th style={{ padding: '0.4rem' }}>D_A</th><th style={{ padding: '0.4rem' }}>D_B</th><th style={{ padding: '0.4rem' }}>Cycle</th><th style={{ padding: '0.4rem' }}>Feedback</th></tr></thead>
                    <tbody>{tableData.map((row, i) => (<tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}><td style={{ padding: '0.4rem', fontWeight: '600' }}>{row.epoch}</td><td style={{ padding: '0.4rem' }}>{row.iters}</td><td style={{ padding: '0.4rem' }}>{row.G_A?.toFixed(3)}</td><td style={{ padding: '0.4rem' }}>{row.G_B?.toFixed(3)}</td><td style={{ padding: '0.4rem' }}>{row.D_A?.toFixed(3)}</td><td style={{ padding: '0.4rem' }}>{row.D_B?.toFixed(3)}</td><td style={{ padding: '0.4rem' }}>{((row.cycle_A || 0) + (row.cycle_B || 0)).toFixed(3)}</td><td style={{ padding: '0.4rem' }}>{((row.feedback_A || 0) + (row.feedback_B || 0)).toFixed(3)}</td></tr>))}</tbody>
                  </table>
                </div>
                <div style={{ padding: '0.75rem', borderTop: '1px solid var(--border-color)', display: 'flex', justifyContent: 'center', gap: '0.5rem', backgroundColor: '#fdfdfd' }}>
                  <button className="chip" style={{ padding: '0.2rem 0.5rem' }} disabled={currentPage === 1} onClick={() => setCurrentPage(1)}><ChevronsLeft size={14} /></button>
                  <button className="chip" style={{ padding: '0.2rem 0.5rem' }} disabled={currentPage === 1} onClick={() => setCurrentPage(prev => prev - 1)}><ChevronLeft size={14} /></button>
                  <span style={{ display: 'flex', alignItems: 'center', paddingLeft: '1rem', paddingRight: '1rem', fontSize: '0.85rem' }}>{currentPage} / {totalPages || 1}</span>
                  <button className="chip" style={{ padding: '0.2rem 0.5rem' }} disabled={currentPage >= totalPages} onClick={() => setCurrentPage(prev => prev + 1)}><ChevronRight size={14} /></button>
                  <button className="chip" style={{ padding: '0.2rem 0.5rem' }} disabled={currentPage >= totalPages} onClick={() => setCurrentPage(totalPages)}><ChevronsRight size={14} /></button>
                </div>
              </>
            )}
          </div>
        </section>
      </div>
    );
  };

  const [statusFilters, setStatusFilters] = useState({ architecture: 'all', size: 'all', sensitivity: 'all' });
  const [globalStepFilter, setGlobalStepFilter] = useState<string>('all');
  const [collapsedPanels, setCollapsedPanels] = useState<Record<string, boolean>>({});
  const [sensitivityType, setSensitivityType] = useState<'beta' | 'lambda'>('beta');

  const renderAblationCharts = () => {
    if (ablationData.length === 0) return null;

    // 1. Sample Size Data
    const sizeData = ablationData
      .filter(d => d.run_id.includes("Size") && !d.run_id.includes("Architecture"))
      .map(d => {
        const match = d.run_id.match(/Size_(\d+)/);
        return { 
          size: match ? parseInt(match[1]) : 0, 
          MAE: parseFloat(d.L1),
          Pearson: parseFloat(d.Pearson),
          Spearman: parseFloat(d.Spearman)
        };
      })
      .filter(d => d.size > 0)
      .sort((a, b) => a.size - b.size);

    // Group by size and average (if multiple runs exist)
    const sizeAvgMap = new Map();
    sizeData.forEach(d => {
      if (!sizeAvgMap.has(d.size)) sizeAvgMap.set(d.size, { size: d.size, MAE: 0, count: 0 });
      const entry = sizeAvgMap.get(d.size);
      entry.MAE += d.MAE;
      entry.count += 1;
    });
    const sizeChartData = Array.from(sizeAvgMap.values()).map(v => ({ 
      size: v.size, 
      MAE: v.MAE / v.count 
    })).sort((a,b) => a.size - b.size);

    // 2. Architecture Data
    // Get baseline (Size 50 run)
    const baselineRun = ablationData.find(d => d.run_id.includes("Size_50") && !d.run_id.includes("Architecture"));
    const archRuns = ablationData.filter(d => d.run_id.includes("Architecture"));
    
    const archChartData = [
      { name: 'Baseline (Both)', MAE: baselineRun ? parseFloat(baselineRun.L1) : null },
      ...archRuns.map(d => ({
        name: d.run_id.includes("AtoB") ? "A → B Only" : "B → A Only",
        MAE: parseFloat(d.L1)
      }))
    ];

    // 3. Sensitivity Data
    const sensData = ablationData
      .filter(d => d.run_id.includes("Sensitivity"))
      .map(d => {
        const bMatch = d.run_id.match(/Beta_([\d.]+)/);
        const lMatch = d.run_id.match(/Lambda_([\d.]+)/);
        return {
          beta: bMatch ? parseFloat(bMatch[1]) : 10.0,
          lambda: lMatch ? parseFloat(lMatch[1]) : 10.0,
          MAE: parseFloat(d.L1)
        };
      });
    
    const sensitivityChartData = (sensitivityType === 'beta' 
      ? sensData.filter(d => d.lambda === 10.0).sort((a,b) => a.beta - b.beta)
      : sensData.filter(d => d.beta === 10.0).sort((a,b) => a.lambda - b.lambda)
    ).map(d => ({
      val: sensitivityType === 'beta' ? d.beta : d.lambda,
      MAE: d.MAE
    }));

    return (
      <div className="card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h3 style={{ margin: 0, fontSize: '1rem' }}>Ablation Analysis Results</h3>
          <div style={{ display: 'flex', gap: '0.25rem', backgroundColor: '#f3f4f6', padding: '0.25rem', borderRadius: '8px' }}>
            <button className={`chip ${projectAblationTab === 'size' ? 'selected' : ''}`} style={{ border: 'none', fontSize: '0.75rem' }} onClick={() => setProjectAblationTab('size')}>Sample Size</button>
            <button className={`chip ${projectAblationTab === 'architecture' ? 'selected' : ''}`} style={{ border: 'none', fontSize: '0.75rem' }} onClick={() => setProjectAblationTab('architecture')}>Architecture</button>
            <button className={`chip ${projectAblationTab === 'sensitivity' ? 'selected' : ''}`} style={{ border: 'none', fontSize: '0.75rem' }} onClick={() => setProjectAblationTab('sensitivity')}>Sensitivity</button>
          </div>
        </div>

        <div style={{ height: '300px', width: '100%' }}>
          {projectAblationTab === 'size' && (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={sizeChartData} margin={{ top: 5, right: 30, left: 20, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="size" label={{ value: 'Training Samples (N)', position: 'bottom', fontSize: 12 }} fontSize={11} />
                <YAxis label={{ value: 'MAE (L1 Loss)', angle: -90, position: 'insideLeft', fontSize: 12 }} fontSize={11} />
                <Tooltip />
                <Line type="monotone" dataKey="MAE" stroke="var(--primary-color)" strokeWidth={2} dot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          )}

          {projectAblationTab === 'architecture' && (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={archChartData} margin={{ top: 5, right: 30, left: 20, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" fontSize={11} />
                <YAxis label={{ value: 'MAE (L1 Loss)', angle: -90, position: 'insideLeft', fontSize: 12 }} fontSize={11} />
                <Tooltip />
                <Bar dataKey="MAE" fill="var(--primary-color)" radius={[4, 4, 0, 0]} barSize={50} />
              </BarChart>
            </ResponsiveContainer>
          )}

          {projectAblationTab === 'sensitivity' && (
            <div style={{ height: '100%' }}>
              <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', justifyContent: 'center' }}>
                <label style={{ fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <input type="radio" checked={sensitivityType === 'beta'} onChange={() => setSensitivityType('beta')} /> Beta (Feedback)
                </label>
                <label style={{ fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <input type="radio" checked={sensitivityType === 'lambda'} onChange={() => setSensitivityType('lambda')} /> Lambda (Cycle)
                </label>
              </div>
              <ResponsiveContainer width="100%" height="250px">
                <LineChart data={sensitivityChartData} margin={{ top: 5, right: 30, left: 20, bottom: 25 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="val" label={{ value: sensitivityType === 'beta' ? 'Beta' : 'Lambda', position: 'bottom', fontSize: 12 }} fontSize={11} />
                  <YAxis label={{ value: 'MAE', angle: -90, position: 'insideLeft', fontSize: 12 }} fontSize={11} />
                  <Tooltip />
                  <Line type="monotone" dataKey="MAE" stroke="var(--primary-color)" strokeWidth={2} dot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    );
  };

  const fetchAblationLogs = (category: string) => {
    setViewingAblationCategory(category);
    setAblationLogs(null);
    axios.get(`${API_BASE}/projects/${selectedProject}/ablation_logs`, { params: { category } })
      .then(res => setAblationLogs(res.data))
      .catch(err => console.error(err));
  };

  const renderAblationAnalyticsModal = () => {
    if (!viewingAblationCategory) return null;

    return (
      <div className="modal-overlay" style={{ zIndex: 2000 }}>
        <div className="modal-content" style={{ maxWidth: '1000px', width: '90%', maxHeight: '90vh', overflowY: 'auto' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
            <h2 style={{ margin: 0, textTransform: 'uppercase' }}>{viewingAblationCategory} Analytics: {selectedProject}</h2>
            <button className="chip" onClick={() => setViewingAblationCategory(null)}><X size={18} /></button>
          </div>

          {!ablationLogs ? (
            <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>
          ) : ablationLogs.length === 0 ? (
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>No logs found for this category.</div>
          ) : (() => {
            const variants: Record<string, any[]> = {};
            ablationLogs.forEach(log => {
              let variant = "Other";
              if (viewingAblationCategory === 'size') {
                const m = log.run_id.match(/Size_(\d+)/);
                if (m) variant = `Size ${m[1]}`;
              } else if (viewingAblationCategory === 'architecture') {
                if (log.run_id.includes("AtoB")) variant = "A → B Only";
                else if (log.run_id.includes("BtoA")) variant = "B → A Only";
                else variant = "Baseline (Both)";
              } else if (viewingAblationCategory === 'sensitivity') {
                const bm = log.run_id.match(/Beta_([\d.]+)/);
                const lm = log.run_id.match(/Lambda_([\d.]+)/);
                if (bm) variant = `Beta ${bm[1]}`;
                else if (lm) variant = `Lambda ${lm[1]}`;
              }
              if (!variants[variant]) variants[variant] = [];
              variants[variant].push(log);
            });

            const metricKeys = ['G_A', 'G_B', 'D_A', 'D_B', 'cycle_A', 'cycle_B', 'feedback_A', 'feedback_B'];
            const summaryData = Object.entries(variants).map(([name, logs]) => {
              const stats: any = { name };
              metricKeys.forEach(k => {
                const finalValues = logs.map(l => l.last[k] || 0);
                const avgFinal = (finalValues.reduce((a, b) => a + b, 0) / (finalValues.length || 1));
                const stdFinal = Math.sqrt(finalValues.map(x => Math.pow(x - avgFinal, 2)).reduce((a, b) => a + b, 0) / (finalValues.length || 1));
                stats[k] = { avgFinal, stdFinal };
              });
              return stats;
            });

            return (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                <section className="card">
                  <h3 style={{ fontSize: '0.9rem', marginBottom: '1rem' }}>Final Loss Values</h3>
                  <div style={{ height: '400px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={summaryData} margin={{ bottom: 60 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="name" angle={-45} textAnchor="end" interval={0} fontSize={11} />
                        <YAxis label={{ value: 'Avg. Final Loss', angle: -90, position: 'insideLeft' }} fontSize={11} />
                        <Tooltip />
                        <Legend verticalAlign="top" height={36}/>
                        <Bar dataKey="G_A.avgFinal" name="Gen A" fill="#007bff" />
                        <Bar dataKey="G_B.avgFinal" name="Gen B" fill="#0056b3" />
                        <Bar dataKey="D_A.avgFinal" name="Disc A" fill="#ef4444" />
                        <Bar dataKey="D_B.avgFinal" name="Disc B" fill="#991b1b" />
                        <Bar dataKey="cycle_A.avgFinal" name="Cycle A" fill="#10b981" />
                        <Bar dataKey="cycle_B.avgFinal" name="Cycle B" fill="#065f46" />
                        <Bar dataKey="feedback_A.avgFinal" name="Feedback A" fill="#f59e0b" />
                        <Bar dataKey="feedback_B.avgFinal" name="Feedback B" fill="#92400e" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </section>

                <section className="card">
                  <h3 style={{ fontSize: '0.9rem', marginBottom: '1rem' }}>Detailed Metrics Table (Mean ± Std)</h3>
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.7rem' }}>
                      <thead>
                        <tr style={{ backgroundColor: '#f9fafb', borderBottom: '2px solid var(--border-color)' }}>
                          <th style={{ padding: '0.75rem', textAlign: 'left' }}>Variant</th>
                          {metricKeys.map(k => (
                            <th key={k} style={{ padding: '0.75rem', textAlign: 'center' }}>
                              {k.replace('_', ' ')}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {summaryData.map(s => (
                          <tr key={s.name} style={{ borderBottom: '1px solid #f0f0f0' }}>
                            <td style={{ padding: '0.75rem', fontWeight: 'bold' }}>{s.name}</td>
                            {metricKeys.map(k => (
                              <td key={`${s.name}-${k}`} style={{ padding: '0.5rem', textAlign: 'center' }}>
                                {s[k].avgFinal.toFixed(3)} <span style={{ color: 'var(--text-muted)', fontSize: '0.6rem' }}>±{s[k].stdFinal.toFixed(3)}</span>
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              </div>
            );
          })()}
        </div>
      </div>
    );
  };

  const renderProjectStatus = () => {
    const projectLogs = resultsStatus.logs
      .filter(l => l.startsWith(selectedProject))
      .map(l => l.replace("_log.txt", ""));
    
    const categories = {
      architecture: projectLogs.filter(id => id.includes("Architecture")),
      size: projectLogs.filter(id => id.includes("Size") && !id.includes("Architecture")),
      sensitivity: projectLogs.filter(id => id.includes("Sensitivity")),
      other: projectLogs.filter(id => !id.includes("Architecture") && !id.includes("Size") && !id.includes("Sensitivity"))
    };

  const SubPanel = ({ title, items, filterKey }: { title: string, items: string[], filterKey: string }) => {
      if (items.length === 0) return null;
      const isCollapsed = collapsedPanels[filterKey] || false;
      
      const filteredItems = items.filter(id => {
        const status = resultsStatus.run_statuses?.[id];
        
        // 1. Global Step Filter
        if (globalStepFilter !== 'all') {
          if (globalStepFilter === 'training') {
            if (status?.training !== 'completed') return false;
          } else {
            const stepKey = globalStepFilter as keyof NonNullable<typeof status>;
            if (!status?.[stepKey]) return false;
          }
        }

        const key = filterKey as keyof typeof statusFilters;
        if (statusFilters[key] === 'all') return true;
        if (statusFilters[key] === 'running') return status?.training === 'running';
        if (statusFilters[key] === 'completed') return status?.training === 'completed';
        if (statusFilters[key] === 'pending') return status?.training === 'idle';
        return true;
      });

      return (
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }} onClick={() => setCollapsedPanels({ ...collapsedPanels, [filterKey]: !isCollapsed })}>
                {isCollapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
                <h4 style={{ margin: 0, fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{title} ({filteredItems.length})</h4>
              </div>
              <button 
                className="chip" 
                style={{ fontSize: '0.6rem', padding: '0.1rem 0.4rem', border: '1px solid var(--primary-color)', color: 'var(--primary-color)' }}
                onClick={() => fetchAblationLogs(filterKey)}
              >
                Ablation Analytics
              </button>
            </div>
            {!isCollapsed && (
              <select 
                value={statusFilters[filterKey as keyof typeof statusFilters]} 
                onChange={(e) => setStatusFilters({ ...statusFilters, [filterKey]: e.target.value })}
                style={{ fontSize: '0.7rem', padding: '2px', borderRadius: '4px', border: '1px solid var(--border-color)' }}
              >
                <option value="all">All Status</option>
                <option value="running">Running</option>
                <option value="completed">Completed</option>
                <option value="pending">Pending</option>
              </select>
            )}
          </div>
          {!isCollapsed && (
            <div className="queue-list">
              {filteredItems.map(runId => {
                const status = resultsStatus.run_statuses?.[runId];
                const isSizeTask = runId.includes("Size") && !runId.includes("Architecture");
                return (
                  <div key={runId} className="queue-item" onClick={() => { setSelectedRunId(runId); setTaskView('overview'); }} style={{ cursor: 'pointer', flexDirection: 'column', alignItems: 'flex-start', gap: '8px' }}>
                    <div style={{ fontWeight: '600', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      {runId}
                      {status?.training === 'running' && <Loader2 size={12} className="animate-spin" style={{ color: '#1890ff' }} />}
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                      <StatusButton label="Training" status={status?.training || 'idle'} />
                      <StatusButton label="Sync Data" status={status?.sync || false} />
                      <StatusButton label="Comparative" status={isSizeTask ? (status?.comparative || false) : 'unavailable'} />
                      <StatusButton label="DEG" status={isSizeTask ? (status?.deg || false) : 'unavailable'} />
                      <StatusButton label="Pathway" status={isSizeTask ? (status?.pathway || false) : 'unavailable'} />
                      <StatusButton label="Pred. Model" status={isSizeTask ? (status?.pred_model || false) : 'unavailable'} />
                    </div>
                  </div>
                );
              })}
              {filteredItems.length === 0 && <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'center', padding: '1rem' }}>No runs matching filter.</div>}
            </div>
          )}
        </div>
      );
    };

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {renderAblationCharts()}
        
        <div style={{ padding: '0.75rem', backgroundColor: '#f8f9fa', borderRadius: '8px', border: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 'bold', color: 'var(--text-muted)' }}>FILTER BY COMPLETION:</div>
            <select 
              value={globalStepFilter} 
              onChange={(e) => setGlobalStepFilter(e.target.value)}
              style={{ fontSize: '0.75rem', padding: '4px 8px', borderRadius: '6px', border: '1px solid var(--border-color)', backgroundColor: '#fff', fontWeight: '600' }}
            >
              <option value="all">Show All Tasks</option>
              <option value="training">Finished: 1. Training</option>
              <option value="sync">Finished: 2. Sync Data</option>
              <option value="comparative">Finished: 3. Comparative</option>
              <option value="deg">Finished: 4. DEG</option>
              <option value="pathway">Finished: 5. Pathway</option>
              <option value="pred_model">Finished: 6. Pred. Model</option>
            </select>
          </div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            Total: {projectLogs.length} runs
          </div>
        </div>

        <div>
          <SubPanel title="🏗️ Architecture" items={categories.architecture} filterKey="architecture" />
          <SubPanel title="📏 Sample Size" items={categories.size} filterKey="size" />
          <SubPanel title="⚙️ Sensitivity" items={categories.sensitivity} filterKey="sensitivity" />
          <SubPanel title="📦 Other Runs" items={categories.other} filterKey="architecture" />
        </div>
      </div>
    );
  };

  const renderTaskDashboard = () => {
    if (!selectedRunId) return null;
    const status = resultsStatus.run_statuses?.[selectedRunId];
    const isSizeTask = selectedRunId.includes("Size") && !selectedRunId.includes("Architecture");
    
    if (taskView === 'training') {
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <button className="chip" onClick={() => { setTaskView('overview'); setViewingLog(null); }} style={{ padding: '0.5rem' }}>
              <ArrowLeft size={18} />
            </button>
            <h2 style={{ margin: 0 }}>Training Performance: {selectedRunId}</h2>
          </div>
          {renderLogViewer()}
        </div>
      );
    }

    if (taskView === 'sync') {
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <button className="chip" onClick={() => setTaskView('overview')} style={{ padding: '0.5rem' }}>
              <ArrowLeft size={18} />
            </button>
            <h2 style={{ margin: 0 }}>Sync Data Details: {selectedRunId}</h2>
          </div>
          {renderSyncStatus()}
        </div>
      );
    }

    if (taskView === 'comparative') {
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <button className="chip" onClick={() => setTaskView('overview')} style={{ padding: '0.5rem' }}>
              <ArrowLeft size={18} />
            </button>
            <h2 style={{ margin: 0 }}>Comparative Analysis: {selectedRunId}</h2>
          </div>
          {renderComparativeAnalysis()}
        </div>
      );
    }

    if (taskView === 'deg' || taskView === 'pathway' || taskView === 'prediction') {
      return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <button className="chip" onClick={() => setTaskView('overview')} style={{ padding: '0.5rem' }}>
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
          {taskView === 'deg' && renderDegAnalysis()}
          {taskView === 'pathway' && renderPathwayAnalysis()}
          {taskView === 'prediction' && renderPredictionAnalysis()}
        </div>
      );
    }

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <button className="chip" onClick={() => setSelectedRunId(null)} style={{ padding: '0.5rem' }}>
            <ArrowLeft size={18} />
          </button>
          <h2 style={{ margin: 0 }}>Task: {selectedRunId}</h2>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem' }}>
          <section className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>1. Training</div>
            <StatusButton label={status?.training === 'running' ? 'Running' : (status?.training === 'completed' ? 'Completed' : 'Idle')} status={status?.training || 'idle'} />
            <button className="chip" style={{ fontSize: '0.75rem' }} onClick={() => { setTaskView('training'); fetchLogs(selectedRunId); }}>View Performance</button>
          </section>
          <section className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>2. Sync Data</div>
            <StatusButton label={status?.sync ? 'Generated' : 'Pending'} status={status?.sync || false} />
            {status?.sync ? (
              <button className="chip" style={{ fontSize: '0.75rem' }} onClick={() => fetchSyncStatus(selectedRunId)}>View Details</button>
            ) : (
              <button 
                className={`chip selected ${status?.training !== 'completed' ? 'disabled' : ''}`} 
                style={{ fontSize: '0.75rem', opacity: status?.training !== 'completed' ? 0.5 : 1, cursor: status?.training !== 'completed' ? 'not-allowed' : 'pointer' }} 
                onClick={() => status?.training === 'completed' && handleRunStep(2)}
              >
                Start Sync
              </button>
            )}
          </section>
          <section className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', textAlign: 'center', opacity: isSizeTask ? 1 : 0.5 }}>
            <div style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>3. Comparative</div>
            <StatusButton label={isSizeTask ? (status?.comparative ? 'Done' : 'Pending') : 'Unavailable'} status={isSizeTask ? (status?.comparative || false) : 'unavailable'} />
            {isSizeTask && (status?.comparative ? (
              <button className="chip" style={{ fontSize: '0.75rem' }} onClick={() => fetchComparativeMetrics(selectedRunId)}>View Results</button>
            ) : (
              <button 
                className={`chip selected ${!status?.sync ? 'disabled' : ''}`} 
                style={{ fontSize: '0.75rem', opacity: !status?.sync ? 0.5 : 1, cursor: !status?.sync ? 'not-allowed' : 'pointer' }} 
                onClick={() => status?.sync && handleRunStep(3)}
              >
                Start Analysis
              </button>
            ))}
          </section>
          <section className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem', textAlign: 'center', opacity: isSizeTask ? 1 : 0.5 }}>
            <div style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>4. Bio-marker Analysis</div>
            <StatusButton 
              label={isSizeTask ? (status?.deg && status?.pathway && status?.pred_model ? 'Done' : 'Pending') : 'Unavailable'} 
              status={isSizeTask ? (status?.deg && status?.pathway && status?.pred_model) : 'unavailable'} 
            />
            {isSizeTask && (status?.deg || status?.pathway || status?.pred_model ? (
              <button className="chip" style={{ fontSize: '0.75rem' }} onClick={() => fetchDegMetrics(selectedRunId)}>View Results</button>
            ) : (
              <button 
                className={`chip selected ${!status?.comparative ? 'disabled' : ''}`} 
                style={{ fontSize: '0.75rem', opacity: !status?.comparative ? 0.5 : 1, cursor: !status?.comparative ? 'not-allowed' : 'pointer' }} 
                onClick={() => status?.comparative && handleRunStep(4)}
              >
                Start Analysis
              </button>
            ))}
          </section>
        </div>
      </div>
    );
  };

  const currentProj = projects.find(p => p.id === selectedProject);
  const status = selectedRunId ? resultsStatus.run_statuses?.[selectedRunId] : null;
  const isSizeTask = selectedRunId ? selectedRunId.includes("Size") && !selectedRunId.includes("Architecture") : false;

  const StepItem = ({ num, label, active, status, onClick, disabled }: any) => (
    <div 
      onClick={!disabled ? onClick : undefined}
      className={`nav-item ${active ? 'active' : ''} ${disabled ? 'disabled' : ''}`}
      style={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: '12px', 
        paddingLeft: '24px', 
        fontSize: '0.85rem',
        opacity: disabled ? 0.5 : 1,
        cursor: disabled ? 'not-allowed' : 'pointer'
      }}
    >
      <div style={{ 
        width: '20px', 
        height: '20px', 
        borderRadius: '50%', 
        backgroundColor: active ? 'var(--primary-color)' : (status === 'completed' || status === true ? 'var(--success-color)' : '#e5e7eb'),
        color: active || status === 'completed' || status === true ? 'white' : 'var(--text-muted)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '0.7rem',
        fontWeight: 'bold'
      }}>
        {num}
      </div>
      <span style={{ flex: 1 }}>{label}</span>
      {status === 'running' && <Loader2 size={12} className="animate-spin" />}
    </div>
  );

  if (loading) return <div style={{ padding: '2rem' }}>Loading Dashboard...</div>;

  return (
    <div className="dashboard-container">
      <aside className="sidebar">
        <div className="sidebar-header"><Activity size={24} /><span>GANomics Dashboard</span></div>
        <nav className="nav-menu">
          <div style={{ padding: '0.5rem 1rem', fontSize: '0.65rem', fontWeight: 'bold', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Menu</div>
          <a className={`nav-item ${activeTab === 'train' && !selectedRunId ? 'active' : ''}`} onClick={() => { setActiveTab('train'); setSelectedRunId(null); setTaskView('overview'); }}>
            <LayoutDashboard size={18} /> Project
          </a>

          {selectedRunId && (
            <>
              <div style={{ padding: '1.5rem 1rem 0.5rem 1rem', fontSize: '0.65rem', fontWeight: 'bold', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Task Pipeline</div>
              <div style={{ padding: '0.25rem 1rem 0.75rem 1rem', fontSize: '0.75rem', color: 'var(--primary-color)', fontWeight: '600', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {selectedRunId}
              </div>
              <StepItem 
                num="1" label="Training" 
                active={taskView === 'training' || (taskView === 'overview' && !status?.training)} 
                status={status?.training}
                onClick={() => { setTaskView('training'); fetchLogs(selectedRunId); }}
              />
              <StepItem 
                num="2" label="Sync Data" 
                active={taskView === 'sync'}
                status={status?.sync}
                disabled={!status?.sync} 
                onClick={() => fetchSyncStatus(selectedRunId)}
              />
              <StepItem 
                num="3" label="Comparative" 
                active={taskView === 'comparative'}
                status={isSizeTask ? status?.comparative : 'unavailable'}
                disabled={!isSizeTask || !status?.comparative}
                onClick={() => fetchComparativeMetrics(selectedRunId)}
              />
              <StepItem 
                num="4" label="Bio-markers" 
                active={['deg', 'pathway', 'prediction'].includes(taskView)}
                status={isSizeTask ? (status?.deg && status?.pathway && status?.pred_model) : 'unavailable'} 
                disabled={!isSizeTask || (!status?.deg && !status?.pathway && !status?.pred_model)} 
                onClick={() => fetchDegMetrics(selectedRunId)}
              />
            </>
          )}

          <div style={{ marginTop: 'auto', padding: '1rem' }}>
            {previouslySelected.length > 0 && (
              <div style={{ marginBottom: '1rem' }}>
                <div style={{ padding: '0 0 0.5rem 0', fontSize: '0.65rem', fontWeight: 'bold', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: '4px' }}>
                   Quick Re-visit
                </div>
                {previouslySelected.map(runId => (
                  <div 
                    key={runId} 
                    className={`nav-item ${selectedRunId === runId ? 'active' : ''}`}
                    onClick={() => { setSelectedRunId(runId); setTaskView('overview'); }}
                    style={{ fontSize: '0.7rem', padding: '0.4rem 0.75rem', borderRadius: '6px', marginBottom: '2px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                    title={runId}
                  >
                    {runId}
                  </div>
                ))}
              </div>
            )}
            <div className="nav-item"><Settings size={18} /> Settings</div>
          </div>
        </nav>
      </aside>

      <main className="main-content">
        {selectedRunId ? (
          renderTaskDashboard()
        ) : (
          <>
            <header className="header">
              <div className="header-info">
                <h1>{activeTab === 'train' ? 'Model Training' : 'Performance Analysis'}</h1>
                <p>{currentProj ? `Project: ${currentProj.name}` : 'Select a project.'}</p>
              </div>
              {activeTab === 'train' && (
                <button className="chip selected" style={{ borderRadius: '8px', padding: '0.75rem 1.5rem', display: 'flex', alignItems: 'center' }} onClick={handleStartSession}>
                  <Play size={16} style={{ marginRight: '8px' }} /> Start Session
                </button>
              )}
            </header>

            {activeTab === 'train' ? (
              <>
                <section className="card">
                  <h3>Project Selection</h3>
                  <div className="chip-grid">
                    {projects.map(p => (<div key={p.id} className={`chip ${selectedProject === p.id ? 'selected' : ''}`} onClick={() => setSelectedProject(p.id)}>{p.name}</div>))}
                  </div>
                </section>
                <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '2rem' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                    <section className="card">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                        <h3 style={{ margin: 0 }}>Ablation Matrix</h3>
                        <div className="chip-grid" style={{ gap: '0.5rem' }}>
                          <button className={`chip ${ablationType === 'size' ? 'selected' : ''}`} style={{ fontSize: '0.7rem' }} onClick={() => setAblationType('size')}>Vary Size (N)</button>
                          <button className={`chip ${ablationType === 'beta' ? 'selected' : ''}`} style={{ fontSize: '0.7rem' }} onClick={() => setAblationType('beta')}>Vary Beta (β)</button>
                          <button className={`chip ${ablationType === 'lambda' ? 'selected' : ''}`} style={{ fontSize: '0.7rem' }} onClick={() => setAblationType('lambda')}>Vary Lambda (λ)</button>
                        </div>
                      </div>
                      
                      {ablationType === 'size' && (
                        <div style={{ marginBottom: '1.5rem' }}>
                          <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Experiment: Sample Size (N)</label>
                          <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>β and λ will be fixed at 10.0</p>
                          <div className="chip-grid" style={{ marginTop: '0.5rem' }}>{sizes.map(s => (<div key={s} className={`chip ${selectedSizes.includes(s) ? 'selected' : ''}`} onClick={() => toggleSelection(s, selectedSizes, setSelectedSizes)}>N={s}</div>))}</div>
                        </div>
                      )}

                      {ablationType === 'beta' && (
                        <div style={{ marginBottom: '1.5rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Experiment: Weights (β)</label>
                            <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>Bio-Feedback Alignment</span>
                          </div>
                          <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>N will be fixed at 50, λ at 10.0</p>
                          <div className="chip-grid" style={{ marginTop: '0.5rem' }}>{betas.map(b => (<div key={b} className={`chip ${selectedBetas.includes(b) ? 'selected' : ''}`} onClick={() => toggleSelection(b, selectedBetas, setSelectedBetas)}>β={b.toFixed(1)}</div>))}</div>
                        </div>
                      )}

                      {ablationType === 'lambda' && (
                        <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Experiment: Weights (λ)</label>
                            <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>Cycle Reconstruction</span>
                          </div>
                          <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>N will be fixed at 50, β at 10.0</p>
                          <div className="chip-grid" style={{ marginTop: '0.5rem' }}>{lambdas.map(l => (<div key={l} className={`chip ${selectedLambdas.includes(l) ? 'selected' : ''}`} onClick={() => toggleSelection(l, selectedLambdas, setSelectedLambdas)}>λ={l.toFixed(1)}</div>))}</div>
                        </div>
                      )}

                      <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <label style={{ fontSize: '0.8rem', fontWeight: '600' }}>Session Repeats:</label>
                        <input 
                          type="number" min={1} max={10} 
                          value={selectedRepeats} 
                          onChange={(e) => setSelectedRepeats(parseInt(e.target.value) || 1)}
                          style={{ width: '60px', padding: '0.25rem', borderRadius: '4px', border: '1px solid var(--border-color)' }}
                        />
                        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>* How many runs for each condition</span>
                      </div>
                    </section>
                    <section className="card">
                      <h3>Project Status</h3>
                      {renderProjectStatus()}
                    </section>
                  </div>
                  {renderAblationAnalyticsModal()}
                  <section className="card">
                    <h3>Quick Config</h3>
                    <div className="config-form" style={{ gridTemplateColumns: '1fr', gap: '1.25rem' }}>
                      <div className="form-group">
                        <label style={{ fontSize: '0.8rem', fontWeight: '600' }}>Training Epochs</label>
                        <div className="chip-grid" style={{ marginTop: '0.5rem', gap: '0.4rem' }}>
                          {[100, 500, 1000].map(e => (
                            <div key={e} className={`chip ${selectedEpochs === e ? 'selected' : ''}`} style={{ fontSize: '0.75rem', padding: '0.2rem 0.6rem' }} onClick={() => setSelectedEpochs(e)}>{e}</div>
                          ))}
                          <div className={`chip ${selectedEpochs === 'custom' ? 'selected' : ''}`} style={{ fontSize: '0.75rem', padding: '0.2rem 0.6rem' }} onClick={() => setSelectedEpochs('custom')}>Custom</div>
                        </div>
                        {selectedEpochs === 'custom' && (
                          <input 
                            type="number" value={customEpochs} 
                            onChange={(e) => setCustomEpochs(parseInt(e.target.value) || 250)}
                            style={{ marginTop: '0.5rem', width: '100%', padding: '0.4rem', borderRadius: '6px', border: '1px solid var(--border-color)', fontSize: '0.85rem' }}
                          />
                        )}
                      </div>

                      <div style={{ marginTop: '0.5rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)' }}>
                        <label style={{ fontSize: '0.75rem', fontWeight: 'bold', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.5rem', display: 'block' }}>Fixed Parameters</label>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.7rem' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}><span style={{ color: 'var(--text-muted)' }}>LR</span><span style={{ fontWeight: '600' }}>{currentProj?.config?.optimizer?.lr || "0.0002"}</span></div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}><span style={{ color: 'var(--text-muted)' }}>Batch Size</span><span style={{ fontWeight: '600' }}>{currentProj?.config?.train?.batch_size || 1}</span></div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}><span style={{ color: 'var(--text-muted)' }}>Input NC</span><span style={{ fontWeight: '600' }}>{currentProj?.config?.model?.input_nc}</span></div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}><span style={{ color: 'var(--text-muted)' }}>Output NC</span><span style={{ fontWeight: '600' }}>{currentProj?.config?.model?.output_nc}</span></div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}><span style={{ color: 'var(--text-muted)' }}>λ IDT</span><span style={{ fontWeight: '600' }}>{currentProj?.config?.model?.lambda_idt}</span></div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}><span style={{ color: 'var(--text-muted)' }}>GAN Mode</span><span style={{ fontWeight: '600' }}>{currentProj?.config?.model?.gan_mode}</span></div>
                        </div>
                      </div>
                    </div>
                  </section>                </div>
              </>
            ) : (
              <section className="card">
                <h3>Comparison: {selectedProject}</h3>
                {metrics.length > 0 ? (
                  <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '1rem' }}>
                    <thead><tr style={{ textAlign: 'left', borderBottom: '2px solid var(--border-color)' }}><th style={{ padding: '0.75rem' }}>Algorithm</th><th style={{ padding: '0.75rem' }}>Pearson</th><th style={{ padding: '0.75rem' }}>Spearman</th><th style={{ padding: '0.75rem' }}>MAE</th></tr></thead>
                    <tbody>{metrics.map((m, i) => (<tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}><td style={{ padding: '0.75rem', fontWeight: '600' }}>{m.Algorithm}</td><td style={{ padding: '0.75rem' }}>{m.Pearson}</td><td style={{ padding: '0.75rem' }}>{m.Spearman}</td><td style={{ padding: '0.75rem' }}>{m.MAE}</td></tr>))}</tbody>
                  </table>
                ) : (<div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>No metrics found.</div>)}
              </section>
            )}
          </>
        )}
      </main>
    </div>
  );
};

export default App;
