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
  ChevronsLeft,
  ChevronsRight
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
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

const API_BASE = "http://localhost:8000/api";

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
  const [selectedSizes, setSelectedSizes] = useState<number[]>([50]);
  const [selectedBetas, setSelectedBetas] = useState<number[]>([10.0]);
  const [selectedLambdas, setSelectedLambdas] = useState<number[]>([10.0]);
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

  const StatusButton = ({ label, status }: { label: string, status: 'running' | 'completed' | 'idle' | boolean }) => {
    let className = "status-badge ";
    let style: React.CSSProperties = {
      fontSize: '0.65rem',
      padding: '2px 6px',
      borderRadius: '4px',
      whiteSpace: 'nowrap',
    };

    if (status === 'running') {
      className += "status-running";
      style = { ...style, background: '#1890ff', color: 'white', animation: 'pulse 1.5s infinite' };
    } else if (status === 'completed' || status === true) {
      className += "status-success";
      style = { ...style, background: '#52c41a', color: 'white' };
    } else {
      className += "status-error";
      style = { ...style, border: '1px solid #ff4d4f', color: '#ff4d4f', opacity: 0.6 };
    }

    return (
      <div className={className} style={style}>
        {label}
      </div>
    );
  };

  const renderProjectStatus = () => {
    const projectLogs = resultsStatus.logs
      .filter(l => l.startsWith(selectedProject))
      .map(l => l.replace("_log.txt", ""));
    
    return (
      <div className="queue-list">
        {projectLogs.map(runId => {
          const status = resultsStatus.run_statuses?.[runId];
          return (
            <div key={runId} className="queue-item" onClick={() => fetchLogs(runId)} style={{ cursor: 'pointer', flexDirection: 'column', alignItems: 'flex-start', gap: '8px' }}>
              <div style={{ fontWeight: '600', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                {runId}
                {status?.training === 'running' && <Loader2 size={12} className="animate-spin" style={{ color: '#1890ff' }} />}
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                <StatusButton label="Training" status={status?.training || 'idle'} />
                <StatusButton label="Sync Data" status={status?.sync || false} />
                <StatusButton label="Comparative" status={status?.comparative || false} />
                <StatusButton label="DEG" status={status?.deg || false} />
                <StatusButton label="Pathway" status={status?.pathway || false} />
                <StatusButton label="Pred. Model" status={status?.pred_model || false} />
              </div>
            </div>
          );
        })}
      </div>
    );
  };
  const [metrics, setMetrics] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Log Viewer State
  const [viewingLog, setViewingLog] = useState<string | null>(null);
  const [logData, setLogData] = useState<LogResponse | null>(null);
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
    try {
      await axios.post(`${API_BASE}/train`, {
        config_path: project.config_path,
        sizes: selectedSizes,
        betas: selectedBetas,
        lambdas: selectedLambdas,
        repeats: 1
      });
      alert("Training session started!");
    } catch (err) {
      console.error(err);
      alert("Failed to start session");
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [projRes, resStatus] = await Promise.all([
          axios.get(`${API_BASE}/projects`),
          axios.get(`${API_BASE}/results`)
        ]);
        setProjects(projRes.data);
        if (projRes.data.length > 0 && !selectedProject) setSelectedProject(projRes.data[0].id);
        setResultsStatus(resStatus.data);
      } catch (err) {
        console.error("Failed to fetch dashboard data", err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
    const interval = setInterval(fetchData, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, [selectedProject]);

  useEffect(() => {
    if (activeTab === 'analysis' && selectedProject) {
      axios.get(`${API_BASE}/metrics/${selectedProject}`)
        .then(res => setMetrics(res.data))
        .catch(() => setMetrics([]));
    }
  }, [activeTab, selectedProject]);

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

  if (loading) return <div style={{ padding: '2rem' }}>Loading Dashboard...</div>;
  const currentProj = projects.find(p => p.id === selectedProject);

  return (
    <div className="dashboard-container">
      <aside className="sidebar">
        <div className="sidebar-header"><Activity size={24} /><span>GANomics Dashboard</span></div>
        <nav className="nav-menu">
          <a className={`nav-item ${activeTab === 'train' ? 'active' : ''}`} onClick={() => setActiveTab('train')}><LayoutDashboard size={20} /> Training</a>
          <a className={`nav-item ${activeTab === 'analysis' ? 'active' : ''}`} onClick={() => setActiveTab('analysis')}><BarChart3 size={20} /> Analysis</a>
          <a className="nav-item"><Database size={20} /> Datasets</a>
          <a className="nav-item"><Settings size={20} /> Configuration</a>
        </nav>
      </aside>

      <main className="main-content">
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
                  <h3>Ablation Matrix</h3>
                  <div style={{ marginBottom: '1rem' }}>
                    <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Sizes (N)</label>
                    <div className="chip-grid" style={{ marginTop: '0.5rem' }}>{sizes.map(s => (<div key={s} className={`chip ${selectedSizes.includes(s) ? 'selected' : ''}`} onClick={() => toggleSelection(s, selectedSizes, setSelectedSizes)}>N={s}</div>))}</div>
                  </div>
                  <div style={{ marginBottom: '1.5rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                      <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Weights (β)</label>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>Bio-Feedback Alignment</span>
                    </div>
                    <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.2rem', lineHeight: '1.2' }}>
                      Controls the strength of the biological feedback loss, ensuring genes maintain their relative expression patterns across platforms.
                    </p>
                    <div className="chip-grid" style={{ marginTop: '0.5rem' }}>{betas.map(b => (<div key={b} className={`chip ${selectedBetas.includes(b) ? 'selected' : ''}`} onClick={() => toggleSelection(b, selectedBetas, setSelectedBetas)}>β={b.toFixed(1)}</div>))}</div>
                  </div>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                      <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Weights (λ)</label>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>Cycle Reconstruction</span>
                    </div>
                    <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.2rem', lineHeight: '1.2' }}>
                      Controls the cycle-consistency weight, enforcing that translating back to the original platform reproduces the input.
                    </p>
                    <div className="chip-grid" style={{ marginTop: '0.5rem' }}>{lambdas.map(l => (<div key={l} className={`chip ${selectedLambdas.includes(l) ? 'selected' : ''}`} onClick={() => toggleSelection(l, selectedLambdas, setSelectedLambdas)}>λ={l.toFixed(1)}</div>))}</div>
                  </div>
                </section>
                <section className="card">
                  <h3>Project Status</h3>
                  {renderProjectStatus()}
                </section>
              </div>
              <section className="card">
                <h3>Quick Config</h3>
                <div className="config-form" style={{ gridTemplateColumns: '1fr', gap: '1rem' }}>
                  <div className="form-group"><label>Epochs</label><input type="number" defaultValue={currentProj?.config?.train?.n_epochs || 250} /></div>
                  <div className="form-group"><label>Learning Rate</label><input type="text" defaultValue={currentProj?.config?.optimizer?.lr || "0.0002"} /></div>
                  <div className="form-group"><label>Batch Size</label><input type="number" defaultValue={currentProj?.config?.train?.batch_size || 1} /></div>
                  
                  {currentProj?.config && (
                    <div style={{ marginTop: '0.5rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)' }}>
                      <label style={{ fontSize: '0.75rem', fontWeight: 'bold', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.5rem', display: 'block' }}>Read-only Parameters</label>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.7rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}><span style={{ color: 'var(--text-muted)' }}>Input NC</span><span style={{ fontWeight: '600' }}>{currentProj.config.model?.input_nc}</span></div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}><span style={{ color: 'var(--text-muted)' }}>Output NC</span><span style={{ fontWeight: '600' }}>{currentProj.config.model?.output_nc}</span></div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}><span style={{ color: 'var(--text-muted)' }}>λ IDT</span><span style={{ fontWeight: '600' }}>{currentProj.config.model?.lambda_idt}</span></div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}><span style={{ color: 'var(--text-muted)' }}>GAN Mode</span><span style={{ fontWeight: '600' }}>{currentProj.config.model?.gan_mode}</span></div>
                      </div>
                    </div>
                  )}
                </div>
              </section>
            </div>
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
      </main>

      {viewingLog && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}>
          <div className="card" style={{ width: '95%', height: '90%', display: 'flex', flexDirection: 'column', gap: '0.75rem', overflow: 'hidden' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                <h3>Logs: {viewingLog.replace("_log.txt", "")}</h3>
                <div className="chip-grid">
                  <div className={`chip ${logMode === 'chart' ? 'selected' : ''}`} style={{ padding: '0.2rem 0.6rem', fontSize: '0.8rem' }} onClick={() => setLogMode('chart')}><LineChartIcon size={12} /> Figure</div>
                  <div className={`chip ${logMode === 'structured' ? 'selected' : ''}`} style={{ padding: '0.2rem 0.6rem', fontSize: '0.8rem' }} onClick={() => setLogMode('structured')}><TableIcon size={12} /> Table</div>
                </div>
              </div>
              <X onClick={() => setViewingLog(null)} style={{ cursor: 'pointer' }} />
            </div>

            {logData?.summary && (
              <div style={{ display: 'flex', gap: '0.5rem', padding: '0.25rem 0', flexWrap: 'wrap' }}>
                <div style={{ background: '#f8f9fa', padding: '0.5rem 1rem', borderRadius: '6px', border: '1px solid var(--border-color)', minWidth: '120px' }}><div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', fontWeight: 'bold' }}>PROGRESS</div><div style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>E{logData.summary.epoch} | I{logData.summary.iters}</div></div>
                {LOSS_METRICS.map(m => (
                  <div key={m.key} style={{ background: visibleMetrics.includes(m.key) ? '#f8f9fa' : '#fff', opacity: visibleMetrics.includes(m.key) ? 1 : 0.5, padding: '0.5rem 1rem', borderRadius: '6px', borderLeft: `4px solid ${m.color}`, borderRight: '1px solid var(--border-color)', borderTop: '1px solid var(--border-color)', borderBottom: '1px solid var(--border-color)', minWidth: '100px', flex: 1, cursor: 'pointer' }} onClick={() => toggleSelection(m.key, visibleMetrics, setVisibleMetrics)}>
                    <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', fontWeight: 'bold', display: 'flex', justifyContent: 'space-between' }}>{m.label.toUpperCase()}{visibleMetrics.includes(m.key) ? <Eye size={10} /> : <EyeOff size={10} />}</div>
                    <div style={{ fontSize: '0.9rem', fontWeight: 'bold', color: m.color }}>{logData.summary[m.key]?.toFixed(4) || "0.0000"}</div>
                  </div>
                ))}
              </div>
            )}
            
            <div style={{ flex: 1, overflow: 'auto', borderRadius: '8px', border: '1px solid var(--border-color)', backgroundColor: '#fff', display: 'flex', flexDirection: 'column' }}>
              {!logData ? (
                <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>
              ) : logMode === 'chart' ? (
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
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Page {currentPage} of {totalPages || 1} ({filteredData.length} matches)</div>
                  </div>
                  <div style={{ flex: 1, overflowY: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                      <thead style={{ position: 'sticky', top: 0, backgroundColor: '#fff', zIndex: 1 }}><tr style={{ textAlign: 'left', borderBottom: '2px solid var(--border-color)' }}><th style={{ padding: '0.4rem' }}>Epoch</th><th style={{ padding: '0.4rem' }}>Iters</th><th style={{ padding: '0.4rem' }}>G_A</th><th style={{ padding: '0.4rem' }}>G_B</th><th style={{ padding: '0.4rem' }}>D_A</th><th style={{ padding: '0.4rem' }}>D_B</th><th style={{ padding: '0.4rem' }}>Cycle</th><th style={{ padding: '0.4rem' }}>Feedback</th><th style={{ padding: '0.4rem' }}>IDT</th></tr></thead>
                      <tbody>{tableData.map((row, i) => (<tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}><td style={{ padding: '0.4rem', fontWeight: '600' }}>{row.epoch}</td><td style={{ padding: '0.4rem' }}>{row.iters}</td><td style={{ padding: '0.4rem' }}>{row.G_A?.toFixed(3)}</td><td style={{ padding: '0.4rem' }}>{row.G_B?.toFixed(3)}</td><td style={{ padding: '0.4rem' }}>{row.D_A?.toFixed(3)}</td><td style={{ padding: '0.4rem' }}>{row.D_B?.toFixed(3)}</td><td style={{ padding: '0.4rem' }}>{((row.cycle_A || 0) + (row.cycle_B || 0)).toFixed(3)}</td><td style={{ padding: '0.4rem' }}>{((row.feedback_A || 0) + (row.feedback_B || 0)).toFixed(3)}</td><td style={{ padding: '0.4rem' }}>{((row.idt_A || 0) + (row.idt_B || 0)).toFixed(3)}</td></tr>))}</tbody>
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
            {logData && <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Displaying all {logData.structured.length} recorded training points.</div>}
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
