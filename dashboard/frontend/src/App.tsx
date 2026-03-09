import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  LayoutDashboard, 
  Settings, 
  Play, 
  Activity, 
  Database,
  X,
  FileText,
  Table as TableIcon,
  BarChart3,
  LineChart as LineChartIcon,
  CheckCircle2,
  Loader2
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
}

interface LogResponse {
  run_id: string;
  summary: any;
  structured: any[];
  raw: string[];
  total_lines: number;
}

const API_BASE = "http://localhost:8000/api";

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'train' | 'analysis'>('train');
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<string>('');
  const [selectedSizes, setSelectedSizes] = useState<number[]>([50]);
  const [selectedBetas, setSelectedBetas] = useState<number[]>([10.0]);
  const [resultsStatus, setResultsStatus] = useState<{checkpoints: string[], logs: string[]}>({checkpoints: [], logs: []});
  const [metrics, setMetrics] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Log Viewer State
  const [viewingLog, setViewingLog] = useState<string | null>(null);
  const [logData, setLogData] = useState<LogResponse | null>(null);
  const [logMode, setLogMode] = useState<'structured' | 'raw' | 'chart'>('structured');

  const sizes = [10, 20, 50, 100, 200];
  const betas = [0.0, 1.0, 5.0, 10.0, 50.0];

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [projRes, resStatus] = await Promise.all([
          axios.get(`${API_BASE}/projects`),
          axios.get(`${API_BASE}/results`)
        ]);
        setProjects(projRes.data);
        if (projRes.data.length > 0) setSelectedProject(projRes.data[0].id);
        setResultsStatus(resStatus.data);
      } catch (err) {
        console.error("Failed to fetch dashboard data", err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

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

  const handleStartSession = async () => {
    const proj = projects.find(p => p.id === selectedProject);
    if (!proj) return;

    try {
      await axios.post(`${API_BASE}/train`, {
        config_path: proj.config_path,
        sizes: selectedSizes,
        betas: selectedBetas,
        lambdas: [10.0],
        repeats: 1
      });
      alert("Training session started in background");
    } catch (err) {
      alert("Failed to start training session");
    }
  };

  const fetchLogs = (runId: string) => {
    setViewingLog(runId);
    setLogData(null);
    axios.get(`${API_BASE}/runs/${runId}/logs`)
      .then(res => setLogData(res.data))
      .catch((err) => {
        setLogData({ 
            run_id: runId, 
            summary: {}, 
            structured: [], 
            raw: [err.response?.data?.detail || "Could not find logs for this run."], 
            total_lines: 0 
        });
      });
  };

  if (loading) return <div style={{ padding: '2rem' }}>Loading GANomics Dashboard...</div>;

  const currentProj = projects.find(p => p.id === selectedProject);

  // Transform structured logs for chart
  const chartData = logData?.structured.map((d, i) => ({
    name: `E${d.epoch}`,
    index: i,
    G_Loss: ((d.G_A || 0) + (d.G_B || 0)) / 2,
    D_Loss: ((d.D_A || 0) + (d.D_B || 0)) / 2,
    Cycle: (d.cycle_A || 0) + (d.cycle_B || 0),
    Feedback: (d.feedback_A || 0) + (d.feedback_B || 0),
    IDT: (d.idt_A || 0) + (d.idt_B || 0),
  })) || [];

  return (
    <div className="dashboard-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <Activity size={24} />
          <span>GANomics Dashboard</span>
        </div>
        <nav className="nav-menu">
          <a className={`nav-item ${activeTab === 'train' ? 'active' : ''}`} onClick={() => setActiveTab('train')}><LayoutDashboard size={20} /> Training</a>
          <a className={`nav-item ${activeTab === 'analysis' ? 'active' : ''}`} onClick={() => setActiveTab('analysis')}><BarChart3 size={20} /> Analysis</a>
          <a className="nav-item"><Database size={20} /> Datasets</a>
          <a className="nav-item"><Settings size={20} /> Configuration</a>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="header">
          <div className="header-info">
            <h1>{activeTab === 'train' ? 'Model Training' : 'Performance Analysis'}</h1>
            <p>{currentProj ? `Project: ${currentProj.name}` : 'Select a project to begin.'}</p>
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
                {projects.map(p => (
                  <div key={p.id} className={`chip ${selectedProject === p.id ? 'selected' : ''}`} onClick={() => setSelectedProject(p.id)}>{p.name}</div>
                ))}
              </div>
            </section>

            <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '2rem' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                <section className="card">
                  <h3>Ablation Matrix</h3>
                  <div style={{ marginBottom: '1rem' }}>
                    <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Sample Sizes (N)</label>
                    <div className="chip-grid" style={{ marginTop: '0.5rem' }}>
                      {sizes.map(s => (<div key={s} className={`chip ${selectedSizes.includes(s) ? 'selected' : ''}`} onClick={() => toggleSelection(s, selectedSizes, setSelectedSizes)}>N={s}</div>))}
                    </div>
                  </div>
                  <div>
                    <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Feedback Weights (β)</label>
                    <div className="chip-grid" style={{ marginTop: '0.5rem' }}>
                      {betas.map(b => (<div key={b} className={`chip ${selectedBetas.includes(b) ? 'selected' : ''}`} onClick={() => toggleSelection(b, selectedBetas, setSelectedBetas)}>β={b.toFixed(1)}</div>))}
                    </div>
                  </div>
                </section>
                <section className="card">
                  <h3>Project Status</h3>
                  <div className="queue-list">
                    {resultsStatus.checkpoints.filter(c => c.startsWith(selectedProject)).map(cp => (
                      <div key={cp} className="queue-item" onClick={() => fetchLogs(cp)} style={{ cursor: 'pointer' }}>
                        <div style={{ fontWeight: '600', fontSize: '0.85rem' }}>{cp}</div>
                        <span className="status-badge status-success">Completed</span>
                      </div>
                    ))}
                    {resultsStatus.logs.filter(l => l.startsWith(selectedProject) && !resultsStatus.checkpoints.includes(l.replace("_log.txt", ""))).map(log => (
                      <div key={log} className="queue-item" onClick={() => fetchLogs(log)} style={{ cursor: 'pointer' }}>
                        <div style={{ fontWeight: '600', fontSize: '0.85rem' }}>{log.replace("_log.txt", "")}</div>
                        <span className="status-badge status-running">In Progress</span>
                      </div>
                    ))}
                  </div>
                </section>
              </div>
              <section className="card">
                <h3>Quick Config</h3>
                <div className="config-form" style={{ gridTemplateColumns: '1fr' }}>
                  <div className="form-group"><label>Epochs</label><input type="number" defaultValue={250} /></div>
                  <div className="form-group"><label>Learning Rate</label><input type="text" defaultValue="0.0002" /></div>
                  <div className="form-group"><label>Batch Size</label><input type="number" defaultValue={1} /></div>
                </div>
              </section>
            </div>
          </>
        ) : (
          <section className="card">
            <h3>Method Comparison: {selectedProject}</h3>
            {metrics.length > 0 ? (
              <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '1rem' }}>
                <thead>
                  <tr style={{ textAlign: 'left', borderBottom: '2px solid var(--border-color)' }}>
                    <th style={{ padding: '0.75rem' }}>Algorithm</th>
                    <th style={{ padding: '0.75rem' }}>Pearson</th>
                    <th style={{ padding: '0.75rem' }}>Spearman</th>
                    <th style={{ padding: '0.75rem' }}>MAE</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={{ padding: '0.75rem', fontWeight: '600' }}>{m.Algorithm}</td>
                      <td style={{ padding: '0.75rem' }}>{m.Pearson}</td>
                      <td style={{ padding: '0.75rem' }}>{m.Spearman}</td>
                      <td style={{ padding: '0.75rem' }}>{m.MAE}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (<div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>No metrics found. Run comparisons to generate Table 2.</div>)}
          </section>
        )}
      </main>

      {/* Log Viewer Modal */}
      {viewingLog && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}>
          <div className="card" style={{ width: '90%', height: '85%', display: 'flex', flexDirection: 'column', gap: '1rem', overflow: 'hidden' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                <h3>Logs: {viewingLog.replace("_log.txt", "")}</h3>
                <div className="chip-grid">
                  <div className={`chip ${logMode === 'chart' ? 'selected' : ''}`} style={{ padding: '0.25rem 0.75rem' }} onClick={() => setLogMode('chart')}>
                    <LineChartIcon size={14} style={{ marginRight: '6px' }} /> Loss Trends
                  </div>
                  <div className={`chip ${logMode === 'structured' ? 'selected' : ''}`} style={{ padding: '0.25rem 0.75rem' }} onClick={() => setLogMode('structured')}>
                    <TableIcon size={14} style={{ marginRight: '6px' }} /> Metrics Table
                  </div>
                  <div className={`chip ${logMode === 'raw' ? 'selected' : ''}`} style={{ padding: '0.25rem 0.75rem' }} onClick={() => setLogMode('raw')}>
                    <FileText size={14} style={{ marginRight: '6px' }} /> Raw Output
                  </div>
                </div>
              </div>
              <X onClick={() => setViewingLog(null)} style={{ cursor: 'pointer' }} />
            </div>

            {logData?.summary && (
              <div style={{ display: 'flex', gap: '1rem', padding: '0.5rem 0' }}>
                <div style={{ background: '#f8f9fa', padding: '0.75rem 1.25rem', borderRadius: '8px', border: '1px solid var(--border-color)', flex: 1 }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 'bold' }}>CURRENT PROGRESS</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>Epoch {logData.summary.epoch} | Iter {logData.summary.iters}</div>
                </div>
                <div style={{ background: '#f8f9fa', padding: '0.75rem 1.25rem', borderRadius: '8px', border: '1px solid var(--border-color)', flex: 1 }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 'bold' }}>GEN LOSS (AVG)</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 'bold', color: 'var(--primary-color)' }}>{logData.summary.loss_g?.toFixed(4)}</div>
                </div>
                <div style={{ background: '#f8f9fa', padding: '0.75rem 1.25rem', borderRadius: '8px', border: '1px solid var(--border-color)', flex: 1 }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 'bold' }}>DISC LOSS (AVG)</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 'bold', color: '#dc3545' }}>{logData.summary.loss_d?.toFixed(4)}</div>
                </div>
              </div>
            )}
            
            <div style={{ flex: 1, overflow: 'auto', borderRadius: '8px', border: '1px solid var(--border-color)', backgroundColor: logMode === 'raw' ? '#1e1e1e' : '#fff' }}>
              {!logData ? (
                <div style={{ padding: '4rem', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
                  <Loader2 className="animate-spin" size={32} />
                  <span>Parsing structured log data...</span>
                </div>
              ) : logMode === 'raw' ? (
                <div style={{ color: '#d4d4d4', padding: '1rem', minHeight: '100%', fontFamily: 'monospace', fontSize: '0.85rem' }}>
                  {logData.raw.map((line, i) => <div key={i}>{line}</div>)}
                </div>
              ) : logMode === 'chart' ? (
                <div style={{ width: '100%', height: '100%', padding: '1rem' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="name" interval="preserveStartEnd" minTickGap={50} />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="G_Loss" stroke="#007bff" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="D_Loss" stroke="#dc3545" dot={false} strokeWidth={2} />
                      <Line type="monotone" dataKey="Cycle" stroke="#ffc107" dot={false} strokeWidth={1} />
                      <Line type="monotone" dataKey="Feedback" stroke="#28a745" dot={false} strokeWidth={1} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                  <thead style={{ position: 'sticky', top: 0, backgroundColor: '#fff', zIndex: 1 }}>
                    <tr style={{ textAlign: 'left', borderBottom: '2px solid var(--border-color)' }}>
                      <th style={{ padding: '0.5rem' }}>Epoch</th>
                      <th style={{ padding: '0.5rem' }}>Iters</th>
                      <th style={{ padding: '0.5rem' }}>G_A</th>
                      <th style={{ padding: '0.5rem' }}>G_B</th>
                      <th style={{ padding: '0.5rem' }}>D_A</th>
                      <th style={{ padding: '0.5rem' }}>D_B</th>
                      <th style={{ padding: '0.5rem' }}>Cycle</th>
                      <th style={{ padding: '0.5rem' }}>Feedback</th>
                      <th style={{ padding: '0.5rem' }}>IDT</th>
                    </tr>
                  </thead>
                  <tbody>
                    {logData.structured.map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                        <td style={{ padding: '0.5rem', fontWeight: '600' }}>{row.epoch}</td>
                        <td style={{ padding: '0.5rem' }}>{row.iters}</td>
                        <td style={{ padding: '0.5rem' }}>{row.G_A?.toFixed(3)}</td>
                        <td style={{ padding: '0.5rem' }}>{row.G_B?.toFixed(3)}</td>
                        <td style={{ padding: '0.5rem' }}>{row.D_A?.toFixed(3)}</td>
                        <td style={{ padding: '0.5rem' }}>{row.D_B?.toFixed(3)}</td>
                        <td style={{ padding: '0.5rem' }}>{((row.cycle_A || 0) + (row.cycle_B || 0)).toFixed(3)}</td>
                        <td style={{ padding: '0.5rem' }}>{((row.feedback_A || 0) + (row.feedback_B || 0)).toFixed(3)}</td>
                        <td style={{ padding: '0.5rem' }}>{((row.idt_A || 0) + (row.idt_B || 0)).toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
            {logData && <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Showing last {logData.structured.length + logData.raw.length} lines of {logData.total_lines} total.</div>}
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
