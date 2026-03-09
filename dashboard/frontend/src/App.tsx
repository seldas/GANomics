import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  LayoutDashboard, 
  Settings, 
  Play, 
  Activity, 
  Database,
  Terminal
} from 'lucide-react';
import './App.css';

interface Project {
  id: string;
  name: string;
  genes: number;
  samples: number;
  config_path: string;
}

const API_BASE = "http://localhost:8000/api";

const App: React.FC = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<string>('');
  const [selectedSizes, setSelectedSizes] = useState<number[]>([50]);
  const [selectedBetas, setSelectedBetas] = useState<number[]>([10.0]);
  const [resultsStatus, setResultsStatus] = useState<{checkpoints: string[], logs: string[]}>({checkpoints: [], logs: []});
  const [loading, setLoading] = useState(true);

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
      const res = await axios.post(`${API_BASE}/train`, {
        config: proj.config_path,
        sizes: selectedSizes,
        betas: selectedBetas,
        lambdas: [10.0],
        repeats: 1
      });
      alert(res.data.message);
    } catch (err) {
      alert("Failed to start training session");
    }
  };

  if (loading) return <div style={{ padding: '2rem' }}>Loading GANomics Dashboard...</div>;

  const currentProj = projects.find(p => p.id === selectedProject);

  return (
    <div className="dashboard-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <Activity size={24} />
          <span>GANomics Dashboard</span>
        </div>
        <nav className="nav-menu">
          <a className="nav-item active"><LayoutDashboard size={20} /> Overview</a>
          <a className="nav-item"><Database size={20} /> Datasets</a>
          <a className="nav-item"><Settings size={20} /> Configuration</a>
          <a className="nav-item"><Terminal size={20} /> Logs</a>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="header">
          <div className="header-info">
            <h1>Training & Ablation</h1>
            <p>Configure and monitor your GANomics experiments.</p>
          </div>
          <button 
            className="chip selected" 
            style={{ borderRadius: '8px', padding: '0.75rem 1.5rem', display: 'flex', alignItems: 'center' }}
            onClick={handleStartSession}
          >
            <Play size={16} style={{ marginRight: '8px' }} />
            Start Session
          </button>
        </header>

        {/* Project & Data Selection */}
        <section className="card">
          <h3>Project Selection</h3>
          <div className="chip-grid">
            {projects.map(p => (
              <div 
                key={p.id} 
                className={`chip ${selectedProject === p.id ? 'selected' : ''}`}
                onClick={() => setSelectedProject(p.id)}
              >
                {p.name}
              </div>
            ))}
            <div className="chip" style={{ borderStyle: 'dashed' }}>+ Upload New Dataset</div>
          </div>
          {currentProj && (
            <div style={{ marginTop: '1rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Selected: <strong>{currentProj.id}</strong> | Detected {currentProj.genes} genes and {currentProj.samples} samples.
            </div>
          )}
        </section>

        <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '2rem' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <section className="card">
              <h3>Sample Sizes (Ablation)</h3>
              <div className="chip-grid">
                {sizes.map(s => (
                  <div 
                    key={s} 
                    className={`chip ${selectedSizes.includes(s) ? 'selected' : ''}`}
                    onClick={() => toggleSelection(s, selectedSizes, setSelectedSizes)}
                  >
                    N={s}
                  </div>
                ))}
              </div>
            </section>

            <section className="card">
              <h3>Biological Feedback (Beta)</h3>
              <div className="chip-grid">
                {betas.map(b => (
                  <div 
                    key={b} 
                    className={`chip ${selectedBetas.includes(b) ? 'selected' : ''}`}
                    onClick={() => toggleSelection(b, selectedBetas, setSelectedBetas)}
                  >
                    β={b.toFixed(1)}
                  </div>
                ))}
              </div>
            </section>
          </div>

          {/* Queue Monitor - Simplified for now to show results found */}
          <section className="card">
            <h3>Completed Runs (Checkpoints)</h3>
            <div className="queue-list">
              {resultsStatus.checkpoints.length > 0 ? (
                resultsStatus.checkpoints.map(cp => (
                  <div key={cp} className="queue-item">
                    <div style={{ fontWeight: '600', fontSize: '0.85rem' }}>{cp}</div>
                    <span className="status-badge status-success">Ready</span>
                  </div>
                ))
              ) : (
                <div style={{ color: 'var(--text-muted)' }}>No completed runs found.</div>
              )}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};

export default App;
