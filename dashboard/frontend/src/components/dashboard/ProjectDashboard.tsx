import React, { useState } from 'react';
import { 
  LayoutDashboard, Plus, Search, ChevronRight, ChevronDown, 
  Clock, Timer, Square, RotateCcw, Loader2 
} from 'lucide-react';
import type { Project, ResultsStatus } from '../../types';
import { StatusButton } from '../common/UIComponents';

interface ProjectDashboardProps {
  projects: Project[];
  selectedProject: string;
  onSelectProject: (id: string) => void;
  resultsStatus: ResultsStatus;
  onSelectRun: (id: string) => void;
  onFetchAblationLogs: (category: string) => void;
  onStopTask: (id: string) => void;
  onRestartTask: (id: string) => void;
  onFetchLogs: (id: string) => void;
}

export const ProjectDashboard: React.FC<ProjectDashboardProps> = ({
  projects, selectedProject, onSelectProject, resultsStatus, onSelectRun, 
  onFetchAblationLogs, onStopTask, onRestartTask, onFetchLogs
}) => {
  const [collapsedPanels, setCollapsedPanels] = useState<Record<string, boolean>>({});
  const [statusFilters, setStatusFilters] = useState<Record<string, string>>({
    architecture: 'all', size: 'all', sensitivity: 'all', other: 'all'
  });
  const [globalStepFilter, setGlobalStepFilter] = useState<string>('all');

  const togglePanel = (key: string) => {
    setCollapsedPanels(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const projectLogs = resultsStatus.logs
    .filter(l => l.startsWith(selectedProject))
    .map(l => l.replace("_log.txt", ""));

  const categories = {
    architecture: projectLogs.filter(id => id.includes("Architecture")),
    size: projectLogs.filter(id => id.includes("Size") && !id.includes("Architecture")),
    sensitivity: projectLogs.filter(id => id.includes("Sensitivity")),
    other: projectLogs.filter(id => !id.includes("Architecture") && !id.includes("Size") && !id.includes("Sensitivity"))
  };

  const renderOngoingTasks = () => {
    const runningTasks = Object.entries(resultsStatus.run_statuses || {})
      .filter(([_, s]) => s.training === 'running' || s.stopped)
      .map(([id, s]) => ({ id, ...s }));

    if (runningTasks.length === 0) return null;

    return (
      <section className="card" style={{ borderLeft: '4px solid var(--primary-color)', padding: '1.5rem', marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
          <Clock size={20} className={runningTasks.some(t => t.training === 'running') ? "animate-spin-slow" : ""} style={{ color: 'var(--primary-color)' }} />
          <h3 style={{ margin: 0, fontSize: '1rem' }}>Task Monitor ({runningTasks.length})</h3>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {runningTasks.map(task => {
            const progress = (task.total_epochs || 500) > 0 ? ((task.current_epoch || 0) / (task.total_epochs || 500)) * 100 : 0;
            const isRunning = task.training === 'running';
            
            return (
              <div key={task.id} className="card" style={{ margin: 0, padding: '1rem', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', opacity: isRunning ? 1 : 0.8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <span style={{ fontWeight: 'bold', fontSize: '0.9rem' }}>{task.id}</span>
                    <span className={`status-badge ${isRunning ? 'status-running' : 'status-disabled'}`} style={{ fontSize: '0.65rem', padding: '2px 8px' }}>
                      {isRunning ? 'RUNNING' : 'STOPPED'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px', marginRight: '0.5rem' }}>
                      <Timer size={14} /> Epoch {task.current_epoch} / {task.total_epochs}
                    </div>
                    
                    <button className="chip" style={{ fontSize: '0.7rem' }} onClick={() => onFetchLogs(task.id)}>Logs</button>

                    {isRunning ? (
                      <button className="chip" style={{ fontSize: '0.7rem', color: '#ff4d4f', border: '1px solid #ff4d4f' }} onClick={() => onStopTask(task.id)}>Stop</button>
                    ) : (
                      <button className="chip" style={{ fontSize: '0.7rem', color: 'var(--primary-color)', border: '1px solid var(--primary-color)' }} onClick={() => onRestartTask(task.id)}>Restart</button>
                    )}
                  </div>
                </div>
                <div style={{ width: '100%', height: '8px', backgroundColor: '#e2e8f0', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ width: `${progress}%`, height: '100%', backgroundColor: isRunning ? 'var(--primary-color)' : '#94a3b8', transition: 'width 0.5s ease-out' }} />
                </div>
              </div>
            );
          })}
        </div>
      </section>
    );
  };

  const SubPanel = ({ title, items, filterKey }: { title: string, items: string[], filterKey: string }) => {
    if (items.length === 0) return null;
    const isCollapsed = collapsedPanels[filterKey];

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

      const key = filterKey;
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
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }} onClick={() => togglePanel(filterKey)}>
              {isCollapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
              <h4 style={{ margin: 0, fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{title} ({filteredItems.length})</h4>
            </div>
            <button 
              className="chip" 
              style={{ fontSize: '0.6rem', padding: '0.1rem 0.4rem', border: '1px solid var(--primary-color)', color: 'var(--primary-color)' }}
              onClick={() => onFetchAblationLogs(filterKey)}
            >
              Ablation Analytics
            </button>
          </div>
          {!isCollapsed && (
            <select 
              value={statusFilters[filterKey]} 
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
                <div key={runId} className="queue-item" onClick={() => onSelectRun(runId)} style={{ cursor: 'pointer', flexDirection: 'column', alignItems: 'flex-start', gap: '8px' }}>
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <header className="header" style={{ marginBottom: '2rem' }}>
        <div className="header-info">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <LayoutDashboard size={24} style={{ color: 'var(--primary-color)' }} />
            <h1 style={{ margin: 0 }}>Project Dashboard</h1>
          </div>
          <p style={{ color: 'var(--text-muted)' }}>Monitoring {projects.length} active projects.</p>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <div className="card" style={{ padding: '0.75rem 1.5rem', margin: 0, textAlign: 'center' }}>
            <span style={{ fontSize: '0.7rem', fontWeight: 'bold' }}>TOTAL RUNS</span><br />
            <span style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{resultsStatus.logs.length}</span>
          </div>
          <div className="card" style={{ padding: '0.75rem 1.5rem', margin: 0, textAlign: 'center', borderLeft: '4px solid #1890ff' }}>
            <span style={{ fontSize: '0.7rem', fontWeight: 'bold' }}>ACTIVE</span><br />
            <span style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{Object.values(resultsStatus.run_statuses || {}).filter(s => s.training === 'running').length}</span>
          </div>
        </div>
      </header>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
        {renderOngoingTasks()}

        <section className="card">
          <h3>Project Selection</h3>
          <div className="chip-grid">
            {projects.map(p => (
              <div key={p.id} className={`chip ${selectedProject === p.id ? 'selected' : ''}`} onClick={() => onSelectProject(p.id)}>{p.name}</div>
            ))}
          </div>
        </section>

        <section className="card" style={{ padding: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
            <h3>Project Status & Experimental Analysis</h3>
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
          </div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <SubPanel title="🏗️ Architecture" items={categories.architecture} filterKey="architecture" />
            <SubPanel title="📏 Sample Size" items={categories.size} filterKey="size" />
            <SubPanel title="⚙️ Sensitivity" items={categories.sensitivity} filterKey="sensitivity" />
            <SubPanel title="📦 Other Runs" items={categories.other} filterKey="other" />
          </div>
        </section>
      </div>
    </div>
  );
};

