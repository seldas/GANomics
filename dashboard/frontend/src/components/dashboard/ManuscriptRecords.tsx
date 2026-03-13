import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { 
  FileText, Loader2, Calendar, CheckCircle2, Clock, LineChart as ChartIcon, X, LayoutGrid, Globe
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { API_BASE } from '../../constants';

interface ManuscriptTask {
  run_id: string;
  project: string;
  major_group: number;
  size: number;
  repeats: number;
  status: {
    sync: boolean;
    comparative: boolean;
    deg: boolean;
    pathway: boolean;
    prediction: boolean;
  };
  mtime: number;
}

const LossModal = ({ runId, onClose }: { runId: string, onClose: () => void }) => {
  const [data, setData] = useState<any[] | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get(`${API_BASE}/manuscript/logs/${runId}`)
      .then(res => setData(res.data.structured))
      .catch(err => console.error(err))
      .finally(() => setLoading(false));
  }, [runId]);

  const displayedData = useMemo(() => {
    if (!data || data.length === 0) return [];
    
    // 1. Apply Moving Average (Window size = 20)
    const windowSize = 20;
    const smoothed = data.map((point, index) => {
      const start = Math.max(0, index - Math.floor(windowSize / 2));
      const end = Math.min(data.length, start + windowSize);
      const subset = data.slice(start, end);
      
      const avg = (key: string) => subset.reduce((sum, p) => sum + (p[key] || 0), 0) / subset.length;
      
      return {
        ...point,
        G_loss: avg('G_loss'),
        D_loss: avg('D_loss'),
        cycle_loss: avg('cycle_loss')
      };
    });

    // 2. Downsample to max 50 points for display performance
    if (smoothed.length <= 50) return smoothed;
    const step = Math.ceil(smoothed.length / 50);
    return smoothed.filter((_, i) => i % step === 0).slice(0, 50);
  }, [data]);

  return (
    <div className="modal-overlay" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1100 }}>
      <div className="modal-content" style={{ width: '80%', maxWidth: '900px', height: '600px', display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h3 style={{ margin: 0 }}>Validation Losses: {runId}</h3>
          <button onClick={onClose} className="chip" style={{ padding: '4px' }}><X size={20} /></button>
        </div>
        
        <div style={{ marginBottom: '1rem', padding: '0.5rem 1rem', backgroundColor: '#f0f9ff', borderRadius: '6px', fontSize: '0.85rem', color: '#0369a1' }}>
          <b>Note:</b> These curves represent the <b>Validation Loss</b> measured during the training process (Smoothed MA-20).
        </div>
        
        <div style={{ flex: 1, minHeight: 0 }}>
          {loading ? (
            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Loader2 className="animate-spin" /></div>
          ) : displayedData && displayedData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={displayedData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                <YAxis 
                  domain={['auto', 'auto']}
                  label={{ value: 'Validation Loss', angle: -90, position: 'insideLeft' }} 
                />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="G_loss" stroke="#3b82f6" dot={false} strokeWidth={2} name="Generator Loss" />
                <Line type="monotone" dataKey="D_loss" stroke="#ef4444" dot={false} strokeWidth={2} name="Discriminator Loss" />
                <Line type="monotone" dataKey="cycle_loss" stroke="#16a34a" dot={false} strokeWidth={1} name="Cycle Loss" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>No loss data available in logs.</div>
          )}
        </div>
      </div>
    </div>
  );
};

const TaskTable = ({ tasks, title, icon: Icon }: { tasks: ManuscriptTask[], title: string, icon: any }) => {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleDateString();
  };

  const StatusIcon = ({ exists }: { exists: boolean }) => (
    exists 
      ? <CheckCircle2 size={16} color="#16a34a" title="Completed" /> 
      : <Clock size={16} color="#94a3b8" title="Not Found" />
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '0 0.5rem' }}>
        <Icon size={18} color="var(--primary-color)" />
        <h3 style={{ margin: 0, fontSize: '1.1rem' }}>{title}</h3>
        <span className="chip" style={{ fontSize: '0.7rem' }}>{tasks.length} tasks</span>
      </div>
      
      <div className="card" style={{ padding: '0' }}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ textAlign: 'left', backgroundColor: '#f8fafc', borderBottom: '2px solid var(--border-color)' }}>
                <th style={{ padding: '1rem 1.5rem' }}>Run ID</th>
                <th style={{ padding: '1rem' }}>Project</th>
                <th style={{ padding: '1rem', textAlign: 'center' }}>Size</th>
                <th style={{ padding: '1rem', textAlign: 'center' }}>Rep.</th>
                <th style={{ padding: '1rem', textAlign: 'center' }}>Losses</th>
                <th style={{ padding: '1rem' }}>Sync</th>
                <th style={{ padding: '1rem' }}>Comp.</th>
                <th style={{ padding: '1rem' }}>DEG</th>
                <th style={{ padding: '1rem' }}>Path.</th>
                <th style={{ padding: '1rem' }}>Pred.</th>
                <th style={{ padding: '1rem 1.5rem', textAlign: 'right' }}>Archived</th>
              </tr>
            </thead>
            <tbody>
              {tasks.map((task) => (
                <tr key={task.run_id} className="hover-row" style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '0.75rem 1.5rem', fontWeight: 'bold', fontFamily: 'monospace' }}>
                    {task.run_id}
                  </td>
                  <td style={{ padding: '0.75rem 1rem' }}>
                    <span className="chip" style={{ fontSize: '0.7rem', backgroundColor: task.project === 'CycleGAN' ? '#fef3c7' : 'var(--bg-muted)', color: task.project === 'CycleGAN' ? '#92400e' : 'inherit' }}>
                      {task.project}
                    </span>
                  </td>
                  <td style={{ padding: '0.75rem 1rem', textAlign: 'center', fontWeight: '600' }}>{task.size}</td>
                  <td style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>{task.repeats}</td>
                  <td style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>
                    <button 
                      className="chip selected" 
                      style={{ padding: '4px 8px' }}
                      onClick={() => setSelectedRunId(task.run_id)}
                    >
                      <ChartIcon size={14} />
                    </button>
                  </td>
                  <td style={{ padding: '0.75rem 1rem' }}><StatusIcon exists={task.status.sync} /></td>
                  <td style={{ padding: '0.75rem 1rem' }}><StatusIcon exists={task.status.comparative} /></td>
                  <td style={{ padding: '0.75rem 1rem' }}><StatusIcon exists={task.status.deg} /></td>
                  <td style={{ padding: '0.75rem 1rem' }}><StatusIcon exists={task.status.pathway} /></td>
                  <td style={{ padding: '0.75rem 1rem' }}><StatusIcon exists={task.status.prediction} /></td>
                  <td style={{ padding: '0.75rem 1.5rem', textAlign: 'right', color: 'var(--text-muted)' }}>
                    {formatDate(task.mtime)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      {selectedRunId && (
        <LossModal 
          runId={selectedRunId} 
          onClose={() => setSelectedRunId(null)} 
        />
      )}
    </div>
  );
};

export const ManuscriptRecords: React.FC = () => {
  const [tasks, setTasks] = useState<ManuscriptTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTasks = async () => {
      try {
        const res = await axios.get(`${API_BASE}/manuscript/tasks`);
        setTasks(res.data);
      } catch (err) {
        console.error('Failed to fetch manuscript tasks:', err);
        setError('Failed to load manuscript records.');
      } finally {
        setLoading(false);
      }
    };
    fetchTasks();
  }, []);

  const { nbGroup, otherGroup } = useMemo(() => {
    return {
      nbGroup: tasks.filter(t => t.major_group === 0),
      otherGroup: tasks.filter(t => t.major_group === 1)
    };
  }, [tasks]);

  if (loading) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;

  if (error) return (
    <div className="card" style={{ padding: '2rem', textAlign: 'center', color: '#ef4444' }}>
      {error}
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <div className="card" style={{ padding: '1.5rem 2rem' }}>
        <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '12px' }}>
          <FileText size={24} color="var(--primary-color)" />
          Manuscript Records
        </h2>
        <p style={{ margin: '0.5rem 0 0 0', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          Displaying archived tasks for the manuscript.
        </p>
      </div>

      <TaskTable 
        title="Neuroblastoma (NB & CycleGAN)" 
        tasks={nbGroup} 
        icon={LayoutGrid}
      />

      <TaskTable 
        title="Generalization (Other Projects)" 
        tasks={otherGroup} 
        icon={Globe}
      />

      <div style={{ padding: '0 0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
        * Note: For manuscript records, <b>G_loss</b>, <b>D_loss</b>, and <b>cycle_loss</b> are calculated as the average of their respective A and B components (e.g., (G_A + G_B)/2) for consistency with the project dashboard visualization.
      </div>
    </div>
  );
};
