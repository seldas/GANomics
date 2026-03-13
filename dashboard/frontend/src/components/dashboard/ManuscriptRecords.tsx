import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import { 
  FileText, Loader2, Calendar, CheckCircle2, Clock, LineChart as ChartIcon, X
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { API_BASE } from '../../constants';

interface ManuscriptTask {
  run_id: string;
  project: string;
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
    if (!data) return [];
    if (data.length <= 50) return data;
    const step = Math.ceil(data.length / 50);
    // Downsample to max 50 points
    return data.filter((_, i) => i % step === 0).slice(0, 50);
  }, [data]);

  return (
    <div className="modal-overlay" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1100 }}>
      <div className="modal-content" style={{ width: '80%', maxWidth: '900px', height: '600px', display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h3 style={{ margin: 0 }}>Training Losses: {runId}</h3>
          <button onClick={onClose} className="chip" style={{ padding: '4px' }}><X size={20} /></button>
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
                  scale="log" 
                  domain={['auto', 'auto']}
                  label={{ value: 'Loss (log)', angle: -90, position: 'insideLeft' }} 
                />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="G_loss" stroke="#3b82f6" dot={{ r: 3 }} strokeWidth={2} name="Generator Loss" />
                <Line type="monotone" dataKey="D_loss" stroke="#ef4444" dot={{ r: 3 }} strokeWidth={2} name="Discriminator Loss" />
                <Line type="monotone" dataKey="cycle_loss" stroke="#16a34a" dot={{ r: 3 }} strokeWidth={1} name="Cycle Loss" />
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

export const ManuscriptRecords: React.FC = () => {
  const [tasks, setTasks] = useState<ManuscriptTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

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

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleDateString();
  };

  const StatusIcon = ({ exists }: { exists: boolean }) => (
    exists 
      ? <CheckCircle2 size={16} color="#16a34a" title="Completed" /> 
      : <Clock size={16} color="#94a3b8" title="Not Found" />
  );

  if (loading) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;

  if (error) return (
    <div className="card" style={{ padding: '2rem', textAlign: 'center', color: '#ef4444' }}>
      {error}
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div className="card" style={{ padding: '1.5rem 2rem' }}>
        <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '12px' }}>
          <FileText size={24} color="var(--primary-color)" />
          Manuscript Records
        </h2>
        <p style={{ margin: '0.5rem 0 0 0', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          Displaying tasks archived for the manuscript. Click the chart icon to view training progress.
        </p>
      </div>

      <div className="card" style={{ padding: '0' }}>
        {tasks.length === 0 ? (
          <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
            No manuscript tasks found.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ textAlign: 'left', backgroundColor: '#f8fafc', borderBottom: '2px solid var(--border-color)' }}>
                  <th style={{ padding: '1rem 1.5rem' }}>Run ID</th>
                  <th style={{ padding: '1rem' }}>Project</th>
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
                      <span className="chip" style={{ fontSize: '0.7rem' }}>{task.project}</span>
                    </td>
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
        )}
      </div>

      <div style={{ padding: '0 0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
        * Note: For manuscript records, <b>G_loss</b>, <b>D_loss</b>, and <b>cycle_loss</b> are calculated as the average of their respective A and B components (e.g., (G_A + G_B)/2) for consistency with the project dashboard visualization.
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
