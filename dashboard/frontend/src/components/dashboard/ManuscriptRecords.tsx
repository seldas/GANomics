import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  FileText, Loader2, Calendar, Database, CheckCircle2, XCircle, Clock
} from 'lucide-react';
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
          Displaying tasks archived for the manuscript. Status icons represent availability of results in each pipeline step.
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
    </div>
  );
};
