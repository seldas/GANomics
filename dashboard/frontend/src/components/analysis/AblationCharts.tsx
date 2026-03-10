import React from 'react';
import { 
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';

interface AblationChartsProps {
  ablationData: any[];
  sensitivityType: 'beta' | 'lambda';
  onSetSensitivityType: (type: 'beta' | 'lambda') => void;
}

export const AblationCharts: React.FC<AblationChartsProps> = ({ 
  ablationData, sensitivityType, onSetSensitivityType 
}) => {
  if (!ablationData || ablationData.length === 0) {
    return (
      <div style={{ padding: '4rem', textAlign: 'center', backgroundColor: '#fff', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
        <Activity size={48} style={{ color: '#cbd5e1', marginBottom: '1rem' }} />
        <h3 style={{ color: '#64748b' }}>No Ablation Data Available</h3>
        <p style={{ color: '#94a3b8' }}>Complete at least one experimental run to see comparative metrics here.</p>
      </div>
    );
  }

  // 1. Sample Size Data
  const sizeData = ablationData
    .filter(d => d.run_id && d.run_id.includes("Size") && !d.run_id.includes("Architecture"))
    .map(d => {
      const match = d.run_id.match(/Size_(\d+)/);
      return {
        size: match ? parseInt(match[1]) : 0,
        Pearson: parseFloat(d.Pearson || 0),
        Spearman: parseFloat(d.Spearman || 0),
        MAE: parseFloat(d.L1 || 0)
      };
    })
    .filter(d => d.size > 0)
    .sort((a, b) => a.size - b.size);

  // 2. Sensitivity Data (Beta/Lambda)
  const sensitivityData = ablationData
    .filter(d => d.run_id && d.run_id.includes("Sensitivity"))
    .map(d => {
      const bMatch = d.run_id.match(/Beta_([\d.]+)/);
      const lMatch = d.run_id.match(/Lambda_([\d.]+)/);
      return {
        beta: bMatch ? parseFloat(bMatch[1]) : 10.0,
        lambda: lMatch ? parseFloat(lMatch[1]) : 10.0,
        MAE: parseFloat(d.L1 || 0)
      };
    })
    .sort((a, b) => sensitivityType === 'beta' ? a.beta - b.beta : a.lambda - b.lambda);

  const hasSizeData = sizeData.length > 0;
  const hasSensitivityData = sensitivityData.length > 0;

  if (!hasSizeData && !hasSensitivityData) {
    return (
      <div style={{ padding: '4rem', textAlign: 'center', backgroundColor: '#fff', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
        <Info size={48} style={{ color: '#cbd5e1', marginBottom: '1rem' }} />
        <h3 style={{ color: '#64748b' }}>Ablation Data Found, but No Chartable Metrics</h3>
        <p style={{ color: '#94a3b8' }}>Found {ablationData.length} results, but they don't match standard Size/Sensitivity patterns.</p>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <section className="card">
        <h3>Scaling Performance (Sample Size N)</h3>
        <div style={{ height: '350px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sizeData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="size" label={{ value: 'Training Samples (N)', position: 'insideBottom', offset: -5 }} />
              <YAxis domain={[0.5, 1]} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="Pearson" stroke="var(--primary-color)" strokeWidth={3} />
              <Line type="monotone" dataKey="Spearman" stroke="#818cf8" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <h3>Hyperparameter Sensitivity</h3>
          <div className="chip-grid">
            <button className={`chip ${sensitivityType === 'beta' ? 'selected' : ''}`} onClick={() => onSetSensitivityType('beta')}>Beta (β)</button>
            <button className={`chip ${sensitivityType === 'lambda' ? 'selected' : ''}`} onClick={() => onSetSensitivityType('lambda')}>Lambda (λ)</button>
          </div>
        </div>
        <div style={{ height: '350px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={sensitivityData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey={sensitivityType} />
              <YAxis label={{ value: 'MAE (L1 Loss)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Bar dataKey="MAE" fill="var(--primary-color)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  );
};
