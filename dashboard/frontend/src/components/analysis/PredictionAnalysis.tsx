import React from 'react';
import { Loader2 } from 'lucide-react';

interface PredictionAnalysisProps {
  data: any | null;
}

export const PredictionAnalysis: React.FC<PredictionAnalysisProps> = ({ data }) => {
  if (!data) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;
  
  const algos = Object.keys(data);
  if (algos.length === 0) return <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>No prediction results available.</div>;

  return (
    <section className="card">
      <div style={{ marginBottom: '1.5rem' }}>
        <h3>Classifier Performance Analysis</h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Random Forest models evaluated across multiple cross-platform scenarios (Real vs Synthetic).
        </p>
      </div>
      
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ textAlign: 'left', borderBottom: '2px solid var(--border-color)' }}>
              <th style={{ padding: '1rem' }}>Algorithm</th>
              <th style={{ padding: '1rem' }}>Scenario</th>
              <th style={{ padding: '1rem', textAlign: 'right' }}>MCC</th>
              <th style={{ padding: '1rem', textAlign: 'right' }}>Accuracy</th>
              <th style={{ padding: '1rem', textAlign: 'right' }}>Recall</th>
              <th style={{ padding: '1rem', textAlign: 'right' }}>Precision</th>
              <th style={{ padding: '1rem', textAlign: 'right' }}>F1 Score</th>
            </tr>
          </thead>
          <tbody>
            {algos.map(algo => (
              data[algo].map((d: any, i: number) => (
                <tr key={`${algo}-${i}`} style={{ borderBottom: '1px solid #f3f4f6', backgroundColor: algo === 'GANomics' ? '#f0fdf4' : 'transparent' }}>
                  <td style={{ padding: '0.75rem 1rem', fontWeight: 'bold' }}>{i === 0 ? algo : ''}</td>
                  <td style={{ padding: '0.75rem 1rem' }}>{d.Scenario}</td>
                  <td style={{ padding: '0.75rem 1rem', textAlign: 'right', fontWeight: '600' }}>{d.MCC.toFixed(4)}</td>
                  <td style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>{d.Accuracy.toFixed(4)}</td>
                  <td style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>{d.Recall.toFixed(4)}</td>
                  <td style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>{d.Precision.toFixed(4)}</td>
                  <td style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>{d.F1.toFixed(4)}</td>
                </tr>
              ))
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
};
