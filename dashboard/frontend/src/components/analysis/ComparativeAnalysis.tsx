import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Loader2 } from 'lucide-react';

interface ComparativeAnalysisProps {
  data: any[] | null;
}

export const ComparativeAnalysis: React.FC<ComparativeAnalysisProps> = ({ data }) => {
  const [corrGroup, setCorrGroup] = useState<'MA' | 'RS'>('MA');

  if (!data) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;
  if (data.length === 0) return <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>No comparative data found.</div>;

  const processed = data.map(d => ({
    ...d,
    pearsonNum: parseFloat(d.Pearson),
    spearmanNum: parseFloat(d.Spearman)
  }));

  const chartData = processed
    .filter(d => d.Algorithm.includes(`(${corrGroup})`))
    .sort((a, b) => b.pearsonNum - a.pearsonNum);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <section className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
          <h3>Platform Alignment Correlation ({corrGroup})</h3>
          <div className="chip-grid">
            <button className={`chip ${corrGroup === 'MA' ? 'selected' : ''}`} onClick={() => setCorrGroup('MA')}>Microarray (MA)</button>
            <button className={`chip ${corrGroup === 'RS' ? 'selected' : ''}`} onClick={() => setCorrGroup('RS')}>RNA-Seq (RS)</button>
          </div>
        </div>
        
        <div style={{ height: '450px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ bottom: 80 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="Algorithm" angle={-45} textAnchor="end" interval={0} fontSize={11} />
              <YAxis domain={[0, 1]} label={{ value: 'Correlation Coefficient (r)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend verticalAlign="top" />
              <Bar dataKey="pearsonNum" name="Pearson" fill="var(--primary-color)" radius={[4, 4, 0, 0]} />
              <Bar dataKey="spearmanNum" name="Spearman" fill="#818cf8" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <section className="card">
        <h3>Detailed Metrics</h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ textAlign: 'left', borderBottom: '2px solid var(--border-color)' }}>
                <th style={{ padding: '1rem' }}>Algorithm</th>
                <th style={{ padding: '1rem', textAlign: 'right' }}>Pearson</th>
                <th style={{ padding: '1rem', textAlign: 'right' }}>Spearman</th>
                <th style={{ padding: '1rem', textAlign: 'right' }}>MAE (L1)</th>
              </tr>
            </thead>
            <tbody>
              {processed.map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f3f4f6', backgroundColor: row.Algorithm.includes('GANomics') ? '#f0f9ff' : 'transparent' }}>
                  <td style={{ padding: '0.75rem 1rem', fontWeight: row.Algorithm.includes('GANomics') ? '700' : 'normal' }}>{row.Algorithm}</td>
                  <td style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>{row.Pearson}</td>
                  <td style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>{row.Spearman}</td>
                  <td style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>{row.L1}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
