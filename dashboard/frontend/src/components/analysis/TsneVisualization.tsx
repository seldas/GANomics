import React from 'react';
import { 
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ZAxis 
} from 'recharts';
import { Loader2 } from 'lucide-react';

interface TsneVisualizationProps {
  data: any[] | null;
}

export const TsneVisualization: React.FC<TsneVisualizationProps> = ({ data }) => {
  if (!data) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;
  if (data.length === 0) return <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>No t-SNE visualization data available.</div>;

  // Group data by label
  const groups: Record<string, any[]> = {};
  data.forEach(item => {
    if (!groups[item.label]) groups[item.label] = [];
    groups[item.label].push(item);
  });

  const colors: Record<string, string> = {
    'MA Real': '#1f77b4',
    'MA Fake': '#d62728',
    'RS Real': '#ff7f0e',
    'RS Fake': '#2ca02c'
  };

  return (
    <div className="card" style={{ padding: '2rem' }}>
      <div style={{ marginBottom: '2rem' }}>
        <h3>t-SNE Visualization</h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Clustering analysis showing the proximity of real and synthetic profiles across both platforms.
        </p>
      </div>
      
      <div style={{ height: '500px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" dataKey="x" name="t-SNE 1" label={{ value: 't-SNE Component 1', position: 'insideBottom', offset: -10 }} />
            <YAxis type="number" dataKey="y" name="t-SNE 2" label={{ value: 't-SNE Component 2', angle: -90, position: 'insideLeft' }} />
            <ZAxis type="number" range={[60, 60]} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Legend verticalAlign="top" height={36}/>
            {Object.entries(groups).map(([label, points]) => (
              <Scatter 
                key={label} 
                name={label} 
                data={points} 
                fill={colors[label] || '#8884d8'} 
                fillOpacity={0.7}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
