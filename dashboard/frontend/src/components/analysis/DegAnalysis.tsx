import React from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { Loader2 } from 'lucide-react';

interface DegAnalysisProps {
  data: any | null;
}

export const DegAnalysis: React.FC<DegAnalysisProps> = ({ data }) => {
  if (!data) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;
  
  const algos = Object.keys(data);
  if (algos.length === 0) return <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>No DEG results available.</div>;

  // Flatten data for chart
  const thresholds = Array.from(new Set(Object.values(data).flatMap((pts: any) => pts.map((p: any) => p.threshold))))
    .sort((a: any, b: any) => a - b);

  const chartData = thresholds.map(t => {
    const entry: any = { threshold: t, label: `p < ${t}` };
    algos.forEach(a => {
      const pt = data[a].find((p: any) => p.threshold === t);
      if (pt) entry[a] = pt.jaccard;
    });
    return entry;
  });

  const colors = ['var(--primary-color)', '#16a34a', '#818cf8', '#ea580c', '#db2777', '#7c3aed'];

  return (
    <div className="card" style={{ padding: '2rem' }}>
      <div style={{ marginBottom: '2rem' }}>
        <h3>DEG Bio-marker Preservation</h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Jaccard similarity between real and synthetic Differentially Expressed Genes across significance thresholds.
        </p>
      </div>
      
      <div style={{ height: '450px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="label" fontSize={11} />
            <YAxis domain={[0, 1]} label={{ value: 'Jaccard Similarity Index', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend verticalAlign="top" />
            {algos.map((algo, i) => (
              <Line 
                key={algo} 
                type="monotone" 
                dataKey={algo} 
                stroke={colors[i % colors.length]} 
                strokeWidth={algo === 'GANomics' ? 3 : 1.5}
                dot={{ r: algo === 'GANomics' ? 5 : 3 }}
                activeDot={{ r: 8 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
