import React, { useState } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { Loader2 } from 'lucide-react';

interface PathwayAnalysisProps {
  data: any | null;
}

export const PathwayAnalysis: React.FC<PathwayAnalysisProps> = ({ data }) => {
  const [selectedLibrary, setSelectedLibrary] = useState<string | null>(null);

  if (!data) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;
  
  const libraries = Object.keys(data);
  if (libraries.length === 0) return <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>No pathway results available.</div>;

  const currentLib = selectedLibrary || libraries[0];
  const libData = data[currentLib];
  if (!libData?.concordance) return null;

  const chartData = Object.keys(libData.concordance).map(algo => ({
    name: algo,
    rho: libData.concordance[algo].Spearman_Rho,
    p: libData.concordance[algo].P_Value
  })).sort((a, b) => b.rho - a.rho);

  return (
    <div className="card" style={{ padding: '2rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <div>
          <h3>Pathway Concordance (ρ)</h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Spearman correlation of enrichment scores between real and synthetic data.</p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', justifyContent: 'flex-end', maxWidth: '50%' }}>
          {libraries.map(lib => (
            <button 
              key={lib} 
              className={`chip ${currentLib === lib ? 'selected' : ''}`} 
              style={{ fontSize: '0.7rem' }}
              onClick={() => setSelectedLibrary(lib)}
            >
              {lib.replace(/_/g, ' ')}
            </button>
          ))}
        </div>
      </div>

      <div style={{ height: '400px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" angle={-45} textAnchor="end" interval={0} fontSize={11} />
            <YAxis domain={[0, 1]} label={{ value: 'Spearman Rho (ρ)', angle: -90, position: 'insideLeft' }} />
            <Tooltip 
              formatter={(value: number) => [value.toFixed(4), 'Concordance (ρ)']}
            />
            <Bar dataKey="rho" fill="var(--primary-color)" barSize={40} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
