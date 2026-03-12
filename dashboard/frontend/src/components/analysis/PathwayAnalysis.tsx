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
  const [selectedAlgo, setSelectedAlgo] = useState<string>('GANomics');

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

  const availableAlgos = Object.keys(libData.details || {});
  const currentDetails = libData.details?.[selectedAlgo] || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
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
                formatter={(value: any) => [typeof value === 'number' ? value.toFixed(4) : 'N/A', 'Concordance (ρ)']}
              />
              <Bar dataKey="rho" fill="var(--primary-color)" barSize={40} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {availableAlgos.length > 0 && (
        <section className="card" style={{ padding: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
            <div>
              <h3 style={{ margin: 0 }}>Top Ranked Pathways Comparison</h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '4px 0 0 0' }}>
                Sorted by Real Dataset ranking (Top to Bottom).
              </p>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 'bold', color: 'var(--text-muted)' }}>ALGORITHM:</div>
              <select 
                className="chip" 
                value={selectedAlgo} 
                onChange={(e) => setSelectedAlgo(e.target.value)}
                style={{ border: 'none', fontWeight: '600' }}
              >
                {availableAlgos.map(a => <option key={a} value={a}>{a}</option>)}
              </select>
            </div>
          </div>

          <div style={{ overflowX: 'auto', maxHeight: '500px' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead style={{ position: 'sticky', top: 0, backgroundColor: '#fff', zIndex: 1 }}>
                <tr style={{ textAlign: 'left', borderBottom: '2px solid #eee' }}>
                  <th style={{ padding: '0.75rem 1rem' }}>Pathway Name</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>Genes</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>Real Rank</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>Syn Rank</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>p-val (Real)</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>p-val (Syn)</th>
                </tr>
              </thead>
              <tbody>
                {currentDetails.map((p: any, i: number) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f3f4f6' }}>
                    <td style={{ padding: '0.6rem 1rem', fontWeight: '500' }}>{p.set}</td>
                    <td style={{ padding: '0.6rem 1rem', textAlign: 'center', color: 'var(--text-muted)' }}>{p.Real_Count || '-'}</td>
                    <td style={{ padding: '0.6rem 1rem', textAlign: 'center' }}>{Math.round(p.Real_Rank)}</td>
                    <td style={{ padding: '0.6rem 1rem', textAlign: 'center', color: Math.abs(p.Real_Rank - p.Syn_Rank) < 10 ? 'var(--success-color)' : 'inherit' }}>
                      {Math.round(p.Syn_Rank)}
                    </td>
                    <td style={{ padding: '0.6rem 1rem', textAlign: 'right' }}>
                      {p.Real_P !== undefined && p.Real_P !== null ? p.Real_P.toExponential(2) : '-'}
                    </td>
                    <td style={{ padding: '0.6rem 1rem', textAlign: 'right' }}>
                      {p.Syn_P !== undefined && p.Syn_P !== null ? p.Syn_P.toExponential(2) : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
};
