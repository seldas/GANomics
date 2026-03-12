import React, { useState, useMemo } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { Loader2, Filter, List } from 'lucide-react';

interface PathwayAnalysisProps {
  data: any | null;
}

export const PathwayAnalysis: React.FC<PathwayAnalysisProps> = ({ data }) => {
  const [selectedLibrary, setSelectedLibrary] = useState<string | null>(null);
  const [selectedAlgo, setSelectedAlgo] = useState<string>('GANomics');
  const [showAll, setShowAll] = useState(false);

  // 1. Determine base variables safely
  const libraries = useMemo(() => data ? Object.keys(data) : [], [data]);
  const currentLib = selectedLibrary || libraries[0] || '';
  const libData = data && currentLib ? data[currentLib] : null;
  const availableAlgos = useMemo(() => libData?.details ? Object.keys(libData.details) : [], [libData]);
  const currentDetails = useMemo(() => libData?.details?.[selectedAlgo] || [], [libData, selectedAlgo]);

  // 2. Calculate Significance Ratio Curve
  const chartData = useMemo(() => {
    if (!currentDetails.length) return [];
    
    const results: any[] = [];
    let significantInSyn = 0;
    const maxK = Math.min(currentDetails.length, 100);
    
    for (let k = 1; k <= maxK; k++) {
      const pathway = currentDetails[k-1];
      if (pathway.Syn_P < 0.05) {
        significantInSyn++;
      }
      
      if (k % 5 === 0 || k === maxK) {
        results.push({
          k,
          ratio: significantInSyn / k,
          label: `Top ${k}`
        });
      }
    }
    return results;
  }, [currentDetails]);

  // 3. Filter Table
  const filteredTableData = useMemo(() => {
    if (showAll) return currentDetails;
    return currentDetails.filter((p: any) => p.Real_P < 0.05);
  }, [currentDetails, showAll]);

  if (!data) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;
  if (libraries.length === 0) return <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>No pathway results available.</div>;
  if (!libData?.details) return <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>Loading pathway details...</div>;

  const colors = ['var(--primary-color)', '#16a34a', '#818cf8', '#ea580c', '#db2777', '#7c3aed'];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Figure Panel: Significance Ratio */}
      <div className="card" style={{ padding: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
          <div>
            <h3>Significance Preservation Ratio</h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              Fraction of Top X pathways (ranked by Real P-value) that are also significant (p &lt; 0.05) in Synthetic data.
            </p>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
            {libraries.map(lib => (
              <button key={lib} className={`chip ${currentLib === lib ? 'selected' : ''}`} style={{ fontSize: '0.7rem' }} onClick={() => setSelectedLibrary(lib)}>
                {lib.replace(/_/g, ' ')}
              </button>
            ))}
          </div>
        </div>

        <div style={{ height: '400px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="k" label={{ value: 'Top X Pathways (Real)', position: 'insideBottom', offset: -10 }} />
              <YAxis domain={[0, 1]} label={{ value: 'Sig. Ratio (Syn p < 0.05)', angle: -90, position: 'insideLeft' }} />
              <Tooltip formatter={(val: number) => [val.toFixed(2), 'Preservation Ratio']} />
              <Line type="monotone" dataKey="ratio" stroke="var(--primary-color)" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 8 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Table Panel: ORA Details */}
      {availableAlgos.length > 0 && (
        <section className="card" style={{ padding: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
            <div>
              <h3 style={{ margin: 0 }}>Functional Annotation (ORA)</h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: '4px 0 0 0' }}>
                Ranked by Real P-value (Most significant first).
              </p>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <button 
                className={`chip ${!showAll ? 'selected' : ''}`} 
                onClick={() => setShowAll(false)}
                style={{ fontSize: '0.7rem', display: 'flex', alignItems: 'center', gap: '4px' }}
              >
                <Filter size={12} /> Significant (Real p &lt; 0.05)
              </button>
              <button 
                className={`chip ${showAll ? 'selected' : ''}`} 
                onClick={() => setShowAll(true)}
                style={{ fontSize: '0.7rem', display: 'flex', alignItems: 'center', gap: '4px' }}
              >
                <List size={12} /> Show All
              </button>
              <div style={{ height: '24px', width: '1px', background: '#e2e8f0', margin: '0 0.5rem' }} />
              <div style={{ fontSize: '0.75rem', fontWeight: 'bold', color: 'var(--text-muted)' }}>ALGO:</div>
              <select className="chip" value={selectedAlgo} onChange={(e) => setSelectedAlgo(e.target.value)} style={{ border: 'none', fontWeight: '600' }}>
                {availableAlgos.map(a => <option key={a} value={a}>{a}</option>)}
              </select>
            </div>
          </div>

          <div style={{ overflowX: 'auto', maxHeight: '600px' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead style={{ position: 'sticky', top: 0, backgroundColor: '#fff', zIndex: 1 }}>
                <tr style={{ textAlign: 'left', borderBottom: '2px solid #eee' }}>
                  <th style={{ padding: '0.75rem 1rem' }}>Pathway Name</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>Genes</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>p-val (Real)</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>p-val (Syn)</th>
                </tr>
              </thead>
              <tbody>
                {filteredTableData.map((p: any, i: number) => (
                  <tr key={i} style={{ 
                    borderBottom: '1px solid #f3f4f6',
                    backgroundColor: p.Real_P &lt; 0.05 && p.Syn_P &lt; 0.05 ? '#f0fdf4' : 'transparent'
                  }}>
                    <td style={{ padding: '0.6rem 1rem', fontWeight: '500' }}>{p.set}</td>
                    <td style={{ padding: '0.6rem 1rem', textAlign: 'center', color: 'var(--text-muted)' }}>{p.Genes || '-'}</td>
                    <td style={{ padding: '0.6rem 1rem', textAlign: 'right', fontWeight: p.Real_P &lt; 0.05 ? '600' : 'normal' }}>{p.Real_P?.toExponential(2)}</td>
                    <td style={{ padding: '0.6rem 1rem', textAlign: 'right', color: p.Syn_P &lt; 0.05 ? '#16a34a' : 'inherit' }}>
                      {p.Syn_P ? p.Syn_P.toExponential(2) : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {filteredTableData.length === 0 && (
              <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                No significant pathways found (Real p &lt; 0.05).
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
};
