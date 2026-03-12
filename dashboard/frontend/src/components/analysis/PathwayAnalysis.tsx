import React, { useState, useMemo } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, ReferenceLine, ComposedChart, Area
} from 'recharts';
import { Loader2, Filter, List, TrendingUp, BarChart2, Info } from 'lucide-react';

interface PathwayAnalysisProps {
  data: any | null;
}

const computeHistogram = (data: number[], bins: number = 30) => {
  if (!data || data.length === 0) return [];
  const finite = data.filter(x => isFinite(x));
  if (finite.length === 0) return [];
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const step = (max - min) / bins;
  const hist = new Array(bins).fill(0);
  finite.forEach(x => {
    const binIdx = Math.min(Math.floor((x - min) / step), bins - 1);
    hist[binIdx]++;
  });
  return hist.map((count, i) => ({
    x: min + i * step + step / 2,
    count
  }));
};

export const PathwayAnalysis: React.FC<PathwayAnalysisProps> = ({ data }) => {
  const [selectedLibrary, setSelectedLibrary] = useState<string | null>(null);
  const [selectedAlgo, setSelectedAlgo] = useState<string>('GANomics_MA_to_RS');
  const [showAll, setShowAll] = useState(false);

  // 1. Determine base variables safely
  const libraries = useMemo(() => data ? Object.keys(data) : [], [data]);
  const currentLib = selectedLibrary || libraries[0] || '';
  const libData = data && currentLib ? data[currentLib] : null;
  const availableAlgos = useMemo(() => libData?.details ? Object.keys(libData.details) : [], [libData]);
  
  // Stats and details for the selected algorithm
  const currentDetails = useMemo(() => libData?.details?.[selectedAlgo] || [], [libData, selectedAlgo]);
  const currentStats = useMemo(() => libData?.stats?.[selectedAlgo] || [], [libData, selectedAlgo]);
  const currentConcordance = useMemo(() => libData?.concordance?.[selectedAlgo] || {}, [libData, selectedAlgo]);
  const currentDistributions = useMemo(() => libData?.distributions?.[selectedAlgo] || [], [libData, selectedAlgo]);

  // 2. Histogram Data
  const histogramData = useMemo(() => {
    if (!currentDistributions || currentDistributions.length === 0) return [];
    
    const nullDist = currentDistributions.map((d: any) => d.Null_Rho).filter((v: any) => v !== null);
    const bootDist = currentDistributions.map((d: any) => d.Bootstrap_Rho).filter((v: any) => v !== null);
    
    if (nullDist.length === 0 && bootDist.length === 0) return [];
    
    const allValues = [...nullDist, ...bootDist];
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const bins = 30;
    const step = (max - min) / bins;
    
    const result = [];
    for (let i = 0; i < bins; i++) {
      const binMin = min + i * step;
      const binMax = binMin + step;
      const nullCount = nullDist.filter((v: number) => v >= binMin && v < binMax).length;
      const bootCount = bootDist.filter((v: number) => v >= binMin && v < binMax).length;
      result.push({
        x: Number(binMin.toFixed(3)),
        'Permutation Null': nullCount,
        'Bootstrap ρ': bootCount
      });
    }
    return result;
  }, [currentDistributions]);

  // 3. Significance Preservation Ratio Curve
  const chartData = useMemo(() => {
    if (!libData?.details) return [];
    const results: any[] = [];
    const maxK = 100;
    for (let k = 5; k <= maxK; k += 5) {
      const entry: any = { k, label: `Top ${k}` };
      availableAlgos.forEach(algo => {
        const details = libData.details[algo] || [];
        const topK = details.slice(0, k);
        const significantInSyn = topK.filter((p: any) => p.Syn_P < 0.05).length;
        entry[algo] = significantInSyn / k;
      });
      results.push(entry);
    }
    return results;
  }, [libData, availableAlgos]);

  const filteredTableData = useMemo(() => {
    if (showAll) return currentDetails;
    return currentDetails.filter((p: any) => p.Real_P < 0.05);
  }, [currentDetails, showAll]);

  if (!data) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;
  if (libraries.length === 0) return <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>No pathway results available.</div>;

  const colors: any = {
    'Baseline': '#64748b',
    'GANomics_MA_to_RS': 'var(--primary-color)',
    'GANomics_RS_to_MA': '#16a34a'
  };

  const labels: any = {
    'Baseline': 'Native Platforms (MA vs RS)',
    'GANomics_MA_to_RS': 'GANomics (MA → RS)',
    'GANomics_RS_to_MA': 'GANomics (RS → MA)'
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* 1. Header: Library Selection */}
      <div className="card" style={{ padding: '1rem 2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ margin: 0 }}>Pathway Concordance: <span style={{ color: 'var(--primary-color)' }}>{currentLib.replace(/_/g, ' ')}</span></h3>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {libraries.map(lib => (
            <button key={lib} className={`chip ${currentLib === lib ? 'selected' : ''}`} onClick={() => setSelectedLibrary(lib)}>
              {lib.replace(/_/g, ' ')}
            </button>
          ))}
        </div>
      </div>

      {/* 2. Top Summary Metrics (Selected Algorithm) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
        <div className="card" style={{ padding: '1.25rem', textAlign: 'center' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 'bold', textTransform: 'uppercase' }}>Observed ρ</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: 'var(--primary-color)' }}>
            {currentConcordance.Observed_Rho?.toFixed(3) || currentConcordance.Spearman_Rho?.toFixed(3) || 'N/A'}
          </div>
        </div>
        <div className="card" style={{ padding: '1.25rem', textAlign: 'center' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 'bold', textTransform: 'uppercase' }}>Bootstrap Mean</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{currentConcordance.Bootstrap_Mean_Rho?.toFixed(3) || 'N/A'}</div>
          {currentConcordance.CI_95_Low && (
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
              95% CI [{currentConcordance.CI_95_Low.toFixed(2)}, {currentConcordance.CI_95_High.toFixed(2)}]
            </div>
          )}
        </div>
        <div className="card" style={{ padding: '1.25rem', textAlign: 'center' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 'bold', textTransform: 'uppercase' }}>Permutation p</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: (currentConcordance.Permutation_P || currentConcordance.P_Value) < 0.05 ? '#16a34a' : 'inherit' }}>
            {(currentConcordance.Permutation_P || currentConcordance.P_Value)?.toFixed(4) || 'N/A'}
          </div>
        </div>
        <div className="card" style={{ padding: '1.25rem', textAlign: 'center' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 'bold', textTransform: 'uppercase' }}>Glass Δ</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{currentConcordance.Glass_Delta?.toFixed(2) || 'N/A'}</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        {/* 3. Distribution Overlay Plot */}
        <div className="card" style={{ padding: '1.5rem' }}>
          <div style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h4 style={{ margin: 0 }}><BarChart2 size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} /> Rank Concordance Distribution</h4>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}><Info size={14} /> Bootstrap vs Permutation Null</div>
          </div>
          <div style={{ height: '300px' }}>
            {histogramData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={histogramData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="x" fontSize={10} />
                  <YAxis fontSize={10} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="Permutation Null" fill="#94a3b8" opacity={0.6} barSize={10} />
                  <Bar dataKey="Bootstrap ρ" fill="var(--primary-color)" opacity={0.7} barSize={10} />
                  {currentConcordance.Bootstrap_Mean_Rho && (
                    <ReferenceLine x={Number(currentConcordance.Bootstrap_Mean_Rho.toFixed(3))} stroke="red" label={{ value: 'Mean', position: 'top', fontSize: 10, fill: 'red' }} />
                  )}
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                No distribution data available for this comparison.
              </div>
            )}
          </div>
        </div>

        {/* 4. Significance Preservation Ratio Curve */}
        <div className="card" style={{ padding: '1.5rem' }}>
          <h4 style={{ margin: 0, marginBottom: '1rem' }}><TrendingUp size={18} style={{ verticalAlign: 'middle', marginRight: '8px' }} /> Top-K Preservation Ratio</h4>
          <div style={{ height: '300px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="label" fontSize={10} />
                <YAxis domain={[0, 1]} fontSize={10} />
                <Tooltip formatter={(val: number, name: string) => [val.toFixed(2), labels[name] || name]} />
                {availableAlgos.map((algo) => (
                  <Line 
                    key={algo} 
                    type="monotone" 
                    dataKey={algo} 
                    stroke={colors[algo] || '#999'} 
                    strokeWidth={algo === selectedAlgo ? 3 : 1} 
                    dot={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* 5. Detailed Results Table */}
      <section className="card" style={{ padding: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <h3 style={{ margin: 0 }}>Pathway Enrichment Details (t-test)</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 'bold' }}>Algorithm:</span>
              <select className="chip" value={selectedAlgo} onChange={(e) => setSelectedAlgo(e.target.value)} style={{ border: '1px solid var(--primary-color)', fontSize: '0.75rem', fontWeight: '600' }}>
                {availableAlgos.map(a => <option key(a) value(a)>{labels[a] || a}</option>)}
              </select>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <button className={`chip ${!showAll ? 'selected' : ''}`} onClick={() => setShowAll(false)} style={{ fontSize: '0.7rem' }}>
              <Filter size={12} /> Significant (Real p &lt; 0.05)
            </button>
            <button className={`chip ${showAll ? 'selected' : ''}`} onClick={() => setShowAll(true)} style={{ fontSize: '0.7rem' }}>
              <List size={12} /> Show All
            </button>
          </div>
        </div>
        <div style={{ overflowX: 'auto', maxHeight: '500px' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
            <thead style={{ position: 'sticky', top: 0, backgroundColor: '#fff', zIndex: 1 }}>
              <tr style={{ textAlign: 'left', borderBottom: '2px solid #eee' }}>
                <th style={{ padding: '0.75rem 1rem' }}>Pathway Name</th>
                <th style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>Genes</th>
                <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>t-stat (Real)</th>
                <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>p-val (Real)</th>
                <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>p-val (Syn)</th>
              </tr>
            </thead>
            <tbody>
              {filteredTableData.map((p: any, i: number) => (
                <tr key={i} style={{ borderBottom: '1px solid #f3f4f6', backgroundColor: p.Real_P < 0.05 && p.Syn_P < 0.05 ? '#f0fdf4' : 'transparent' }}>
                  <td style={{ padding: '0.6rem 1rem', fontWeight: '500' }}>{p.set}</td>
                  <td style={{ padding: '0.6rem 1rem', textAlign: 'center', color: 'var(--text-muted)' }}>{p.Real_K || p.Genes || '-'}</td>
                  <td style={{ padding: '0.6rem 1rem', textAlign: 'right' }}>{p.Real_T?.toFixed(2)}</td>
                  <td style={{ padding: '0.6rem 1rem', textAlign: 'right' }}>{p.Real_P?.toExponential(2)}</td>
                  <td style={{ padding: '0.6rem 1rem', textAlign: 'right', color: p.Syn_P < 0.05 ? '#16a34a' : 'inherit' }}>
                    {p.Syn_P ? p.Syn_P.toExponential(2) : 'N/A'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* 6. Top-K Jaccard Statistics */}
      {currentStats.length > 0 && (
        <div className="card" style={{ padding: '2rem' }}>
          <div style={{ marginBottom: '1.5rem' }}>
            <h3 style={{ margin: 0 }}>Top-K Jaccard Stability</h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Jaccard overlap of top-ranked pathways against Monte Carlo random expectation.</p>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ textAlign: 'left', borderBottom: '2px solid #eee' }}>
                  <th style={{ padding: '0.75rem 1rem' }}>Threshold</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>Observed Jaccard</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>Random Expectation</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>Significance (p-value)</th>
                </tr>
              </thead>
              <tbody>
                {currentStats.map((s: any, i: number) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f3f4f6' }}>
                    <td style={{ padding: '0.75rem 1rem', fontWeight: '600' }}>Top {s.K}</td>
                    <td style={{ padding: '0.75rem 1rem', textAlign: 'center', color: 'var(--primary-color)', fontWeight: 'bold' }}>{s.Observed_Jaccard?.toFixed(3)}</td>
                    <td style={{ padding: '0.75rem 1rem', textAlign: 'center', color: 'var(--text-muted)' }}>{s.Expected_Jaccard?.toFixed(3)}</td>
                    <td style={{ padding: '0.75rem 1rem', textAlign: 'right', color: s.P_Value < 0.05 ? '#16a34a' : 'inherit', fontWeight: '600' }}>
                      {s.P_Value < 0.001 ? '< 0.001' : s.P_Value.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
