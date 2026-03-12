import React, { useState, useMemo } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, Cell, ComposedChart, ZAxis
} from 'recharts';
import { Loader2, Filter, List, TrendingUp } from 'lucide-react';

interface PathwayAnalysisProps {
  data: any | null;
}

// Benjamini-Hochberg FDR helper
const computeFDR = (pValues: number[]) => {
  const n = pValues.length;
  if (n === 0) return [];
  const sorted = pValues.map((p, i) => ({ p, i })).sort((a, b) => a.p - b.p);
  const qValues = new Array(n);
  let minQ = 1;
  for (let j = n - 1; j >= 0; j--) {
    const q = (sorted[j].p * n) / (j + 1);
    minQ = Math.min(minQ, q);
    qValues[sorted[j].i] = Math.min(minQ, 1);
  }
  return qValues;
};

export const PathwayAnalysis: React.FC<PathwayAnalysisProps> = ({ data }) => {
  const [selectedLibrary, setSelectedLibrary] = useState<string | null>(null);
  const [selectedAlgo, setSelectedAlgo] = useState<string>('GANomics');
  const [showAll, setShowAll] = useState(false);
  const [topKPlot, setTopKPlot] = useState(10);

  // 1. Determine base variables safely
  const libraries = useMemo(() => data ? Object.keys(data) : [], [data]);
  const currentLib = selectedLibrary || libraries[0] || '';
  const libData = data && currentLib ? data[currentLib] : null;
  const availableAlgos = useMemo(() => libData?.details ? Object.keys(libData.details) : [], [libData]);
  const currentDetails = useMemo(() => libData?.details?.[selectedAlgo] || [], [libData, selectedAlgo]);
  const currentStats = useMemo(() => libData?.stats?.[selectedAlgo] || [], [libData, selectedAlgo]);

  // 2. Prepare Slopegraph Data (-log10 FDR comparison) for ALL algorithms
  const slopeData = useMemo(() => {
    if (!libData?.details) return [];
    
    // We use GANomics as the reference for which Top-K pathways to show
    const refAlgo = 'GANomics';
    const refDetails = libData.details[refAlgo] || libData.details['Baseline'] || [];
    if (!refDetails.length) return [];

    // Extract pathway names from top-K of reference
    const topPathways = refDetails.slice(0, topKPlot).map((d: any) => d.set);

    // Build data structure: { pathway: string, GANomics: mlog, Baseline: mlog, ... }
    return topPathways.map(pathName => {
      const entry: any = { pathway: pathName };
      
      availableAlgos.forEach(algo => {
        const details = libData.details[algo] || [];
        const pValues = details.map((d: any) => d.Real_P);
        const fdrValues = computeFDR(pValues);
        const pathwayIdx = details.findIndex((d: any) => d.set === pathName);
        
        if (pathwayIdx !== -1) {
          const q = fdrValues[pathwayIdx];
          entry[algo] = -Math.log10(Math.max(q, 1e-20));
          // Store real mlog only once (it should be the same across algos, but we'll use Baseline/GANomics real)
          if (algo === refAlgo) {
            entry.real_mlog = -Math.log10(Math.max(details[pathwayIdx].Real_P, 1e-20)); // Approximate for visual anchor
          }
        }
      });
      return entry;
    }).reverse();
  }, [libData, availableAlgos, topKPlot]);

  // 3. Calculate Significance Ratio Curve for ALL algorithms
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

  // 3. Filter Table
  const filteredTableData = useMemo(() => {
    if (showAll) return currentDetails;
    return currentDetails.filter((p: any) => p.Real_P < 0.05);
  }, [currentDetails, showAll]);

  if (!data) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;
  if (libraries.length === 0) return <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>No pathway results available.</div>;
  if (!libData?.details) return <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>Loading pathway details...</div>;

  const colors = ['var(--primary-color)', '#16a34a', '#818cf8', '#ea580c', '#db2777', '#7c3aed', '#0891b2', '#4ade80'];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Library Selection Header */}
      <div className="card" style={{ padding: '1rem 2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ margin: 0 }}>Pathway Library: <span style={{ color: 'var(--primary-color)' }}>{currentLib.replace(/_/g, ' ')}</span></h3>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {libraries.map(lib => (
            <button key={lib} className={`chip ${currentLib === lib ? 'selected' : ''}`} onClick={() => setSelectedLibrary(lib)}>
              {lib.replace(/_/g, ' ')}
            </button>
          ))}
        </div>
      </div>

      {/* 1. Slopegraph: Significance Comparison (ALL ALGORITHMS) */}
      {slopeData.length > 0 && (
        <div className="card" style={{ padding: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--primary-color)', marginBottom: '4px' }}>
                <TrendingUp size={18} />
                <span style={{ fontSize: '0.7rem', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Unified Comparison</span>
              </div>
              <h3 style={{ margin: 0 }}>Cross-Algorithm Significance Preservation</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                Preservation of -log10(FDR) across all methods for Top {topKPlot} pathways.
              </p>
            </div>
            <div style={{ display: 'flex', gap: '4px' }}>
              {[10, 20, 30].map(k => (
                <button key={k} className={`chip ${topKPlot === k ? 'selected' : ''}`} onClick={() => setTopKPlot(k)} style={{ fontSize: '0.7rem' }}>
                  Top {k}
                </button>
              ))}
            </div>
          </div>

          <div style={{ height: `${slopeData.length * 45 + 120}px`, minHeight: '500px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={slopeData} layout="vertical" margin={{ top: 20, right: 120, left: 200, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={true} />
                <XAxis type="number" label={{ value: '-log10(FDR q)', position: 'insideBottom', offset: -10 }} />
                <YAxis dataKey="pathway" type="category" width={180} fontSize={10} tick={{ fill: 'var(--text-main)' }} />
                <Tooltip 
                  formatter={(value: any, name: string) => [Number(value).toFixed(2), name]}
                />
                <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '20px' }} />
                
                {/* Dots for Real Anchor */}
                <Scatter name="Real Reference" dataKey="real_mlog" fill="#cbd5e1" shape="diamond" />

                {/* Lines and Dots for each Algorithm */}
                {availableAlgos.map((algo, i) => (
                  <Scatter 
                    key={algo} 
                    name={algo} 
                    dataKey={algo} 
                    fill={algo === 'Baseline' ? '#64748b' : (algo === 'GANomics' ? 'var(--primary-color)' : colors[i % colors.length])} 
                  />
                ))}

                {/* FDR 0.05 Threshold Line */}
                <Line 
                  data={[{ pathway: slopeData[0]?.pathway, threshold: -Math.log10(0.05) }, { pathway: slopeData[slopeData.length-1]?.pathway, threshold: -Math.log10(0.05) }]}
                  dataKey="threshold"
                  stroke="#ef4444"
                  strokeDasharray="5 5"
                  dot={false}
                  activeDot={false}
                  label={{ position: 'top', value: 'q < 0.05', fill: '#ef4444', fontSize: 10 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* 2. Figure Panel: Significance Ratio (ALL ALGORITHMS) */}
      <div className="card" style={{ padding: '2rem' }}>
        <div style={{ marginBottom: '2rem' }}>
          <h3>Significance Preservation Ratio</h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Fraction of Top X pathways (ranked by Real P-value) that are also significant (p &lt; 0.05) in Synthetic data.
          </p>
        </div>

        <div style={{ height: '400px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="k" label={{ value: 'Top X Pathways (Real)', position: 'insideBottom', offset: -10 }} />
              <YAxis domain={[0, 1]} label={{ value: 'Sig. Ratio (Syn p < 0.05)', angle: -90, position: 'insideLeft' }} />
              <Tooltip formatter={(val: number, name: string) => [val.toFixed(2), name]} />
              <Legend verticalAlign="top" />
              {availableAlgos.map((algo, i) => (
                <Line 
                  key={algo} 
                  type="monotone" 
                  dataKey={algo} 
                  stroke={algo === 'Baseline' ? '#64748b' : (algo === 'GANomics' ? 'var(--primary-color)' : colors[i % colors.length])} 
                  strokeWidth={algo === 'GANomics' ? 3 : (algo === 'Baseline' ? 2 : 1.5)} 
                  strokeDasharray={algo === 'Baseline' ? "5 5" : "0"}
                  dot={{ r: algo === 'GANomics' ? 4 : 2 }} 
                  activeDot={{ r: 8 }} 
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ borderTop: '1px solid #e2e8f0', margin: '1rem 0' }} />

      {/* Algorithm Specific Section */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', background: '#f8fafc', padding: '1rem 2rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
        <div style={{ fontWeight: 'bold', fontSize: '0.9rem' }}>Detailed analysis for:</div>
        <select 
          className="chip" 
          value={selectedAlgo} 
          onChange={(e) => setSelectedAlgo(e.target.value)} 
          style={{ border: '1px solid var(--primary-color)', fontWeight: '600', padding: '4px 12px' }}
        >
          {availableAlgos.map(a => <option key={a} value={a}>{a}</option>)}
        </select>
      </div>

      {/* Statistical Validation Panel */}
      {currentStats.length > 0 && (
        <div className="card" style={{ padding: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
            <div>
              <h3 style={{ margin: 0 }}>Statistical Validation: {selectedAlgo}</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Rigorous comparison against random expectations.
              </p>
            </div>
            <div style={{ padding: '0.75rem 1.25rem', borderRadius: '8px', background: 'var(--primary-light)', border: '1px solid var(--primary-color)' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--primary-color)', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Rank Concordance</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>
                Spearman ρ = {currentStats[0].Spearman_Rho?.toFixed(3)}
              </div>
            </div>
          </div>
          {/* ... table content remains same ... */}

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ textAlign: 'left', borderBottom: '2px solid #eee' }}>
                  <th style={{ padding: '0.75rem 1rem' }}>Top-K Pathways</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>Observed Jaccard</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>Random Expectation</th>
                  <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>Significance (p-value)</th>
                </tr>
              </thead>
              <tbody>
                {currentStats.map((s: any, i: number) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f3f4f6' }}>
                    <td style={{ padding: '0.75rem 1rem', fontWeight: '600' }}>Top {s.K}</td>
                    <td style={{ padding: '0.75rem 1rem', textAlign: 'center', color: 'var(--primary-color)', fontWeight: 'bold' }}>
                      {s.Observed_Jaccard?.toFixed(3)}
                    </td>
                    <td style={{ padding: '0.75rem 1rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                      {s.Expected_Jaccard?.toFixed(3)}
                    </td>
                    <td style={{ padding: '0.75rem 1rem', textAlign: 'right', color: s.P_Value < 0.05 ? '#16a34a' : 'inherit', fontWeight: '600' }}>
                      {s.P_Value < 0.001 ? '< 0.001' : s.P_Value.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '1rem', fontStyle: 'italic' }}>
            * Random expectation calculated via Monte Carlo simulation (B=2000) using Hypergeometric null model.
          </p>
        </div>
      )}

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
                    backgroundColor: p.Real_P < 0.05 && p.Syn_P < 0.05 ? '#f0fdf4' : 'transparent'
                  }}>
                    <td style={{ padding: '0.6rem 1rem', fontWeight: '500' }}>{p.set}</td>
                    <td style={{ padding: '0.6rem 1rem', textAlign: 'center', color: 'var(--text-muted)' }}>{p.Genes || '-'}</td>
                    <td style={{ padding: '0.6rem 1rem', textAlign: 'right' }}>{p.Real_P?.toExponential(2)}</td>
                    <td style={{ padding: '0.6rem 1rem', textAlign: 'right', color: p.Syn_P < 0.05 ? '#16a34a' : 'inherit' }}>
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
