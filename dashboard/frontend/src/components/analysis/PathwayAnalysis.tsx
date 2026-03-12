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
  
  // Specific data for GANomics only sections
  const ganomicsDetails = useMemo(() => libData?.details?.['GANomics'] || [], [libData]);
  const ganomicsStats = useMemo(() => libData?.stats?.['GANomics'] || [], [libData]);

  // Data for the ORA table (can still be switched)
  const currentDetails = useMemo(() => libData?.details?.[selectedAlgo] || [], [libData, selectedAlgo]);

  // 2. Prepare Slopegraph Data (-log10 FDR comparison) for GANOMICS ONLY
  const slopeData = useMemo(() => {
    if (!ganomicsDetails.length) return [];
    
    // Extract top pathways from GANomics
    const topPathways = ganomicsDetails.slice(0, topKPlot).map((d: any) => d.set);

    const pReal = ganomicsDetails.map((d: any) => d.Real_P);
    const pSyn = ganomicsDetails.map((d: any) => d.Syn_P || 1.0);
    const fdrReal = computeFDR(pReal);
    const fdrSyn = computeFDR(pSyn);

    return topPathways.map((pathName, idx) => {
      return {
        pathway: pathName,
        real_mlog: -Math.log10(Math.max(fdrReal[idx], 1e-20)),
        GANomics: -Math.log10(Math.max(fdrSyn[idx], 1e-20))
      };
    }).reverse();
  }, [ganomicsDetails, topKPlot]);

  // 3. Calculate Significance Ratio Curve for ALL algorithms
  const chartData = useMemo(() => {
    if (!libData?.details) return [];
    const results: any[] = [];
    for (let k = 5; k <= 100; k += 5) {
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

  const colors = ['var(--primary-color)', '#16a34a', '#818cf8', '#ea580c', '#db2777', '#7c3aed', '#0891b2', '#4ade80'];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* 1. Header: Library Selection */}
      <div className="card" style={{ padding: '1rem 2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ margin: 0 }}>Pathway Analysis: <span style={{ color: 'var(--primary-color)' }}>{currentLib.replace(/_/g, ' ')}</span></h3>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {libraries.map(lib => (
            <button key={lib} className={`chip ${currentLib === lib ? 'selected' : ''}`} onClick={() => setSelectedLibrary(lib)}>
              {lib.replace(/_/g, ' ')}
            </button>
          ))}
        </div>
      </div>

      {/* 2. TOP ITEM: Significance Preservation Ratio (ALL ALGORITHMS) */}
      <div className="card" style={{ padding: '2rem' }}>
        <div style={{ marginBottom: '2rem' }}>
          <h3>Significance Preservation Ratio (Comparative)</h3>
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

      {/* 3. ORA Results Table (with selector) */}
      <section className="card" style={{ padding: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <h3 style={{ margin: 0 }}>Functional Annotation (ORA)</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 'bold' }}>Algorithm:</span>
              <select className="chip" value={selectedAlgo} onChange={(e) => setSelectedAlgo(e.target.value)} style={{ border: '1px solid #ddd', fontSize: '0.75rem' }}>
                {availableAlgos.map(a => <option key={a} value={a}>{a}</option>)}
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
                <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>p-val (Real)</th>
                <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>p-val (Syn)</th>
              </tr>
            </thead>
            <tbody>
              {filteredTableData.map((p: any, i: number) => (
                <tr key={i} style={{ borderBottom: '1px solid #f3f4f6', backgroundColor: p.Real_P < 0.05 && p.Syn_P < 0.05 ? '#f0fdf4' : 'transparent' }}>
                  <td style={{ padding: '0.6rem 1rem', fontWeight: '500' }}>{p.set}</td>
                  <td style={{ padding: '0.6rem 1rem', textAlign: 'center', color: 'var(--text-muted)' }}>{p.Genes || '-'}</td>
                  <td style={{ padding: '0.6rem 1rem', textAlign: 'right' }}>{p.Real_P?.toExponential(2)}</td>
                  <td style={{ padding: '0.6rem 1rem', textAlign: 'right', color: p.Syn_P < 0.05 ? '#16a34a' : 'inherit' }}>{p.Syn_P ? p.Syn_P.toExponential(2) : 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div style={{ borderTop: '2px dashed #e2e8f0', margin: '2rem 0' }} />
      <div style={{ textAlign: 'center', marginBottom: '1rem' }}>
        <h2 style={{ fontSize: '1.25rem', color: 'var(--primary-color)' }}>GANomics In-depth Validation</h2>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Detailed statistical preservation metrics for the GANomics model.</p>
      </div>

      {/* 4. Statistical Validation (GANomics Only) */}
      {ganomicsStats.length > 0 && (
        <div className="card" style={{ padding: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
            <div>
              <h3 style={{ margin: 0 }}>Statistical Rigor (GANomics)</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Concordance against random expectation (Monte Carlo B=2000).</p>
            </div>
            <div style={{ padding: '0.75rem 1.25rem', borderRadius: '8px', background: 'var(--primary-light)', border: '1px solid var(--primary-color)' }}>
              <div style={{ fontSize: '0.7rem', color: 'var(--primary-color)', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Rank Concordance</div>
              <div style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>Spearman ρ = {ganomicsStats[0].Spearman_Rho?.toFixed(3)}</div>
            </div>
          </div>
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
                {ganomicsStats.map((s: any, i: number) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f3f4f6' }}>
                    <td style={{ padding: '0.75rem 1rem', fontWeight: '600' }}>Top {s.K}</td>
                    <td style={{ padding: '0.75rem 1rem', textAlign: 'center', color: 'var(--primary-color)', fontWeight: 'bold' }}>{s.Observed_Jaccard?.toFixed(3)}</td>
                    <td style={{ padding: '0.75rem 1rem', textAlign: 'center', color: 'var(--text-muted)' }}>{s.Expected_Jaccard?.toFixed(3)}</td>
                    <td style={{ padding: '0.75rem 1rem', textAlign: 'right', color: s.P_Value < 0.05 ? '#16a34a' : 'inherit', fontWeight: '600' }}>
                      {s.P_Value < 0.001 ? '< 0.001' : s.P_Value.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 5. Slopegraph (GANomics Only) */}
      {slopeData.length > 0 && (
        <div className="card" style={{ padding: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--primary-color)', marginBottom: '4px' }}>
                <TrendingUp size={18} />
                <span style={{ fontSize: '0.7rem', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Preservation Analysis</span>
              </div>
              <h3 style={{ margin: 0 }}>Pathway Significance Comparison (GANomics)</h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                Preservation of -log10(FDR) between Real and Synthetic data.
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

          <div style={{ height: `${slopeData.length * 45 + 100}px`, minHeight: '400px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={slopeData} layout="vertical" margin={{ top: 20, right: 50, left: 200, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={true} />
                <XAxis type="number" label={{ value: '-log10(FDR q)', position: 'insideBottom', offset: -10 }} />
                <YAxis dataKey="pathway" type="category" width={180} fontSize={10} tick={{ fill: 'var(--text-main)' }} />
                <Tooltip formatter={(value: any, name: string) => [Number(value).toFixed(2), name]} />
                
                <Scatter name="Real" dataKey="real_mlog" fill="#64748b" shape="circle" />
                <Scatter name="GANomics" dataKey="GANomics" fill="var(--primary-color)" shape="circle" />

                <Line 
                  data={[{ pathway: slopeData[0]?.pathway, threshold: -Math.log10(0.05) }, { pathway: slopeData[slopeData.length-1]?.pathway, threshold: -Math.log10(0.05) }]}
                  dataKey="threshold" stroke="#ef4444" strokeDasharray="5 5" dot={false} activeDot={false}
                  label={{ position: 'top', value: 'q < 0.05', fill: '#ef4444', fontSize: 10 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '2rem', marginTop: '1rem', fontSize: '0.8rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><div style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#64748b' }} /> Real Data</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><div style={{ width: '10px', height: '10px', borderRadius: '50%', background: 'var(--primary-color)' }} /> GANomics</div>
          </div>
        </div>
      )}
    </div>
  );
};
