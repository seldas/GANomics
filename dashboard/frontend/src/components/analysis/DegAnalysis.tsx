import React from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { Loader2, Download } from 'lucide-react';

interface DegAnalysisProps {
  data: any | null;
  runId?: string;
}

export const DegAnalysis: React.FC<DegAnalysisProps> = ({ data, runId }) => {
  if (!data) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;
  
  const algos = Object.keys(data);
  if (algos.length === 0) return <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>No DEG results available.</div>;

  const handleDownload = (filename: string) => {
    if (!runId) return;
    const url = `http://localhost:8832/api/runs/${runId}/deg/download?filename=${filename}`;
    window.open(url, '_blank');
  };

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

  const colors: any = {
    'Baseline': '#64748b',
    'GANomics_MA_to_RS': 'var(--primary-color)',
    'GANomics_RS_to_MA': '#16a34a'
  };

  const labels: any = {
    'Baseline': 'Native Platforms (MA vs RS)',
    'GANomics_MA_to_RS': 'GANomics (MA -> RS)',
    'GANomics_RS_to_MA': 'GANomics (RS -> MA)'
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
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
              <Tooltip 
                content={({ active, payload, label }: any) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="card" style={{ padding: '10px', fontSize: '0.75rem', border: '1px solid #ddd' }}>
                        <p style={{ fontWeight: 'bold', margin: '0 0 5px 0' }}>Threshold: {label}</p>
                        {payload.map((p: any, i: number) => {
                          const algo = p.dataKey;
                          const pt = data[algo].find((pt: any) => `p < ${pt.threshold}` === label);
                          return (
                            <div key={i} style={{ color: p.color, marginBottom: '4px' }}>
                              <span style={{ fontWeight: 'bold' }}>{labels[algo] || algo}:</span> {p.value.toFixed(3)}
                              <br />
                              <span style={{ color: '#666', fontSize: '0.7rem' }}>
                                (Overlap: {pt?.n_overlap} | Real: {pt?.n_real} | Fake: {pt?.n_fake})
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend verticalAlign="top" formatter={(value) => labels[value] || value} />
              {algos.map((algo, i) => (
                <Line 
                  key={algo} 
                  type="monotone" 
                  dataKey={algo} 
                  stroke={colors[algo] || '#999'} 
                  strokeWidth={algo.includes('GANomics') ? 3 : 2}
                  strokeDasharray={algo === 'Baseline' ? "5 5" : "0"}
                  dot={{ r: algo.includes('GANomics') ? 5 : 3 }}
                  activeDot={{ r: 8 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {runId && (
        <div className="card" style={{ padding: '2rem' }}>
          <h3>Significant Gene Lists (p &lt; 0.05)</h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
            Download the lists of genes identified as differentially expressed (p &lt; 0.05), ranked by p-value.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
            <button className="btn btn-outline" onClick={() => handleDownload('DEGs_Microarray_Real.txt')} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
              <Download size={16} /> Microarray (Real)
            </button>
            <button className="btn btn-outline" onClick={() => handleDownload('DEGs_RNAseq_Real.txt')} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
              <Download size={16} /> RNA-Seq (Real)
            </button>
            <button className="btn btn-outline" onClick={() => handleDownload('DEGs_Microarray_Fake.txt')} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
              <Download size={16} /> Microarray (Fake)
            </button>
            <button className="btn btn-outline" onClick={() => handleDownload('DEGs_RNAseq_Fake.txt')} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
              <Download size={16} /> RNA-Seq (Fake)
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
