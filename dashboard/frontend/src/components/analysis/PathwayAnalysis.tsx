import React, { useMemo } from 'react';
import { 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Bar, ComposedChart, ReferenceLine, Area
} from 'recharts';
import { Loader2, BarChart2 } from 'lucide-react';

interface PathwayAnalysisProps {
  data: any | null;
}

const HistogramPlot = ({ distributions, concordance, title, color }: any) => {
  const { histogramData, xDomain } = useMemo(() => {
    if (!distributions || distributions.length === 0) return { histogramData: [], xDomain: [0, 1] };
    
    const nullDist = distributions.map((d: any) => d.Null_Rho).filter((v: any) => v !== null);
    const bootDist = distributions.map((d: any) => d.Bootstrap_Rho).filter((v: any) => v !== null);
    const obsRho = concordance?.Observed_Rho || concordance?.Spearman_Rho || 0;

    if (nullDist.length === 0 && bootDist.length === 0) return { histogramData: [], xDomain: [0, 1] };
    
    const allValues = [...nullDist, ...bootDist, obsRho];
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const range = max - min;
    const padding = range * 0.1 || 0.1;
    
    const bins = 30;
    const step = (max - min) / bins || 0.01;
    
    const result = [];
    for (let i = 0; i < bins; i++) {
      const binMin = min + i * step;
      const binMax = binMin + step;
      const nullCount = nullDist.filter((v: number) => v >= binMin && v < binMax).length;
      const bootCount = bootDist.filter((v: number) => v >= binMin && v < binMax).length;
      result.push({
        x: Number((binMin + step / 2).toFixed(3)),
        'Permutation Null': nullCount,
        'Bootstrap ρ': bootCount
      });
    }
    return { 
      histogramData: result, 
      xDomain: [min - padding, max + padding] 
    };
  }, [distributions, concordance]);

  const rho = concordance?.Observed_Rho || concordance?.Spearman_Rho || 0;
  const p = concordance?.Permutation_P || concordance?.P_Value || 0;

  return (
    <div className="card" style={{ padding: '1.5rem', flex: 1 }}>
      <div style={{ marginBottom: '1rem', textAlign: 'center' }}>
        <h4 style={{ margin: 0, color: 'var(--text-main)' }}>{title}</h4>
        <div style={{ fontSize: '0.85rem', marginTop: '4px', fontWeight: 'bold' }}>
          rho={rho.toFixed(3)}, perm-p={p < 0.0001 ? p.toExponential(2) : p.toFixed(4)}
        </div>
      </div>
      <div style={{ height: '250px' }}>
        {histogramData.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={histogramData} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis 
                type="number"
                dataKey="x" 
                domain={xDomain}
                fontSize={10} 
                tickFormatter={(val) => val.toFixed(2)}
                label={{ value: 'Spearman ρ', position: 'bottom', offset: 0, fontSize: 10 }} 
              />
              <YAxis fontSize={10} hide />
              <Tooltip />
              <Legend verticalAlign="top" height={36} iconSize={10} wrapperStyle={{ fontSize: '10px' }} />
              
              <Bar dataKey="Permutation Null" fill="#94a3b8" opacity={0.4} barSize={10} name="Null Dist (Permutation)" />
              <Area type="monotone" dataKey="Bootstrap ρ" fill={color} stroke={color} fillOpacity={0.3} name="Bootstrap Dist (Stability)" />
              
              <ReferenceLine 
                x={rho} 
                stroke="red" 
                strokeWidth={2}
                strokeDasharray="5 5"
                label={{ value: 'Observed', position: 'top', fontSize: 10, fill: 'red', fontWeight: 'bold' }} 
              />
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
            No data available.
          </div>
        )}
      </div>
    </div>
  );
};

export const PathwayAnalysis: React.FC<PathwayAnalysisProps> = ({ data }) => {
  const libraries = useMemo(() => data ? Object.keys(data) : [], [data]);
  const currentLib = libraries[0] || '';
  const libData = data && currentLib ? data[currentLib] : null;

  if (!data) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;
  if (libraries.length === 0) return <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>No pathway results available.</div>;

  const maDist = libData?.distributions?.['GANomics_RS_to_MA'];
  const maStats = libData?.concordance?.['GANomics_RS_to_MA'];
  
  const rsDist = libData?.distributions?.['GANomics_MA_to_RS'];
  const rsStats = libData?.concordance?.['GANomics_MA_to_RS'];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div className="card" style={{ padding: '1rem 2rem' }}>
        <h3 style={{ margin: 0, textAlign: 'center' }}>
          <BarChart2 size={20} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
          Gene-Set Rank Concordance: Null vs Observed ({currentLib.replace(/_/g, ' ')})
        </h3>
      </div>

      <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'stretch' }}>
        <HistogramPlot 
          title="[MA] Microarray gene-set rank concordance"
          distributions={maDist}
          concordance={maStats}
          color="#3b82f6"
        />
        <HistogramPlot 
          title="[RS] RNA-Seq gene-set rank concordance"
          distributions={rsDist}
          concordance={rsStats}
          color="#16a34a"
        />
      </div>

      <div className="card" style={{ padding: '1rem', fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
        <p style={{ margin: 0 }}>
          <b>Interpretation:</b> The gray bars represent the <b>Null Distribution</b> (permutation of pathway ranks). 
          The colored area represents the <b>Bootstrap Distribution</b> (resampling stability). 
          The <span style={{ color: 'red', fontWeight: 'bold' }}>Red Dashed Line</span> is the <b>Observed ρ</b>. 
          If the red line is significantly to the right of the gray bars (high rho, low p), the pathway hierarchy is well-preserved.
        </p>
      </div>
    </div>
  );
};
