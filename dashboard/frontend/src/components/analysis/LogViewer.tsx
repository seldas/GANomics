import React, { useState, useMemo } from 'react';
import { 
  Search, LineChart as LineChartIcon, Table as TableIcon, 
  ChevronsLeft, ChevronLeft, ChevronRight, ChevronsRight, Info 
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import type { LogResponse } from '../../types';
import { LOSS_METRICS } from '../../constants';

interface LogViewerProps {
  logData: LogResponse | null;
  runId: string;
}

export const LogViewer: React.FC<LogViewerProps> = ({ logData, runId }) => {
  const [logMode, setLogMode] = useState<'structured' | 'chart'>('chart');
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [windowSize, setWindowSize] = useState(20);
  const [windowStart, setWindowStart] = useState(0);
  const pageSize = 20;

  if (!logData) return null;

  const first = logData.structured[0] || {};
  const last = logData.structured[logData.structured.length - 1] || {};
  const totalEpochs = logData.structured.length;

  const getLossValue = (row: any, key: string) => {
    if (['G_A', 'G_B', 'D_A', 'D_B'].includes(key)) return row[key];
    if (key === 'Cycle') return (row.cycle_A || 0) + (row.cycle_B || 0);
    if (key === 'Feedback') return (row.feedback_A || 0) + (row.feedback_B || 0);
    if (key === 'IDT') return (row.idt_A || 0) + (row.idt_B || 0);
    return 0;
  };

  const fullChartData = logData.structured.map((d, i) => ({
    name: `E${d.epoch}`,
    index: i,
    G_A: d.G_A || 0, G_B: d.G_B || 0, D_A: d.D_A || 0, D_B: d.D_B || 0,
    Cycle: (d.cycle_A || 0) + (d.cycle_B || 0),
    Feedback: (d.feedback_A || 0) + (d.feedback_B || 0),
    IDT: (d.idt_A || 0) + (d.idt_B || 0),
  }));

  const chartData = useMemo(() => {
    return fullChartData.slice(windowStart, windowStart + windowSize);
  }, [fullChartData, windowStart, windowSize]);

  const filteredData = logData.structured.filter(d => 
    !searchTerm || 
    d.epoch.toString().includes(searchTerm) || 
    d.iters.toString().includes(searchTerm)
  );

  const totalPages = Math.ceil(filteredData.length / pageSize);
  const tableData = [...filteredData].reverse().slice((currentPage - 1) * pageSize, currentPage * pageSize);

  const visibleMetrics = ['G_A', 'G_B', 'D_A', 'D_B', 'Cycle', 'Feedback'];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <section className="card" style={{ padding: '1.5rem' }}>
        <h3 style={{ fontSize: '0.9rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Info size={16} /> Training Process Summary
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem' }}>
          <div className="card" style={{ margin: 0, padding: '0.75rem', backgroundColor: '#f8f9fa' }}>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 'bold' }}>SAMPLES (N)</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>{runId?.match(/Size_(\d+)/)?.[1] || "N/A"}</div>
          </div>
          <div className="card" style={{ margin: 0, padding: '0.75rem', backgroundColor: '#f8f9fa' }}>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 'bold' }}>TOTAL EPOCHS</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>{last.epoch || 0}</div>
          </div>
          <div className="card" style={{ margin: 0, padding: '0.75rem', backgroundColor: '#f8f9fa' }}>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 'bold' }}>ITERATIONS</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>{logData.total_lines}</div>
          </div>
        </div>

        <div style={{ marginTop: '1.5rem', overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: '0.75rem', textAlign: 'left', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
                <th style={{ padding: '0.5rem' }}>Loss Metric</th>
                <th style={{ padding: '0.5rem' }}>Initial (E{first.epoch})</th>
                <th style={{ padding: '0.5rem' }}>Final (E{last.epoch})</th>
                <th style={{ padding: '0.5rem' }}>Improvement</th>
              </tr>
            </thead>
            <tbody>
              {LOSS_METRICS.map(m => {
                const vInit = getLossValue(first, m.key);
                const vFinal = getLossValue(last, m.key);
                const diff = vInit - vFinal;
                const pct = vInit !== 0 ? ((diff / vInit) * 100).toFixed(1) : "0.0";
                return (
                  <tr key={m.key} style={{ borderBottom: '1px solid #f0f0f0' }}>
                    <td style={{ padding: '0.5rem', fontWeight: '600' }}>{m.label}</td>
                    <td style={{ padding: '0.5rem' }}>{vInit?.toFixed(4)}</td>
                    <td style={{ padding: '0.5rem', fontWeight: 'bold', color: m.color }}>{vFinal?.toFixed(4)}</td>
                    <td style={{ padding: '0.5rem', color: diff >= 0 ? 'var(--success-color)' : '#dc3545' }}>
                      {diff >= 0 ? '↓' : '↑'} {Math.abs(diff).toFixed(4)} ({pct}%)
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className="chip-grid">
            <button className={`chip ${logMode === 'chart' ? 'selected' : ''}`} onClick={() => setLogMode('chart')}>
              <LineChartIcon size={14} style={{ marginRight: '6px' }} /> Figure
            </button>
            <button className={`chip ${logMode === 'structured' ? 'selected' : ''}`} onClick={() => setLogMode('structured')}>
              <TableIcon size={14} style={{ marginRight: '6px' }} /> Table
            </button>
          </div>

          {logMode === 'chart' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 'bold' }}>RANGE:</span>
                <input 
                  type="number" 
                  value={windowSize} 
                  onChange={(e) => {
                    const val = Math.max(5, Math.min(parseInt(e.target.value) || 20, totalEpochs));
                    setWindowSize(val);
                    if (windowStart + val > totalEpochs) setWindowStart(Math.max(0, totalEpochs - val));
                  }}
                  style={{ width: '60px', padding: '4px 8px', borderRadius: '6px', border: '1px solid var(--border-color)', fontSize: '0.8rem' }}
                />
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: '200px' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 'bold' }}>PAN:</span>
                <input 
                  type="range" 
                  min={0} 
                  max={Math.max(0, totalEpochs - windowSize)} 
                  value={windowStart} 
                  onChange={(e) => setWindowStart(parseInt(e.target.value))}
                  style={{ flex: 1, cursor: 'pointer' }}
                />
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', minWidth: '60px', textAlign: 'right' }}>
                  E{fullChartData[windowStart]?.name?.substring(1) || 0} - E{fullChartData[Math.min(windowStart + windowSize - 1, totalEpochs - 1)]?.name?.substring(1) || 0}
                </span>
              </div>
            </div>
          )}
        </div>

        <div style={{ borderRadius: '12px', border: '1px solid var(--border-color)', backgroundColor: '#fff', overflow: 'hidden' }}>
          {logMode === 'chart' ? (
            <div style={{ width: '100%', height: '400px', padding: '1.5rem' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" interval={0} fontSize={10} />
                  <YAxis domain={['auto', 'auto']} fontSize={10} />
                  <Tooltip />
                  <Legend iconSize={10} wrapperStyle={{fontSize: '10px'}} />
                  {LOSS_METRICS.map(m => visibleMetrics.includes(m.key) && (
                    <Line key={m.key} type="monotone" dataKey={m.key} stroke={m.color} dot={{ r: 2 }} strokeWidth={1.5} animationDuration={300} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <>
              <div style={{ padding: '1rem', borderBottom: '1px solid var(--border-color)', display: 'flex', gap: '1rem', alignItems: 'center', backgroundColor: '#fdfdfd' }}>
                <div style={{ position: 'relative', flex: 1, maxWidth: '300px' }}>
                  <Search size={14} style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
                  <input
                    type="text" placeholder="Search epoch..."
                    style={{ padding: '0.5rem 0.75rem 0.5rem 2rem', width: '100%', borderRadius: '8px', border: '1px solid var(--border-color)', fontSize: '0.85rem' }}     
                    value={searchTerm} onChange={(e) => {setSearchTerm(e.target.value); setCurrentPage(1);}}
                  />
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Page {currentPage} of {totalPages || 1}</div>
              </div>
              <div style={{ flex: 1, overflowY: 'auto', maxHeight: '400px' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                  <thead style={{ position: 'sticky', top: 0, backgroundColor: '#fff', zIndex: 1 }}>
                    <tr style={{ textAlign: 'left', borderBottom: '2px solid var(--border-color)' }}>
                      <th style={{ padding: '0.75rem' }}>Epoch</th>
                      <th style={{ padding: '0.75rem' }}>Iters</th>
                      <th style={{ padding: '0.75rem' }}>G_A</th>
                      <th style={{ padding: '0.75rem' }}>G_B</th>
                      <th style={{ padding: '0.75rem' }}>D_A</th>
                      <th style={{ padding: '0.75rem' }}>D_B</th>
                      <th style={{ padding: '0.75rem' }}>Cycle</th>
                      <th style={{ padding: '0.75rem' }}>Feedback</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tableData.map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
                        <td style={{ padding: '0.75rem', fontWeight: '600' }}>{row.epoch}</td>
                        <td style={{ padding: '0.75rem' }}>{row.iters}</td>
                        <td style={{ padding: '0.75rem' }}>{row.G_A?.toFixed(3)}</td>
                        <td style={{ padding: '0.75rem' }}>{row.G_B?.toFixed(3)}</td>
                        <td style={{ padding: '0.75rem' }}>{row.D_A?.toFixed(3)}</td>
                        <td style={{ padding: '0.75rem' }}>{row.D_B?.toFixed(3)}</td>
                        <td style={{ padding: '0.75rem' }}>{((row.cycle_A || 0) + (row.cycle_B || 0)).toFixed(3)}</td>
                        <td style={{ padding: '0.75rem' }}>{((row.feedback_A || 0) + (row.feedback_B || 0)).toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{ padding: '1rem', borderTop: '1px solid var(--border-color)', display: 'flex', justifyContent: 'center', gap: '0.5rem', backgroundColor: '#fdfdfd' }}>
                <button className="chip" disabled={currentPage === 1} onClick={() => setCurrentPage(1)}><ChevronsLeft size={14}/></button>
                <button className="chip" disabled={currentPage === 1} onClick={() => setCurrentPage(prev => prev - 1)}><ChevronLeft size={14}/></button>
                <span style={{ display: 'flex', alignItems: 'center', padding: '0 1rem', fontSize: '0.85rem' }}>{currentPage} / {totalPages || 1}</span>
                <button className="chip" disabled={currentPage >= totalPages} onClick={() => setCurrentPage(prev => prev + 1)}><ChevronRight size={14}/></button>
                <button className="chip" disabled={currentPage >= totalPages} onClick={() => setCurrentPage(totalPages)}><ChevronsRight size={14}/></button>
              </div>
            </>
          )}
        </div>
      </section>
    </div>
  );
};
