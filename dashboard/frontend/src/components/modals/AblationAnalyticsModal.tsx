import React from 'react';
import { X, Loader2, Info } from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';

interface AblationAnalyticsModalProps {
  category: string | null;
  onClose: () => void;
  projectName: string;
  ablationLogs: any[] | null;
  sensitivityType: 'beta' | 'lambda';
  onSetSensitivityType: (type: 'beta' | 'lambda') => void;
}

export const AblationAnalyticsModal: React.FC<AblationAnalyticsModalProps> = ({
  category, onClose, projectName, ablationLogs, sensitivityType, onSetSensitivityType
}) => {
  if (!category) return null;

  return (
    <div className="modal-overlay" style={{ zIndex: 2000 }}>
      <div className="modal-content" style={{ maxWidth: '1000px', width: '90%', maxHeight: '90vh', overflowY: 'auto' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h2 style={{ margin: 0, textTransform: 'uppercase' }}>{category} Analytics: {projectName}</h2>
          <button className="chip" onClick={onClose}><X size={18} /></button>
        </div>

        {!ablationLogs ? (
          <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>
        ) : ablationLogs.length === 0 ? (
          <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>No logs found for this category.</div>
        ) : (() => {
          // Group and Parse Data
          const groups: Record<string, { logs: any[], type: string, val: number, name: string }> = {};
          
          ablationLogs.forEach(log => {
            let name = "Other";
            let type = "other";
            let val = 0;

            if (category === 'size') {
              const m = log.run_id.match(/Size_(\d+)/);
              if (m) { val = parseInt(m[1]); name = `Size ${val}`; type = 'size'; }
            } else if (category === 'architecture') {
              if (log.run_id.includes("AtoB")) { name = "A → B Only"; type = "arch"; val = 1; }
              else if (log.run_id.includes("BtoA")) { name = "B → A Only"; type = "arch"; val = 2; }
              else { name = "Both"; type = "arch"; val = 0; }
            } else if (category === 'sensitivity') {
              const bm = log.run_id.match(/Beta_([\d.]+)/);
              const lm = log.run_id.match(/Lambda_([\d.]+)/);
              if (bm) { val = parseFloat(bm[1]); name = `Beta ${val.toFixed(1)}`; type = 'beta'; }
              else if (lm) { val = parseFloat(lm[1]); name = `Lambda ${val.toFixed(1)}`; type = 'lambda'; }
            }

            if (!groups[name]) groups[name] = { logs: [], type, val, name };
            groups[name].logs.push(log);
          });

          const metricKeys = ['G_A', 'G_B', 'D_A', 'D_B', 'cycle_A', 'cycle_B', 'feedback_A', 'feedback_B'];
          let summaryData = Object.values(groups).map(g => {
            const stats: any = { name: g.name, type: g.type, val: g.val };
            metricKeys.forEach(k => {
              const finalValues = g.logs.map(l => l.last[k] || 0);
              const avgFinal = (finalValues.reduce((a, b) => a + b, 0) / (finalValues.length || 1));
              const stdFinal = Math.sqrt(finalValues.map(x => Math.pow(x - avgFinal, 2)).reduce((a, b) => a + b, 0) / (finalValues.length || 1));
              stats[k] = { avgFinal, stdFinal };
            });
            return stats;
          });

          // Filtering and Sorting
          if (category === 'sensitivity') {
            summaryData = summaryData
              .filter(s => s.type === sensitivityType)
              .sort((a, b) => a.val - b.val);
          } else {
            summaryData = summaryData.sort((a, b) => a.val - b.val);
          }

          const formatValue = (val: { avgFinal: number, stdFinal: number }) => {
            if (!val || val.avgFinal < 0.00001) return <span style={{ color: 'var(--text-muted)', fontStyle: 'italic', fontSize: '0.75rem' }}>Unavailable</span>;
            return (
              <>
                <span style={{ fontWeight: 'bold' }}>{val.avgFinal.toFixed(4)}</span>
                <div style={{ fontSize: '0.7rem', opacity: 0.6 }}>±{val.stdFinal.toFixed(4)}</div>
              </>
            );
          };

          return (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
              {category === 'sensitivity' && (
                <div style={{ display: 'flex', justifyContent: 'center', backgroundColor: '#f1f5f9', padding: '0.5rem', borderRadius: '12px', alignSelf: 'center' }}>
                  <button 
                    className={`chip ${sensitivityType === 'beta' ? 'selected' : ''}`} 
                    style={{ border: 'none', padding: '0.5rem 1.5rem' }} 
                    onClick={() => onSetSensitivityType('beta')}
                  >
                    Beta (Feedback Weight)
                  </button>
                  <button 
                    className={`chip ${sensitivityType === 'lambda' ? 'selected' : ''}`} 
                    style={{ border: 'none', padding: '0.5rem 1.5rem' }} 
                    onClick={() => onSetSensitivityType('lambda')}
                  >
                    Lambda (Cycle Weight)
                  </button>
                </div>
              )}

              {category !== 'architecture' && (
                <section className="card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                    <h3 style={{ fontSize: '0.9rem', margin: 0 }}>Final Loss Values ({category === 'sensitivity' ? (sensitivityType === 'beta' ? 'β Sensitivity' : 'λ Sensitivity') : 'Trend'})</h3>
                  </div>
                  <div style={{ height: '400px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={summaryData} margin={{ bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="name" angle={0} textAnchor="middle" interval={0} fontSize={11} />
                        <YAxis label={{ value: 'Avg. Final Loss', angle: -90, position: 'insideLeft' }} fontSize={11} />
                        <Tooltip />
                        <Legend verticalAlign="top" height={36}/>
                        <Bar dataKey="G_A.avgFinal" name="Gen A" fill="#007bff" />
                        <Bar dataKey="G_B.avgFinal" name="Gen B" fill="#0056b3" />
                        <Bar dataKey="D_A.avgFinal" name="Disc A" fill="#ef4444" />
                        <Bar dataKey="D_B.avgFinal" name="Disc B" fill="#991b1b" />
                        <Bar dataKey="cycle_A.avgFinal" name="Cycle A" fill="#10b981" />
                        <Bar dataKey="cycle_B.avgFinal" name="Cycle B" fill="#065f46" />
                        <Bar dataKey="feedback_A.avgFinal" name="Feedback A" fill="#f59e0b" />
                        <Bar dataKey="feedback_B.avgFinal" name="Feedback B" fill="#92400e" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </section>
              )}

              <section className="card">
                <h3 style={{ fontSize: '0.9rem', marginBottom: '1.5rem' }}>Detailed Metrics Comparison (Mean ± Std)</h3>
                <div style={{ overflowX: 'auto' }}>
                  {category === 'architecture' ? (() => {
                    const both = summaryData.find(s => s.name === 'Both');
                    const atob = summaryData.find(s => s.name === 'A → B Only');
                    const btoa = summaryData.find(s => s.name === 'B → A Only');

                    const archRows = [
                      { label: 'Generator A (A→B)', key: 'G_A' },
                      { label: 'Discriminator B (Target: B)', key: 'D_B' },
                      { label: 'Feedback Alignment A', key: 'feedback_A' },
                      { label: 'Generator B (B→A)', key: 'G_B' },
                      { label: 'Discriminator A (Target: A)', key: 'D_A' },
                      { label: 'Feedback Alignment B', key: 'feedback_B' },
                    ];

                    return (
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                        <thead>
                          <tr style={{ backgroundColor: '#f9fafb', borderBottom: '2px solid var(--border-color)' }}>
                            <th style={{ padding: '1rem', textAlign: 'left' }}>Loss Metric</th>
                            <th style={{ padding: '1rem', textAlign: 'center', backgroundColor: '#f0f9ff' }}>Both (Full Model)</th>
                            <th style={{ padding: '1rem', textAlign: 'center' }}>A → B Only</th>
                            <th style={{ padding: '1rem', textAlign: 'center' }}>B → A Only</th>
                          </tr>
                        </thead>
                        <tbody>
                          {archRows.map(row => (
                            <tr key={row.key} style={{ borderBottom: '1px solid #f3f4f6' }}>
                              <td style={{ padding: '0.75rem 1rem', fontWeight: '600' }}>{row.label}</td>
                              <td style={{ padding: '0.75rem 1rem', textAlign: 'center', color: 'var(--primary-color)', backgroundColor: '#f8fafc' }}>
                                {both ? formatValue(both[row.key]) : '-'}
                              </td>
                              <td style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>
                                {atob ? formatValue(atob[row.key]) : '-'}
                              </td>
                              <td style={{ padding: '0.75rem 1rem', textAlign: 'center' }}>
                                {btoa ? formatValue(btoa[row.key]) : '-'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    );
                  })() : (
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.7rem' }}>
                        <thead>
                          <tr style={{ backgroundColor: '#f9fafb', borderBottom: '2px solid var(--border-color)' }}>
                            <th style={{ padding: '0.75rem', textAlign: 'left' }}>Variant</th>
                            {metricKeys.map(k => (
                              <th key={k} style={{ padding: '0.75rem', textAlign: 'center' }}>
                                {k.replace('_', ' ')}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {summaryData.map(s => (
                            <tr key={s.name} style={{ borderBottom: '1px solid #f0f0f0' }}>
                              <td style={{ padding: '0.75rem', fontWeight: 'bold' }}>{s.name}</td>
                              {metricKeys.map(k => (
                                <td key={`${s.name}-${k}`} style={{ padding: '0.5rem', textAlign: 'center' }}>
                                  {s[k].avgFinal.toFixed(3)} <span style={{ color: 'var(--text-muted)', fontSize: '0.6rem' }}>±{s[k].stdFinal.toFixed(3)}</span>
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </div>
                </section>
            </div>
          );
        })()}
      </div>
    </div>
  );
};
