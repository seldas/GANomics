import React from 'react';
import { Loader2 } from 'lucide-react';

interface SyncStatusDetailsProps {
  data: any | null;
}

export const SyncStatusDetails: React.FC<SyncStatusDetailsProps> = ({ data }) => {
  if (!data) return <div style={{ padding: '4rem', textAlign: 'center' }}><Loader2 className="animate-spin" size={32} /></div>;
  if (!data.exists) return <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>Sync data not found for this run.</div>;

  const details = data.details;
  const platforms = ["Microarray", "RNA-Seq"];
  const CheckIcon = ({ checked }: { checked: boolean }) => (
    <div style={{ color: checked ? 'var(--success-color)' : '#ff4d4f', fontWeight: 'bold', fontSize: '1.2rem' }}>
      {checked ? "✓" : "✗"}
    </div>
  );

  return (
    <div className="card" style={{ padding: '2rem' }}>
      <h3 style={{ marginBottom: '1.5rem' }}>Synchronized Data Availability</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: '2px solid var(--border-color)', textAlign: 'left' }}>
            <th style={{ padding: '1rem' }}>Category / Row</th>
            {platforms.map(p => <th key={p} style={{ padding: '1rem', textAlign: 'center' }}>{p}</th>)}
          </tr>
        </thead>
        <tbody>
          {Object.keys(details.train || {}).length > 0 && (
            <>
              <tr style={{ backgroundColor: '#f9fafb', fontWeight: 'bold' }}>
                <td colSpan={3} style={{ padding: '0.75rem 1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>GANOMICS (TRAINING)</td>
              </tr>
              {["Real", "Fake"].map(type => (
                <tr key={`train-${type}`} style={{ borderBottom: '1px solid var(--border-color)' }}>
                  <td style={{ padding: '0.75rem 1rem' }}>{type}</td>
                  {platforms.map(p => (
                    <td key={p} style={{ textAlign: 'center' }}>
                      <CheckIcon checked={details.train[p]?.[type]} />
                    </td>
                  ))}
                </tr>
              ))}
            </>
          )}
          <tr style={{ backgroundColor: '#f9fafb', fontWeight: 'bold' }}>
            <td colSpan={3} style={{ padding: '0.75rem 1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>GANOMICS (TESTING)</td>
          </tr>
          {["Real", "Fake"].map(type => (
            <tr key={`test-${type}`} style={{ borderBottom: '1px solid var(--border-color)' }}>
              <td style={{ padding: '0.75rem 1rem' }}>{type}</td>
              {platforms.map(p => (
                <td key={p} style={{ textAlign: 'center' }}>
                  <CheckIcon checked={details.test[p]?.[type]} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
