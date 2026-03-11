import React from 'react';
import { X, RefreshCw, Info, Download, Loader2, Plus } from 'lucide-react';
import { FileUploadBox } from '../common/UIComponents';

interface SyncExternalModalProps {
  show: boolean;
  onClose: () => void;
  extAgFile: File | null;
  setExtAgFile: (f: File | null) => void;
  extRsFile: File | null;
  setExtRsFile: (f: File | null) => void;
  extDescription: string;
  setExtDescription: (s: string) => void;
  extCustomSuffix: string;
  setExtCustomSuffix: (s: string) => void;
  isRunning: boolean;
  result: any;
  onRunSync: () => void;
  onDownloadGenelist: () => void;
  onSwitchBranch: (id: string) => void;
}

export const SyncExternalModal: React.FC<SyncExternalModalProps> = ({
  show, onClose, extAgFile, setExtAgFile, extRsFile, setExtRsFile,
  extDescription, setExtDescription, extCustomSuffix, setExtCustomSuffix,
  isRunning, result, onRunSync, onDownloadGenelist, onSwitchBranch
}) => {
  if (!show) return null;

  const extCustomId = `ext_${extCustomSuffix}`;

  return (
    <div className="modal-overlay" style={{ zIndex: 3000 }}>
      <div className="modal-content" style={{ maxWidth: '700px', width: '95%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ backgroundColor: '#f0fdf4', padding: '0.5rem', borderRadius: '8px' }}>
              <RefreshCw size={20} style={{ color: '#16a34a' }} />
            </div>
            <h2 style={{ margin: 0, fontSize: '1.25rem' }}>Create New External Test Set</h2>
          </div>
          <button className="chip" onClick={onClose}><X size={18} /></button>
        </div>

        {!result ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 'bold', marginBottom: '0.5rem', color: '#475569' }}>TEST SET NAME</label>
                <div style={{ display: 'flex', alignItems: 'stretch' }}>
                  <div style={{ backgroundColor: '#f1f5f9', border: '1px solid #cbd5e1', borderRight: 'none', borderTopLeftRadius: '8px', borderBottomLeftRadius: '8px', display: 'flex', alignItems: 'center', padding: '0 0.75rem', fontSize: '0.9rem', fontWeight: 'bold', color: '#64748b' }}>
                    ext_
                  </div>
                  <input 
                    type="text" 
                    style={{ flex: 1, padding: '0.6rem', border: '1px solid #cbd5e1', borderTopRightRadius: '8px', borderBottomRightRadius: '8px', fontSize: '0.9rem', outline: 'none' }}
                    value={extCustomSuffix}
                    onChange={(e) => setExtCustomSuffix(e.target.value.replace(/[^a-zA-Z0-9_-]/g, ''))}
                    placeholder="1"
                  />
                </div>
              </div>
              <div>
                <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 'bold', marginBottom: '0.5rem', color: '#475569' }}>DESCRIPTION</label>
                <input 
                  type="text" 
                  className="chip" 
                  style={{ width: '100%', padding: '0.6rem', border: '1px solid #cbd5e1', fontSize: '0.9rem' }}
                  value={extDescription}
                  onChange={(e) => setExtDescription(e.target.value)}
                />
              </div>
            </div>

            <div style={{ padding: '1.25rem', backgroundColor: '#f8fafc', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#475569', marginBottom: '0.75rem', fontWeight: 'bold', fontSize: '0.85rem' }}>
                <Info size={16} /> DATA PREPARATION
              </div>
              <p style={{ fontSize: '0.85rem', margin: 0, color: '#64748b', lineHeight: '1.6' }}>
                Upload microarray or RNA-seq files from an external cohort. 
                Files must use the <b>exact same gene list</b> as the training data.
              </p>
              <div style={{ marginTop: '1rem', display: 'flex', gap: '1rem' }}>
                <button className="chip" style={{ backgroundColor: '#fff' }} onClick={onDownloadGenelist}>
                  <Download size={14} style={{ marginRight: '6px' }} /> Download genelist.tsv
                </button>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <FileUploadBox label="External Microarray (test_ag)" file={extAgFile} setFile={setExtAgFile} accept=".tsv,.txt" />
              <FileUploadBox label="External RNA-Seq (test_rs)" file={extRsFile} setFile={setExtRsFile} accept=".tsv,.txt" />
            </div>

            <button 
              className={`chip selected ${(!extAgFile && !extRsFile) || isRunning ? 'disabled' : ''}`}
              style={{ padding: '1rem', justifyContent: 'center', gap: '0.75rem', fontSize: '1rem' }}
              disabled={(!extAgFile && !extRsFile) || isRunning}
              onClick={onRunSync}
            >
              {isRunning ? <Loader2 size={20} className="animate-spin" /> : <Plus size={20} />}
              {isRunning ? 'Creating Test Set...' : 'Create External Test Set'}
            </button>
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '1rem' }}>
            <div style={{ color: 'var(--success-color)', marginBottom: '1.5rem' }}>
              <div style={{ backgroundColor: 'var(--success-color-bg)', width: '64px', height: '64px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1rem auto' }}>
                <RefreshCw size={32} />
              </div>
              <h3 style={{ margin: 0 }}>External Test Set Created</h3>
              <p>Branch <b>{extCustomId}</b> is now available.</p>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {result.results.map((r: any, i: number) => (
                <div key={i} className="card" style={{ textAlign: 'left', backgroundColor: '#f8fafc', margin: 0 }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 'bold', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                    {r.direction === 'AtoB' ? 'MICROARRAY → RNA-SEQ' : 'RNA-SEQ → MICROARRAY'}
                  </div>
                  <div style={{ fontSize: '0.9rem', fontWeight: '600', fontFamily: 'monospace' }}>{r.output_file}</div>
                </div>
              ))}
            </div>

            <button 
              className="chip selected" 
              style={{ marginTop: '2rem', width: '100%', padding: '1rem' }}
              onClick={() => onSwitchBranch(extCustomId)}
            >
              Switch to New Test Set
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
