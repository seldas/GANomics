import React from 'react';
import { Info, Table as TableIcon, X, Upload } from 'lucide-react';

export const StatusButton = ({ label, status }: { label: string, status: 'running' | 'completed' | 'idle' | 'unavailable' | boolean }) => {
  let className = "status-badge ";
  let style: React.CSSProperties = {
    fontSize: '0.65rem', padding: '2px 6px', borderRadius: '4px', whiteSpace: 'nowrap',
  };
  if (status === 'running') {
    className += "status-running";
    style = { ...style, background: '#1890ff', color: 'white', animation: 'pulse 1.5s infinite' };
  } else if (status === 'completed' || status === true) {
    className += "status-success";
    style = { ...style, background: '#52c41a', color: 'white' };
  } else if (status === 'unavailable') {
    className += "status-disabled";
    style = { ...style, background: '#f5f5f5', color: '#bfbfbf', border: '1px solid #d9d9d9', opacity: 0.8 };
  } else {
    className += "status-error";
    style = { ...style, border: '1px solid #ff4d4f', color: '#ff4d4f', opacity: 0.6 };
  }
  return <div className={className} style={style}>{label}</div>;
};

export const StepItem = ({ num, label, active, status, onClick, disabled }: any) => (
  <div 
    onClick={!disabled ? onClick : undefined} 
    className={`nav-item ${active ? 'active' : ''} ${disabled ? 'disabled' : ''}`} 
    style={{ 
      display: 'flex', 
      alignItems: 'center', 
      gap: '12px', 
      paddingLeft: '24px', 
      fontSize: '0.85rem', 
      opacity: disabled ? 0.5 : 1,
      cursor: disabled ? 'not-allowed' : 'pointer'
    }}
  >
    <div style={{ 
      width: '20px', 
      height: '20px', 
      borderRadius: '50%', 
      backgroundColor: active ? 'var(--primary-color)' : (status === 'completed' || status === true ? 'var(--success-color)' : '#e5e7eb'), 
      color: 'white', 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center', 
      fontSize: '0.7rem',
      fontWeight: 'bold'
    }}>
      {num}
    </div>
    <span style={{ flex: 1 }}>{label}</span>
  </div>
);

export const FileUploadBox = ({ label, file, setFile, accept, required }: any) => (
  <div style={{ marginBottom: '1.5rem' }}>
    <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 'bold', marginBottom: '0.5rem', color: 'var(--text-main)' }}>
      {label} {required && <span style={{ color: '#ff4d4f' }}>*</span>}
    </label>
    <div 
      className={`file-drop-zone ${file ? 'has-file' : ''}`}
      style={{ 
        border: '2px dashed #e2e8f0', 
        borderRadius: '12px', 
        padding: '1.5rem', 
        textAlign: 'center',
        backgroundColor: file ? '#f0f9ff' : '#f8fafc',
        cursor: 'pointer',
        transition: 'all 0.2s ease',
        position: 'relative'
      }}
      onClick={() => document.getElementById(`file-input-${label}`)?.click()}
    >
      <input 
        id={`file-input-${label}`}
        type="file" 
        accept={accept} 
        style={{ display: 'none' }} 
        onChange={(e) => setFile(e.target.files?.[0] || null)} 
      />
      {file ? (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', color: 'var(--primary-color)' }}>
          <TableIcon size={20} />
          <div style={{ textAlign: 'left' }}>
            <div style={{ fontWeight: '600', fontSize: '0.9rem' }}>{file.name}</div>
            <div style={{ fontSize: '0.75rem', opacity: 0.8 }}>{(file.size / 1024 / 1024).toFixed(2)} MB</div>
          </div>
          <X 
            size={18} 
            style={{ marginLeft: 'auto', color: '#64748b' }} 
            onClick={(e) => { e.stopPropagation(); setFile(null); }} 
          />
        </div>
      ) : (
        <div style={{ color: '#64748b' }}>
          <Upload size={24} style={{ marginBottom: '0.5rem', opacity: 0.5 }} />
          <div style={{ fontSize: '0.85rem' }}>Click or drag to upload {label}</div>
          <div style={{ fontSize: '0.7rem', opacity: 0.7, marginTop: '0.25rem' }}>Supports .tsv, .csv, .txt</div>
        </div>
      )}
    </div>
  </div>
);

export const MetaPanel = ({ description, samples, genes, note }: any) => (
  <section className="card" style={{ marginTop: '2rem', padding: '1.5rem', backgroundColor: '#f8fafc', border: '1px solid #e2e8f0' }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem', color: '#475569' }}>
      <Info size={18} />
      <h4 style={{ margin: 0, fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Test Dataset Details</h4>
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem' }}>
      <div>
        <div style={{ fontSize: '0.7rem', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', marginBottom: '0.25rem' }}>Context</div>
        <div style={{ fontSize: '0.85rem', color: '#475569', fontWeight: '500' }}>{description || 'No description available.'}</div>
      </div>
      <div style={{ display: 'flex', gap: '2rem' }}>
        <div>
          <div style={{ fontSize: '0.7rem', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', marginBottom: '0.25rem' }}>Samples</div>
          <div style={{ fontSize: '1rem', color: 'var(--primary-color)', fontWeight: 'bold' }}>{samples || 0}</div>
        </div>
        <div>
          <div style={{ fontSize: '0.7rem', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', marginBottom: '0.25rem' }}>Genes</div>
          <div style={{ fontSize: '1rem', color: 'var(--primary-color)', fontWeight: 'bold' }}>{genes || 0}</div>
        </div>
      </div>
      {note && (
        <div>
          <div style={{ fontSize: '0.7rem', fontWeight: 'bold', color: '#94a3b8', textTransform: 'uppercase', marginBottom: '0.25rem' }}>Note</div>
          <div style={{ fontSize: '0.8rem', color: '#64748b', fontStyle: 'italic' }}>{note}</div>
        </div>
      )}
    </div>
  </section>
);
