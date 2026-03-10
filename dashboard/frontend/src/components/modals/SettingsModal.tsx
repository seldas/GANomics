import React from 'react';
import { X, Settings, Plus, Loader2 } from 'lucide-react';
import { FileUploadBox } from '../common/UIComponents';

interface SettingsModalProps {
  show: boolean;
  onClose: () => void;
  newProjectName: string;
  setNewProjectName: (s: string) => void;
  newProjectDescription: string;
  setNewProjectDescription: (s: string) => void;
  agFile: File | null;
  setAgFile: (f: File | null) => void;
  rsFile: File | null;
  setRsFile: (f: File | null) => void;
  labelFile: File | null;
  setLabelFile: (f: File | null) => void;
  isCreating: boolean;
  onCreateProject: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  show, onClose, newProjectName, setNewProjectName, newProjectDescription, setNewProjectDescription,
  agFile, setAgFile, rsFile, setRsFile, labelFile, setLabelFile, isCreating, onCreateProject
}) => {
  if (!show) return null;

  return (
    <div className="modal-overlay" style={{ zIndex: 3000 }}>
      <div className="modal-content" style={{ maxWidth: '600px', width: '90%' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ backgroundColor: '#f1f5f9', padding: '0.5rem', borderRadius: '8px' }}>
              <Settings size={20} style={{ color: '#475569' }} />
            </div>
            <h2 style={{ margin: 0, fontSize: '1.25rem' }}>Project Management</h2>
          </div>
          <button className="chip" onClick={onClose}><X size={18} /></button>
        </div>

        <section className="card" style={{ margin: 0, padding: '1.5rem', border: '1px solid var(--border-color)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
            <Plus size={18} style={{ color: 'var(--primary-color)' }} />
            <h3 style={{ margin: 0, fontSize: '1rem' }}>Create New Project</h3>
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Project Name <span style={{ color: '#ff4d4f' }}>*</span></label>
            <input 
              type="text" 
              placeholder="e.g. MyDataset_2024" 
              style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border-color)', fontSize: '0.9rem' }}
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
            />
          </div>

          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Description</label>
            <textarea 
              placeholder="Briefly describe the dataset and project goals..." 
              style={{ width: '100%', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border-color)', fontSize: '0.9rem', minHeight: '80px', fontFamily: 'inherit' }}
              value={newProjectDescription}
              onChange={(e) => setNewProjectDescription(e.target.value)}
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <FileUploadBox label="Microarray (df_ag)" file={agFile} setFile={setAgFile} required accept=".tsv,.csv,.txt" />
            <FileUploadBox label="RNA-Seq (df_rs)" file={rsFile} setFile={setRsFile} required accept=".tsv,.csv,.txt" />
          </div>

          <FileUploadBox label="Clinical Labels (label.txt)" file={labelFile} setFile={setLabelFile} accept=".tsv,.csv,.txt" />

          <div style={{ marginTop: '2rem', display: 'flex', gap: '1rem' }}>
            <button 
              className={`chip selected ${isCreating ? 'disabled' : ''}`} 
              style={{ flex: 1, padding: '1rem', justifyContent: 'center', gap: '0.75rem' }}
              onClick={onCreateProject}
              disabled={isCreating}
            >
              {isCreating ? <Loader2 size={18} className="animate-spin" /> : <Plus size={18} />}
              {isCreating ? 'Creating Project...' : 'Initialize Project'}
            </button>
            <button className="chip" style={{ padding: '1rem' }} onClick={onClose}>Cancel</button>
          </div>
        </section>
      </div>
    </div>
  );
};
