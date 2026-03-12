import React from 'react';
import { Plus, ArrowLeft } from 'lucide-react';
import type { Project } from '../../types';
import { SIZES, BETAS, LAMBDAS } from '../../constants';

interface NewSessionPanelProps {
  projects: Project[];
  selectedProject: string;
  onSelectProject: (id: string) => void;
  ablationType: 'size' | 'beta' | 'lambda';
  onSetAblationType: (type: 'size' | 'beta' | 'lambda') => void;
  selectedSizes: number[];
  setSelectedSizes: (s: number[]) => void;
  selectedBetas: number[];
  setSelectedBetas: (b: number[]) => void;
  selectedLambdas: number[];
  setSelectedLambdas: (l: number[]) => void;
  selectedRepeats: number;
  setSelectedRepeats: (r: number) => void;
  selectedEpochs: number | 'custom';
  setSelectedEpochs: (e: number | 'custom') => void;
  customEpochs: number;
  setCustomEpochs: (e: number) => void;
  useGpu: boolean;
  setUseGpu: (u: boolean) => void;
  onStartSession: () => void;
  onBack: () => void;
  toggleSelection: (val: any, list: any[], setter: (val: any[]) => void) => void;
}

export const NewSessionPanel: React.FC<NewSessionPanelProps> = ({
  projects, selectedProject, onSelectProject, ablationType, onSetAblationType,
  selectedSizes, setSelectedSizes, selectedBetas, setSelectedBetas, selectedLambdas, setSelectedLambdas,
  selectedRepeats, setSelectedRepeats, selectedEpochs, setSelectedEpochs, customEpochs, setCustomEpochs,
  useGpu, setUseGpu, onStartSession, onBack, toggleSelection
}) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      <header className="header" style={{ marginBottom: '0' }}>
        <div className="header-info">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
            <div style={{ backgroundColor: 'var(--primary-color)', color: 'white', padding: '0.5rem', borderRadius: '8px' }}>
              <Plus size={24} />
            </div>
            <h1 style={{ margin: 0 }}>Start New Experiment</h1>
          </div>
        </div>
        <button className="chip" onClick={onBack}>
          <ArrowLeft size={16} style={{ marginRight: '8px' }} /> Back to Dashboard
        </button>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '2rem', alignItems: 'start' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <section className="card" style={{ padding: '2rem' }}>
            <h3 style={{ fontSize: '1rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>1. PROJECT SELECTION</h3>
            <div className="chip-grid">
              {projects.map(p => (
                <div key={p.id} className={`chip ${selectedProject === p.id ? 'selected' : ''}`} style={{ padding: '0.75rem 1.5rem' }} onClick={() => onSelectProject(p.id)}>
                  {p.name}
                </div>
              ))}
            </div>
          </section>

          <section className="card" style={{ padding: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
              <h3 style={{ fontSize: '1rem', margin: 0 }}>2. ABLATION MATRIX CONFIGURATION</h3>
              <div className="chip-grid" style={{ gap: '0.4rem' }}>
                <button className={`chip ${ablationType === 'size' ? 'selected' : ''}`} onClick={() => onSetAblationType('size')}>Size (N)</button>
                <button className={`chip ${ablationType === 'beta' ? 'selected' : ''}`} onClick={() => onSetAblationType('beta')}>Beta (β)</button>
                <button className={`chip ${ablationType === 'lambda' ? 'selected' : ''}`} onClick={() => onSetAblationType('lambda')}>Lambda (λ)</button>
              </div>
            </div>
            
            <div style={{ backgroundColor: '#f8fafc', padding: '1.5rem', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
              {ablationType === 'size' && (
                <div>
                  <label style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>Sample Size (N)</label>
                  <div className="chip-grid">{SIZES.map(s => (<div key={s} className={`chip ${selectedSizes.includes(s) ? 'selected' : ''}`} style={{ padding: '0.5rem 1rem' }} onClick={() => toggleSelection(s, selectedSizes, setSelectedSizes)}>N={s}</div>))}</div>
                </div>
              )}
              {ablationType === 'beta' && (
                <div>
                  <label style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>Weight (β) - Feedback Alignment</label>
                  <div className="chip-grid">{BETAS.map(b => (<div key={b} className={`chip ${selectedBetas.includes(b) ? 'selected' : ''}`} style={{ padding: '0.5rem 1rem' }} onClick={() => toggleSelection(b, selectedBetas, setSelectedBetas)}>β={b.toFixed(1)}</div>))}</div>
                </div>
              )}
              {ablationType === 'lambda' && (
                <div>
                  <label style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>Weight (λ) - Cycle Reconstruction</label>
                  <div className="chip-grid">{LAMBDAS.map(l => (<div key={l} className={`chip ${selectedLambdas.includes(l) ? 'selected' : ''}`} style={{ padding: '0.5rem 1rem' }} onClick={() => toggleSelection(l, selectedLambdas, setSelectedLambdas)}>λ={l.toFixed(1)}</div>))}</div>
                </div>
              )}
            </div>
          </section>

          <section className="card" style={{ padding: '2rem' }}>
            <h3 style={{ fontSize: '1rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>3. SESSION REPEATS</h3>
            <input type="number" min={1} max={10} value={selectedRepeats} onChange={(e) => setSelectedRepeats(parseInt(e.target.value) || 1)} style={{ width: '100px', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border-color)', textAlign: 'center' }} />
          </section>

          <section className="card" style={{ padding: '2rem' }}>
            <h3 style={{ fontSize: '1rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>4. HARDWARE ACCELERATION</h3>
            <label style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }}>
              <input type="checkbox" checked={useGpu} onChange={(e) => setUseGpu(e.target.checked)} style={{ width: '20px', height: '20px' }} />
              Use GPU acceleration (if available)
            </label>
          </section>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', position: 'sticky', top: '2rem' }}>
          <section className="card" style={{ padding: '1.5rem' }}>
            <h3 style={{ fontSize: '0.9rem', marginBottom: '1.25rem', color: 'var(--text-muted)' }}>5. QUICK CONFIG</h3>
            <div className="form-group" style={{ marginBottom: '1.5rem' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: '600' }}>Training Epochs</label>
              <div className="chip-grid" style={{ marginTop: '0.75rem', gap: '0.5rem' }}>
                {[100, 500, 1000].map(e => (<div key={e} className={`chip ${selectedEpochs === e ? 'selected' : ''}`} style={{ fontSize: '0.8rem' }} onClick={() => setSelectedEpochs(e)}>{e}</div>))}
                <div className={`chip ${selectedEpochs === 'custom' ? 'selected' : ''}`} style={{ fontSize: '0.8rem' }} onClick={() => setSelectedEpochs('custom')}>Custom</div>
              </div>
              {selectedEpochs === 'custom' && (<input type="number" value={customEpochs} onChange={(e) => setCustomEpochs(parseInt(e.target.value) || 250)} style={{ marginTop: '0.75rem', width: '100%', padding: '0.6rem', borderRadius: '8px', border: '1px solid var(--border-color)' }} />)}
            </div>
            <button className="chip selected" style={{ width: '100%', padding: '1rem', fontSize: '1rem' }} onClick={onStartSession}>Launch Experiment</button>
          </section>
        </div>
      </div>
    </div>
  );
};
