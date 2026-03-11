export const API_BASE = "http://localhost:8832/api";

export const LOSS_METRICS = [
  { key: 'G_A', color: '#007bff', label: 'Gen A' },
  { key: 'G_B', color: '#0056b3', label: 'Gen B' },
  { key: 'D_A', color: '#dc3545', label: 'Disc A' },
  { key: 'D_B', color: '#a71d2a', label: 'Disc B' },
  { key: 'Cycle', color: '#ffc107', label: 'Cycle' },
  { key: 'Feedback', color: '#28a745', label: 'Feedback' },
  { key: 'IDT', color: '#6f42c1', label: 'Identity' },
];

export const SIZES = [10, 20, 50, 100, 200];
export const BETAS = [0.0, 1.0, 5.0, 10.0, 50.0];
export const LAMBDAS = [0.0, 1.0, 5.0, 10.0, 50.0];
