export type Project = {
  id: string;
  name: string;
  description?: string;
  genes: number;
  samples: number;
  config_path: string;
  config?: any;
  has_label: boolean;
};

export type LogResponse = {
  run_id: string;
  summary: any;
  structured: any[];
  total_lines: number;
};

export type RunStatus = {
  training: 'running' | 'completed' | 'idle';
  stopped?: boolean;
  current_epoch?: number;
  total_epochs?: number;
  sync: boolean;
  comparative: boolean;
  deg: boolean;
  pathway: boolean;
  pred_model: boolean;
  metadata?: any;
  ext_ids?: string[];
  ext_statuses?: Record<string, any>;
};

export type ResultsStatus = {
  checkpoints: string[];
  logs: string[];
  run_statuses?: Record<string, RunStatus>;
};
