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

export type Dataset = {
  dataset_name: string;
  folder: string;
  config_file: string;
};

export type ExperimentInfo = {
  exp_name: string;
  dataset_name: string;
  result_category: string;
  training_checkpoints_folder: string;
  training_logs: string;
  sync_data_files: any;
  comparative_analysis_results: string;
  deg_analysis_result_folder: string;
  modeling_result_folder: string;
};

export type LogResponse = {
  run_id: string;
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
  algo_details?: {
    comparative: Record<string, boolean>;
    deg: Record<string, boolean>;
    pathway: Record<string, boolean>;
    pred_model: Record<string, boolean>;
  };
  metadata?: any;
  ext_ids?: string[];
  ext_statuses?: Record<string, any>;
};

export type ResultsStatus = {
  logs: string[];
  run_statuses?: Record<string, RunStatus>;
};
