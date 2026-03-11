import os
import yaml
import subprocess
import re
import time
import sys
import signal
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, BackgroundTasks, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import psutil
import json
import numpy as np

app = FastAPI(title="GANomics API")

# Global process tracker: { run_id: { pid: int, command: List[str], type: 'training' | 'step', step?: int } }
running_processes: Dict[str, dict] = {}

def kill_proc_tree(pid, sig=signal.SIGTERM, include_parent=True,
                   timeout=None, on_terminate=None):
    """Kill a process tree (including children) with psutil."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.send_signal(sig)
        if include_parent:
            parent.send_signal(sig)
        gone, alive = psutil.wait_procs(children + ([parent] if include_parent else []),
                                        timeout=timeout, on_terminate=on_terminate)
        return gone, alive
    except psutil.NoSuchProcess:
        return [], []

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Path Definitions ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "..", ".."))

DATASET_DIR = os.path.join(BACKEND_DIR, "dataset")
RESULTS_DIR = os.path.join(BACKEND_DIR, "results")
SCRIPTS_DIR = os.path.join(BACKEND_DIR, "scripts")
TEMP_DIR = os.path.join(BACKEND_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

TRAINING_DIR = os.path.join(RESULTS_DIR, "1_Training")
LOGS_DIR = os.path.join(TRAINING_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(TRAINING_DIR, "checkpoints")

SYNC_DATA_DIR = os.path.join(RESULTS_DIR, "2_SyncData")
COMPARATIVE_DIR = os.path.join(RESULTS_DIR, "3_ComparativeAnalysis")
BIOMARKERS_DIR = os.path.join(RESULTS_DIR, "4_Biomarkers")
FIGURES_DIR = os.path.join(RESULTS_DIR, "5_Figures")

class ProjectInfo(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    genes: int
    samples: int
    config_path: str
    has_label: bool
    config: Optional[dict] = None

class TrainRequest(BaseModel):
    config_path: str
    sizes: List[int]
    betas: List[float]
    lambdas: List[float]
    repeats: int = 1
    epochs: Optional[int] = 250
    lr: Optional[float] = 0.0002
    use_gpu: bool = True

def get_available_gpus():
    try:
        import torch
        if torch.cuda.is_available():
            return ",".join([str(i) for i in range(torch.cuda.device_count())])
    except: pass
    return None

def get_project_stats(config_path):
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        path_a = cfg['dataset']['path_A']
        actual_path = path_a if os.path.isabs(path_a) else os.path.abspath(os.path.join(BACKEND_DIR, path_a))
        
        if os.path.exists(actual_path):
            df_headers = pd.read_csv(actual_path, index_col=0, nrows=0, sep=None, engine='python')
            genes_count = len(df_headers.columns)
            with open(actual_path, 'rb') as f:
                samples_count = sum(1 for _ in f) - 1
            return genes_count, samples_count
    except: pass
    return 0, 0

def parse_log_line(line: str):
    line = line.strip()
    if not line.startswith("("): return None
    header_match = re.search(r"\((.*?)\)", line)
    if not header_match: return None
    header = header_match.group(1)
    metrics_str = line[header_match.end():].strip()
    data = {}
    for part in header.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            try: data[k.strip()] = float(v.strip())
            except: data[k.strip()] = v.strip()
    metric_pairs = re.findall(r"(\w+):\s*([\d\.\-]+)", metrics_str)
    for k, v in metric_pairs:
        data[k] = float(v)
    return data

@app.get("/api/projects", response_model=List[ProjectInfo])
async def list_projects():
    projects = []
    if not os.path.exists(DATASET_DIR): return []
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith("_config.yaml"):
                full_path = os.path.join(root, file)
                pid = os.path.basename(root)
                has_label = os.path.exists(os.path.join(root, "label.txt"))
                config_data = {}
                try:
                    with open(full_path, 'r') as f: config_data = yaml.safe_load(f)
                except: pass
                metadata = config_data.get('metadata', {})
                f_genes, f_samples = get_project_stats(full_path)
                projects.append(ProjectInfo(
                    id=pid, name=metadata.get('name', pid), 
                    description=metadata.get('description', ""),
                    genes=metadata.get('genes', f_genes), 
                    samples=metadata.get('samples', f_samples), 
                    config_path=full_path, has_label=has_label, config=config_data
                ))
    return projects

@app.get("/api/projects/{project_id}/samples/download")
async def download_samples(project_id: str):
    proj_dir = os.path.join(DATASET_DIR, project_id)
    samples_path = os.path.join(proj_dir, "samples.tsv")
    if not os.path.exists(samples_path): raise HTTPException(status_code=404)
    return FileResponse(samples_path, filename=f"{project_id}_samples.tsv")

@app.post("/api/projects/{project_id}/labels/upload")
async def upload_labels(project_id: str, file: UploadFile = File(...)):
    proj_dir = os.path.join(DATASET_DIR, project_id)
    final_path = os.path.join(proj_dir, "label.txt")
    with open(final_path, "wb") as f: f.write(await file.read())
    return {"message": "label.txt uploaded successfully"}

@app.post("/api/projects/create")
async def create_project(
    project_name: str = Form(...), description: str = Form(""),
    df_ag: UploadFile = File(...), df_rs: UploadFile = File(...),
    label: Optional[UploadFile] = File(None)
):
    proj_dir = os.path.join(DATASET_DIR, project_name)
    os.makedirs(proj_dir, exist_ok=True)
    try:
        ag_path = os.path.join(proj_dir, "df_ag.tsv")
        rs_path = os.path.join(proj_dir, "df_rs.tsv")
        with open(ag_path, "wb") as f: f.write(await df_ag.read())
        with open(rs_path, "wb") as f: f.write(await df_rs.read())
        if label:
            with open(os.path.join(proj_dir, "label.txt"), "wb") as f: f.write(await label.read())
        
        df_peek = pd.read_csv(ag_path, sep='\t', index_col=0, nrows=0)
        genes_count = len(df_peek.columns)
        with open(ag_path, 'r') as f: samples_count = sum(1 for _ in f) - 1
            
        config = {
            'metadata': {'name': project_name, 'description': description, 'genes': genes_count, 'samples': samples_count},
            'model': {'input_nc': genes_count, 'output_nc': genes_count, 'lambda_A': 10.0, 'lambda_B': 10.0, 'lambda_feedback': 10.0, 'lambda_idt': 0.5, 'gan_mode': 'lsgan'},
            'optimizer': {'lr': 0.0002, 'beta1': 0.5, 'beta2': 0.999},
            'train': {'n_epochs': 500, 'n_epochs_decay': 50, 'batch_size': 1, 'device': 'cuda:0', 'seed': 42, 'print_freq': 10, 'save_epoch_freq': 10, 'explosion_factor': 3.0},
            'dataset': {'path_A': f'dataset/{project_name}/df_ag.tsv', 'path_B': f'dataset/{project_name}/df_rs.tsv', 'max_samples': samples_count, 'force_index_mapping': True},
            'output': {'checkpoints_dir': 'results/1_Training/checkpoints', 'name': f'{project_name}_GANomics', 'logs_dir': 'results/1_Training/logs'},
            'default_ablation': {'size': 50, 'beta': 10.0, 'lambda': 10.0}
        }
        with open(os.path.join(proj_dir, f"{project_name.lower()}_config.yaml"), 'w') as f: yaml.dump(config, f)
        
        df_ag_full = pd.read_csv(ag_path, sep='\t', index_col=0)
        pd.DataFrame({'sample_id': df_ag_full.index.tolist()}).to_csv(os.path.join(proj_dir, "samples.tsv"), sep='\t', index=False)
        pd.DataFrame({'gene_id': df_ag_full.columns.tolist()}).to_csv(os.path.join(proj_dir, "genelist.tsv"), sep='\t', index=False)
        return {"message": "Project created successfully"}
    except Exception as e:
        if os.path.exists(proj_dir): import shutil; shutil.rmtree(proj_dir)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/runs/{run_id}/inference")
async def run_external_inference(
    run_id: str, direction: str = Form(...), ext_id: str = Form(...),
    description: str = Form("External Testing dataset"), file: UploadFile = File(...)
):
    project_id = run_id.split('_')[0]
    ext_dir = os.path.join(DATASET_DIR, project_id, ext_id)
    os.makedirs(ext_dir, exist_ok=True)
    abs_input_path = os.path.join(ext_dir, "test_ag.tsv" if direction == 'AtoB' else "test_rs.tsv")
    with open(abs_input_path, "wb") as f: f.write(await file.read())
    try:
        df_peek = pd.read_csv(abs_input_path, sep='\t', index_col=0)
        with open(os.path.join(ext_dir, "metadata.json"), "w") as f:
            json.dump({"id": ext_id, "description": description, "samples": len(df_peek), "genes": len(df_peek.columns)}, f)
    except: pass
    output_dir = os.path.join(RESULTS_DIR, "2_SyncData", run_id, ext_id)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "translated_rs.tsv" if direction == 'AtoB' else "translated_ag.tsv")
    cmd = [sys.executable, os.path.join(BACKEND_DIR, "scripts", "inference.py"), "--run_id", run_id, "--input", abs_input_path, "--direction", direction, "--output", output_path, "--device", "cpu"]
    subprocess.run(cmd, check=True)
    return {"message": "Inference completed", "ext_id": ext_id}

@app.post("/api/runs/{run_id}/sync_external")
async def sync_external(
    run_id: str, test_ag: Optional[UploadFile] = File(None), test_rs: Optional[UploadFile] = File(None),
    ext_id: str = Form(...), description: str = Form("External Testing dataset")
):
    project_id = run_id.split('_')[0]
    ext_dir = os.path.join(DATASET_DIR, project_id, ext_id)
    os.makedirs(ext_dir, exist_ok=True)
    samples_count, genes_count = 0, 0
    async def save_file(file: UploadFile, filename: str):
        nonlocal samples_count, genes_count
        abs_path = os.path.join(ext_dir, filename)
        content = await file.read()
        with open(abs_path, "wb") as f: f.write(content)
        if samples_count == 0:
            try:
                df = pd.read_csv(abs_path, sep='\t', index_col=0); samples_count, genes_count = len(df), len(df.columns)
            except: pass
    if test_ag: await save_file(test_ag, "test_ag.tsv")
    if test_rs: await save_file(test_rs, "test_rs.tsv")
    with open(os.path.join(ext_dir, "metadata.json"), "w") as f:
        json.dump({"id": ext_id, "description": description, "samples": samples_count, "genes": genes_count}, f)
    return {"message": "Files uploaded successfully", "ext_id": ext_id}

@app.get("/api/results")
async def get_results_status():
    logs = [l.replace("_log.txt", "") for l in os.listdir(LOGS_DIR) if l.endswith("_log.txt")]
    run_statuses = {}
    now = time.time()
    for run_id in logs:
        project_id = run_id.split('_')[0]
        sync_path = os.path.join(SYNC_DATA_DIR, run_id, "test", "microarray_fake.csv")
        log_path = os.path.join(LOGS_DIR, f"{run_id}_log.txt")
        checkpoint_latest = os.path.join(CHECKPOINTS_DIR, run_id, "net_latest.pth")
        is_running = os.path.exists(log_path) and (now - os.path.getmtime(log_path) < 60)
        
        ext_ids, ext_statuses = [], {}
        proj_ds_root = os.path.join(DATASET_DIR, project_id)
        if os.path.exists(proj_ds_root):
            ext_ids = [d for d in os.listdir(proj_ds_root) if d.startswith("ext_") and os.path.isdir(os.path.join(proj_ds_root, d))]
            for eid in ext_ids:
                e_sync_dir = os.path.join(SYNC_DATA_DIR, run_id, eid)
                meta = {"description": "External Testing dataset", "samples": 0, "genes": 0}
                try:
                    with open(os.path.join(proj_ds_root, eid, "metadata.json"), 'r') as f: meta = json.load(f)
                except: pass
                ext_statuses[eid] = {
                    "metadata": meta, "sync": os.path.exists(os.path.join(e_sync_dir, "microarray_fake.csv")) or os.path.exists(os.path.join(e_sync_dir, "translated_ag.tsv")),
                    "comparative": os.path.exists(os.path.join(e_sync_dir, "Test_performance.csv")),
                    "deg": os.path.exists(os.path.join(e_sync_dir, "DEG", "Jaccard_Curve_GANomics.csv")),
                    "pathway": os.path.exists(os.path.join(e_sync_dir, "Pathway", "Pathway_Concordance_GANomics.csv")),
                    "pred_model": os.path.exists(os.path.join(e_sync_dir, "Prediction", "Classifier_Performance_GANomics.csv")),
                }
        
        internal_meta = {"description": "Standard Internal Test Set", "note": "Original Test Set from Unseen data points", "samples": 0, "genes": 0}
        if os.path.exists(sync_path):
            try:
                df = pd.read_csv(sync_path, index_col=0, nrows=0); internal_meta["genes"] = len(df.columns)
                with open(os.path.join(SYNC_DATA_DIR, run_id, "test", "microarray_real.csv"), 'rb') as f: internal_meta["samples"] = sum(1 for _ in f) - 1
            except: pass

        run_statuses[run_id] = {
            "training": "running" if is_running else ("completed" if os.path.exists(checkpoint_latest) else "idle"),
            "sync": os.path.exists(sync_path), "comparative": os.path.exists(os.path.join(COMPARATIVE_DIR, run_id, "Test_performance.csv")),
            "deg": os.path.exists(os.path.join(BIOMARKERS_DIR, "DEG", run_id, "Jaccard_Curve_GANomics.csv")),
            "pathway": os.path.exists(os.path.join(BIOMARKERS_DIR, "Pathway", run_id, "Pathway_Concordance_GANomics.csv")),
            "pred_model": os.path.exists(os.path.join(BIOMARKERS_DIR, "Prediction", run_id, "Classifier_Performance_GANomics.csv")),
            "metadata": internal_meta, "ext_ids": ext_ids, "ext_statuses": ext_statuses, "sync_ext": len(ext_ids) > 0,
        }
    return {"checkpoints": os.listdir(CHECKPOINTS_DIR) if os.path.exists(CHECKPOINTS_DIR) else [], "logs": logs, "run_statuses": run_statuses}

@app.post("/api/train")
async def start_training(req: TrainRequest):
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "run_ablation.py"), "--config", req.config_path, "--repeats", str(req.repeats)]
    if req.sizes: cmd += ["--sizes"] + [str(s) for s in req.sizes]
    if req.betas: cmd += ["--betas"] + [str(b) for b in req.betas]
    if req.lambdas: cmd += ["--lambdas"] + [str(l) for l in req.lambdas]
    gpu_ids = get_available_gpus()
    if req.use_gpu and gpu_ids: cmd += ["--gpu_ids", gpu_ids]
    env = os.environ.copy(); env["PYTHONPATH"] = f"{BACKEND_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"
    process = subprocess.Popen(cmd, cwd=BACKEND_DIR, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
    running_processes['training_session'] = {"pid": process.pid, "cmd": cmd, "env": env, "cwd": BACKEND_DIR, "type": "training"}
    return {"message": "Training started", "pid": process.pid}

@app.post("/api/runs/{run_id}/stop")
async def stop_task(run_id: str):
    for proc in psutil.process_iter(['pid', 'cmdline']):
        if run_id in str(proc.info['cmdline']): kill_proc_tree(proc.info['pid'])
    return {"message": "Stopped"}

@app.post("/api/runs/{run_id}/restart")
async def restart_task(run_id: str):
    raise HTTPException(status_code=501)

@app.post("/api/runs/{run_id}/run_step")
async def run_analysis_step(run_id: str, step: int, config_path: Optional[str] = None, ext_id: Optional[str] = None):
    scripts = {2: "test_sync.py", 3: "comparative_analysis.py", 4: "biomarker.py"}
    script_path = os.path.join(SCRIPTS_DIR, scripts[step])
    cmd = [sys.executable, script_path, "--run_id", run_id]
    if ext_id: cmd += ["--ext_id", ext_id]
    if step == 2: cmd += ["--config", config_path]
    if step == 4:
        label_path = os.path.join(DATASET_DIR, run_id.split('_')[0], "label.txt")
        if os.path.exists(label_path): cmd += ["--labels", label_path]
    env = os.environ.copy(); env["PYTHONPATH"] = f"{BACKEND_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"
    process = subprocess.Popen(cmd, cwd=BACKEND_DIR, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
    return {"message": "Started", "pid": process.pid}

@app.get("/api/runs/{run_id}/logs")
async def stream_run_logs(run_id: str):
    log_path = os.path.join(LOGS_DIR, f"{run_id}_log.txt")
    if not os.path.exists(log_path): raise HTTPException(status_code=404)
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines(); structured = [p for p in [parse_log_line(l) for l in lines] if p]
    return {"run_id": run_id, "structured": structured, "total_lines": len(lines)}

@app.get("/api/runs/{run_id}/comparative")
async def get_run_comparative_metrics(run_id: str, ext_id: Optional[str] = None):
    path = os.path.join(SYNC_DATA_DIR if ext_id else COMPARATIVE_DIR, run_id, ext_id if ext_id else "", "Test_performance.csv")
    if not os.path.exists(path): raise HTTPException(status_code=404)
    return pd.read_csv(path).replace({np.nan: None}).to_dict(orient="records")

@app.get("/api/runs/{run_id}/sync")
async def get_run_sync_status(run_id: str, ext_id: Optional[str] = None):
    sync_dir = os.path.join(SYNC_DATA_DIR, run_id, ext_id if ext_id else "test")
    details = {"test": {"Microarray": {"Real": False, "Fake": False}, "RNA-Seq": {"Real": False, "Fake": False}}}
    if os.path.exists(sync_dir):
        details["test"]["Microarray"]["Real"] = os.path.exists(os.path.join(sync_dir, "microarray_real.csv"))
        details["test"]["Microarray"]["Fake"] = os.path.exists(os.path.join(sync_dir, "microarray_fake.csv"))
        details["test"]["RNA-Seq"]["Real"] = os.path.exists(os.path.join(sync_dir, "rnaseq_real.csv"))
        details["test"]["RNA-Seq"]["Fake"] = os.path.exists(os.path.join(sync_dir, "rnaseq_fake.csv"))
    return {"exists": os.path.exists(sync_dir), "details": details}

@app.get("/api/runs/{run_id}/sync/download")
async def download_sync_file(run_id: str, filename: str, ext_id: Optional[str] = None):
    path = os.path.join(SYNC_DATA_DIR, run_id, ext_id if ext_id else "test", filename)
    if not os.path.exists(path): raise HTTPException(status_code=404)
    return FileResponse(path, filename=f"{run_id}_{filename}")

@app.get("/api/runs/{run_id}/tsne")
async def get_run_tsne_coords(run_id: str, ext_id: Optional[str] = None):
    path = os.path.join(SYNC_DATA_DIR, run_id, ext_id if ext_id else "test", "tsne_coords.csv")
    if not os.path.exists(path): raise HTTPException(status_code=404)
    return pd.read_csv(path).replace({np.nan: None}).to_dict(orient="records")

@app.get("/api/runs/{run_id}/deg")
async def get_run_deg_metrics(run_id: str, ext_id: Optional[str] = None):
    deg_dir = os.path.join(SYNC_DATA_DIR if ext_id else BIOMARKERS_DIR, run_id if not ext_id else os.path.join(run_id, ext_id), "DEG")
    if not os.path.exists(deg_dir): raise HTTPException(status_code=404)
    res = {}
    for f in os.listdir(deg_dir):
        if f.startswith("Jaccard_Curve_"): res[f[14:-4]] = pd.read_csv(os.path.join(deg_dir, f)).replace({np.nan: None}).to_dict(orient="records")
    return res

@app.get("/api/runs/{run_id}/prediction")
async def get_run_prediction_metrics(run_id: str, ext_id: Optional[str] = None):
    pred_dir = os.path.join(SYNC_DATA_DIR if ext_id else BIOMARKERS_DIR, run_id if not ext_id else os.path.join(run_id, ext_id), "Prediction")
    if not os.path.exists(pred_dir): raise HTTPException(status_code=404)
    res = {}
    for f in os.listdir(pred_dir):
        if f.startswith("Classifier_Performance_"): res[f[23:-4]] = pd.read_csv(os.path.join(pred_dir, f)).replace({np.nan: None}).to_dict(orient="records")
    return res

@app.get("/api/runs/{run_id}/pathway")
async def get_run_pathway_metrics(run_id: str, ext_id: Optional[str] = None):
    path_dir = os.path.join(SYNC_DATA_DIR if ext_id else BIOMARKERS_DIR, run_id if not ext_id else os.path.join(run_id, ext_id), "Pathway")
    if not os.path.exists(path_dir): raise HTTPException(status_code=404)
    results = {}
    for f in os.listdir(path_dir):
        parts = f[:-4].split("_")
        if f.startswith("Pathway_Concordance_"):
            lib = "_".join(parts[3:])
            if lib not in results: results[lib] = {"concordance": {}, "details": {}}
            results[lib]["concordance"][parts[2]] = pd.read_csv(os.path.join(path_dir, f)).replace({np.nan: None}).to_dict(orient="records")[0]
        elif f.startswith("Pathway_Details_"):
            lib = "_".join(parts[3:])
            if lib not in results: results[lib] = {"concordance": {}, "details": {}}
            results[lib]["details"][parts[2]] = pd.read_csv(os.path.join(path_dir, f)).replace({np.nan: None}).to_dict(orient="records")
    return results

@app.get("/api/projects/{project_id}/ablation")
async def get_project_ablation_metrics(project_id: str):
    res = []
    if os.path.exists(COMPARATIVE_DIR):
        for rid in os.listdir(COMPARATIVE_DIR):
            if rid.startswith(project_id) and os.path.exists(os.path.join(COMPARATIVE_DIR, rid, "Test_performance.csv")):
                try:
                    df = pd.read_csv(os.path.join(COMPARATIVE_DIR, rid, "Test_performance.csv"))
                    row = df[df['Algorithm'] == 'GANomics'].iloc[0].to_dict(); row['run_id'] = rid; res.append(row)
                except: pass
    return res

@app.get("/api/projects/{project_id}/ablation_logs")
async def get_project_ablation_logs(project_id: str, category: str):
    res = []
    if os.path.exists(LOGS_DIR):
        for f in os.listdir(LOGS_DIR):
            if f.startswith(project_id) and f.endswith("_log.txt"):
                rid = f[:-8]
                if (category == 'architecture' and "Architecture" in rid) or (category == 'size' and "Size" in rid and "Architecture" not in rid) or (category == 'sensitivity' and "Sensitivity" in rid):
                    try:
                        with open(os.path.join(LOGS_DIR, f)) as lf:
                            struct = [p for p in [parse_log_line(l) for l in lf.readlines()] if p]
                            if len(struct) >= 2: res.append({"run_id": rid, "first": struct[0], "last": struct[-1]})
                    except: pass
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8832)
