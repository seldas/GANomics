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
from pydantic import BaseModel
import pandas as pd
import psutil
import json

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
# ROOT_DIR is the original root (for any lingering root-relative needs)
ROOT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "..", ".."))

# All data and results are now nested inside dashboard/backend/
DATASET_DIR = os.path.join(BACKEND_DIR, "dataset")
RESULTS_DIR = os.path.join(BACKEND_DIR, "results")
SCRIPTS_DIR = os.path.join(BACKEND_DIR, "scripts")
TEMP_DIR = os.path.join(BACKEND_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Redesigned Structure
TRAINING_DIR = os.path.join(RESULTS_DIR, "1_Training")
LOGS_DIR = os.path.join(TRAINING_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(TRAINING_DIR, "checkpoints")

SYNC_DATA_DIR = os.path.join(RESULTS_DIR, "2_SyncData")
COMPARATIVE_DIR = os.path.join(RESULTS_DIR, "3_ComparativeAnalysis")
BIOMARKERS_DIR = os.path.join(RESULTS_DIR, "4_Biomarkers")
FIGURES_DIR = os.path.join(RESULTS_DIR, "5_Figures")

print(f"BACKEND_DIR: {BACKEND_DIR}")
print(f"RESULTS_DIR: {RESULTS_DIR}")
print(f"LOGS_DIR: {LOGS_DIR}")
print(f"LOGS EXISTS: {os.path.exists(LOGS_DIR)}")

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
        
        # Resolve path relative to backend or root
        actual_path = path_a
        if not os.path.isabs(path_a):
            # Try root relative
            alt_path = os.path.abspath(os.path.join(ROOT_DIR, path_a))
            if os.path.exists(alt_path): 
                actual_path = alt_path
            else:
                # Try backend relative
                alt_path = os.path.abspath(os.path.join(BACKEND_DIR, path_a))
                if os.path.exists(alt_path):
                    actual_path = alt_path
        
        if os.path.exists(actual_path):
            # Strict Orientation: Columns are Genes, Rows are Samples
            df_headers = pd.read_csv(actual_path, index_col=0, nrows=0, sep=None, engine='python')
            genes_count = len(df_headers.columns)
            
            with open(actual_path, 'rb') as f:
                samples_count = sum(1 for _ in f) - 1 # Subtract header
            
            return genes_count, samples_count
    except Exception as e:
        print(f"Error getting stats for {config_path}: {e}")
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
                
                # Check for label.txt
                has_label = os.path.exists(os.path.join(root, "label.txt"))
                
                config_data = {}
                try:
                    with open(full_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                except: pass
                
                metadata = config_data.get('metadata', {})
                name = metadata.get('name', pid)
                description = metadata.get('description', "")
                
                # Use metadata stats or fallback to file analysis
                genes = metadata.get('genes')
                samples = metadata.get('samples')
                
                if genes is None or samples is None:
                    f_genes, f_samples = get_project_stats(full_path)
                    genes = genes if genes is not None else f_genes
                    samples = samples if samples is not None else f_samples

                projects.append(ProjectInfo(
                    id=pid, 
                    name=name, 
                    description=description,
                    genes=genes, 
                    samples=samples, 
                    config_path=full_path, 
                    has_label=has_label, 
                    config=config_data
                ))
    return projects

from fastapi import UploadFile, File
from fastapi.responses import FileResponse

@app.get("/api/projects/{project_id}/samples/download")
async def download_samples(project_id: str):
    # Find project directory
    proj_dir = os.path.join(DATASET_DIR, project_id)
    # Handle case-sensitivity/heuristic
    if not os.path.exists(proj_dir):
        for d in os.listdir(DATASET_DIR):
            if d.upper() == project_id.upper():
                proj_dir = os.path.join(DATASET_DIR, d)
                break
    
    samples_path = os.path.join(proj_dir, "samples.tsv")
    if not os.path.exists(samples_path):
        raise HTTPException(status_code=404, detail="samples.tsv not found for this project")
    
    return FileResponse(samples_path, filename=f"{project_id}_samples.tsv")

@app.post("/api/projects/{project_id}/labels/upload")
async def upload_labels(project_id: str, file: UploadFile = File(...)):
    # Find project directory
    proj_dir = os.path.join(DATASET_DIR, project_id)
    if not os.path.exists(proj_dir):
        for d in os.listdir(DATASET_DIR):
            if d.upper() == project_id.upper():
                proj_dir = os.path.join(DATASET_DIR, d)
                break
    
    if not os.path.exists(proj_dir):
        raise HTTPException(status_code=404, detail="Project directory not found")

    # Save to temporary location first for validation
    temp_path = os.path.join(proj_dir, "label_temp.txt")
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    try:
        # Validation
        # 1. Check if it's readable and has correct columns
        df_uploaded = pd.read_csv(temp_path)
        if 'sample_id' not in df_uploaded.columns or 'label' not in df_uploaded.columns:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="label.txt must contain 'sample_id' and 'label' columns")
        
        # 2. Check against samples.tsv
        samples_path = os.path.join(proj_dir, "samples.tsv")
        if os.path.exists(samples_path):
            df_samples = pd.read_csv(samples_path, sep='\t')
            valid_ids = set(df_samples['sample_id'])
            uploaded_ids = set(df_uploaded['sample_id'])
            
            invalid_ids = uploaded_ids - valid_ids
            if len(invalid_ids) > 0:
                os.remove(temp_path)
                raise HTTPException(status_code=400, detail=f"Uploaded labels contain invalid sample IDs: {list(invalid_ids)[:5]}...")
        
        # Validation passed, rename to label.txt
        final_path = os.path.join(proj_dir, "label.txt")
        if os.path.exists(final_path): os.remove(final_path)
        os.rename(temp_path, final_path)
        
        return {"message": "label.txt uploaded and validated successfully"}
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects/create")
async def create_project(
    project_name: str = Form(...),
    description: str = Form(""),
    df_ag: UploadFile = File(...),
    df_rs: UploadFile = File(...),
    label: Optional[UploadFile] = File(None)
):
    # 1. Validate Project Name
    if not re.match(r'^[a-zA-Z0-9_-]+$', project_name):
        raise HTTPException(status_code=400, detail="Invalid project name. Use only alphanumeric characters, underscores, and hyphens.")
    
    proj_dir = os.path.join(DATASET_DIR, project_name)
    if os.path.exists(proj_dir):
        raise HTTPException(status_code=400, detail="Project already exists.")
    
    os.makedirs(proj_dir, exist_ok=True)
    
    try:
        # 2. Save Uploaded Files
        ag_path = os.path.join(proj_dir, "df_ag.tsv")
        rs_path = os.path.join(proj_dir, "df_rs.tsv")
        
        with open(ag_path, "wb") as f: f.write(await df_ag.read())
        with open(rs_path, "wb") as f: f.write(await df_rs.read())
        
        if label:
            label_path = os.path.join(proj_dir, "label.txt")
            with open(label_path, "wb") as f: f.write(await label.read())
            
        # 3. Analyze Data to get Genes and Samples
        # Strictly: Columns = Genes, Rows = Samples
        df_ag_peek = pd.read_csv(ag_path, sep='\t', index_col=0, nrows=0)
        genes_count = len(df_ag_peek.columns)
        
        with open(ag_path, 'r') as f:
            samples_count = sum(1 for _ in f) - 1
            
        # 4. Generate YAML Config
        config = {
            'metadata': {
                'name': project_name,
                'description': description,
                'genes': genes_count,
                'samples': samples_count,
                'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
            },
            'model': {
                'input_nc': genes_count,
                'output_nc': genes_count,
                'lambda_A': 10.0,
                'lambda_B': 10.0,
                'lambda_feedback': 10.0,
                'lambda_idt': 0.5,
                'gan_mode': 'lsgan'
            },
            'optimizer': {
                'lr': 0.0002,
                'beta1': 0.5,
                'beta2': 0.999
            },
            'train': {
                'n_epochs': 500,
                'n_epochs_decay': 50,
                'batch_size': 1,
                'device': 'cuda:0',
                'seed': 42,
                'print_freq': 10,
                'save_epoch_freq': 10,
                'explosion_factor': 3.0
            },
            'dataset': {
                'path_A': f'dataset/{project_name}/df_ag.tsv',
                'path_B': f'dataset/{project_name}/df_rs.tsv',
                'max_samples': samples_count,
                'force_index_mapping': True
            },
            'output': {
                'checkpoints_dir': 'results/1_Training/checkpoints',
                'name': f'{project_name}_GANomics',
                'logs_dir': 'results/1_Training/logs'
            },
            'default_ablation': {
                'size': 50,
                'beta': 10.0,
                'lambda': 10.0
            }
        }
        
        config_path = os.path.join(proj_dir, f"{project_name.lower()}_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        # 5. Generate samples.tsv and genelist.tsv
        df_ag_full = pd.read_csv(ag_path, sep='\t', index_col=0)
        samples = df_ag_full.index.tolist()
        genes = df_ag_full.columns.tolist()
        
        samples_path = os.path.join(proj_dir, "samples.tsv")
        pd.DataFrame({'sample_id': samples}).to_csv(samples_path, sep='\t', index=False)
        
        genes_path = os.path.join(proj_dir, "genelist.tsv")
        pd.DataFrame({'gene_id': genes}).to_csv(genes_path, sep='\t', index=False)
        
        return {"message": f"Project {project_name} created successfully", "genes": genes_count, "samples": samples_count}
        
    except Exception as e:
        # Cleanup on failure
        if os.path.exists(proj_dir):
            import shutil
            shutil.rmtree(proj_dir)
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

@app.get("/api/runs/{run_id}/genes/template")
async def get_gene_template(run_id: str):
    checkpoint_dir = os.path.join(RESULTS_DIR, "1_Training", "checkpoints", run_id)
    genes_path = os.path.join(checkpoint_dir, "genes.txt")
    
    # If genes.txt doesn't exist (older run), try to find it from the original project
    if not os.path.exists(genes_path):
        project_id = run_id.split('_')[0]
        proj_dir = os.path.join(DATASET_DIR, project_id)
        ag_file = os.path.join(proj_dir, "df_ag.tsv")
        if os.path.exists(ag_file):
            try:
                df = pd.read_csv(ag_file, sep='\t', index_col=0, nrows=0)
                genes = df.columns.tolist()
                os.makedirs(checkpoint_dir, exist_ok=True)
                with open(genes_path, 'w') as f:
                    f.write("\n".join(genes))
            except:
                raise HTTPException(status_code=404, detail="Could not retrieve gene template for this run.")
        else:
            raise HTTPException(status_code=404, detail="Gene template not found.")

    return FileResponse(
        genes_path, 
        media_type='text/plain', 
        filename=f"{run_id}_genes_template.txt"
    )

def get_next_ext_id(project_id: str):
    ext_test_dir = os.path.join(DATASET_DIR, project_id, "external_test")
    os.makedirs(ext_test_dir, exist_ok=True)
    existing = [d for d in os.listdir(ext_test_dir) if d.startswith("ext") and d[3:].isdigit()]
    if not existing:
        return "ext1"
    ids = [int(d[3:]) for d in existing]
    return f"ext{max(ids) + 1}"

import json

@app.post("/api/runs/{run_id}/inference")
async def run_external_inference(
    run_id: str,
    direction: str = Form(...), # 'AtoB' (MA->RS) or 'BtoA' (RS->MA)
    ext_id: str = Form(...),
    description: str = Form("External Testing dataset"),
    file: UploadFile = File(...)
):
    checkpoint_dir = os.path.join(RESULTS_DIR, "1_Training", "checkpoints", run_id)
    if not os.path.exists(checkpoint_dir):
        raise HTTPException(status_code=404, detail="Run checkpoint not found.")

    if not ext_id.startswith("ext_"):
        raise HTTPException(status_code=400, detail="External ID must start with 'ext_'")

    project_id = run_id.split('_')[0]
    
    # 1. Save uploaded file to dataset folder
    ext_dir = os.path.join(DATASET_DIR, project_id, ext_id)
    os.makedirs(ext_dir, exist_ok=True)
    
    input_filename = "test_ag.tsv" if direction == 'AtoB' else "test_rs.tsv"
    abs_input_path = os.path.join(ext_dir, input_filename)
    
    content = await file.read()
    with open(abs_input_path, "wb") as f:
        f.write(content)

    # Get dimensions
    try:
        df_peek = pd.read_csv(abs_input_path, sep='\t', index_col=0)
        samples_count = len(df_peek)
        genes_count = len(df_peek.columns)
        
        # Save metadata
        metadata = {
            "id": ext_id,
            "description": description,
            "samples": samples_count,
            "genes": genes_count,
            "created_at": time.time()
        }
        with open(os.path.join(ext_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
    except Exception as e: 
        print(f"Metadata error: {e}")

    # 2. Define output path in SyncData
    output_dir = os.path.join(RESULTS_DIR, "2_SyncData", run_id, ext_id)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = "translated_rs.tsv" if direction == 'AtoB' else "translated_ag.tsv"
    output_path = os.path.join(output_dir, output_filename)

    # 3. Run Inference Script
    script_path = os.path.join(BACKEND_DIR, "scripts", "inference.py")
    cmd = [
        sys.executable, script_path,
        "--run_id", run_id,
        "--input", abs_input_path,
        "--direction", direction,
        "--output", output_path,
        "--device", "cpu"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {
            "message": "Inference completed successfully",
            "ext_id": ext_id,
            "output_file": output_filename,
            "output_path": os.path.join(run_id, ext_id, output_filename)
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e.stderr}")

@app.post("/api/runs/{run_id}/sync_external")
async def sync_external(
    run_id: str,
    test_ag: Optional[UploadFile] = File(None),
    test_rs: Optional[UploadFile] = File(None),
    ext_id: str = Form(...),
    description: str = Form("External Testing dataset")
):
    """ONLY UPLOAD FILES. Analysis happens via /run_step."""
    try:
        project_id = run_id.split('_')[0]
        ext_dir = os.path.join(DATASET_DIR, project_id, ext_id)
        os.makedirs(ext_dir, exist_ok=True)
        
        samples_count = 0
        genes_count = 0

        async def save_file(file: UploadFile, filename: str):
            nonlocal samples_count, genes_count
            abs_path = os.path.join(ext_dir, filename)
            content = await file.read()
            with open(abs_path, "wb") as f:
                f.write(content)
            
            if samples_count == 0:
                try:
                    df = pd.read_csv(abs_path, sep='\t', index_col=0)
                    samples_count = len(df)
                    genes_count = len(df.columns)
                except: pass

        if test_ag: await save_file(test_ag, "test_ag.tsv")
        if test_rs: await save_file(test_rs, "test_rs.tsv")

        # Save metadata
        metadata = {
            "id": ext_id,
            "description": description,
            "samples": samples_count,
            "genes": genes_count,
            "created_at": time.time()
        }
        with open(os.path.join(ext_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        return {"message": "Files uploaded successfully", "ext_id": ext_id}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results")
async def get_results_status():
    checkpoints = os.listdir(CHECKPOINTS_DIR) if os.path.exists(CHECKPOINTS_DIR) else []
    logs = [l.replace("_log.txt", "") for l in os.listdir(LOGS_DIR) if l.endswith("_log.txt")]
    
    # Load ongoing/incomplete tasks
    ongoing_task_file = os.path.join(TRAINING_DIR, "ongoing_tasks.txt")
    ongoing_task_ids = []
    if os.path.exists(ongoing_task_file):
        try:
            with open(ongoing_task_file, 'r') as f:
                ongoing_task_ids = [line.strip() for line in f if line.strip()]
        except: pass

    # Define status for each log (run)
    run_statuses = {}
    now = time.time()
    for run_id in logs:
        project_id = run_id.split('_')[0]
        sync_folder = run_id
        sync_path = os.path.join(SYNC_DATA_DIR, sync_folder, "test", "microarray_fake.csv")
        log_path = os.path.join(LOGS_DIR, f"{run_id}_log.txt")
        checkpoint_latest = os.path.join(CHECKPOINTS_DIR, run_id, "net_latest.pth")
        
        # Check if log was updated in the last 60 seconds
        is_running = False
        current_epoch = 0
        total_epochs = 0
        
        if os.path.exists(log_path):
            mtime = os.path.getmtime(log_path)
            if now - mtime < 60:
                is_running = True
                # ... (rest of parsing logic)
                # Try to get current progress from the last few lines
                try:
                    with open(log_path, 'rb') as f:
                        f.seek(0, os.SEEK_END)
                        pos = f.tell()
                        # Read last 2KB to find the last (epoch:...) line
                        f.seek(max(0, pos - 2048))
                        last_lines = f.read().decode('utf-8', errors='ignore').splitlines()
                        for line in reversed(last_lines):
                            parsed = parse_log_line(line)
                            if parsed and 'epoch' in parsed:
                                current_epoch = int(parsed['epoch'])
                                break
                    # Try to infer total epochs from the first few lines of the log
                    # The script usually prints arguments at start
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for _ in range(20): # Check first 20 lines
                            line = f.readline()
                            if not line: break
                            if "epochs" in line.lower() or "n_epochs" in line.lower():
                                match = re.search(r'epochs[:\s=]+(\d+)', line.lower())
                                if match:
                                    total_epochs = int(match.group(1))
                                    break
                except: pass

        if total_epochs == 0: total_epochs = 500 # Default fallback
        
        # External Branch Tracking (SYNCED BY PROJECT DATASET FOLDER)
        ext_ids = []
        ext_statuses = {}
        project_dataset_root = os.path.join(DATASET_DIR, project_id)
        if os.path.exists(project_dataset_root):
            ext_ids = [d for d in os.listdir(project_dataset_root) if d.startswith("ext_") and os.path.isdir(os.path.join(project_dataset_root, d))]
            
            for eid in ext_ids:
                eid_sync_dir = os.path.join(SYNC_DATA_DIR, run_id, eid)
                # Check for translated files or standard names
                sync_done = os.path.exists(os.path.join(eid_sync_dir, "translated_ag.tsv")) or \
                            os.path.exists(os.path.join(eid_sync_dir, "translated_rs.tsv")) or \
                            os.path.exists(os.path.join(eid_sync_dir, "microarray_fake.csv"))
                
                # Metadata from dataset folder
                meta = {"description": "External Testing dataset", "samples": 0, "genes": 0}
                meta_path = os.path.join(project_dataset_root, eid, "metadata.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r') as f: meta = json.load(f)
                    except: pass

                ext_statuses[eid] = {
                    "metadata": meta,
                    "sync": sync_done,
                    "comparative": os.path.exists(os.path.join(eid_sync_dir, "Test_performance.csv")),
                    "deg": os.path.exists(os.path.join(eid_sync_dir, "DEG", "Jaccard_Curve_GANomics.csv")),
                    "pathway": os.path.exists(os.path.join(eid_sync_dir, "Pathway", "Pathway_Concordance_GANomics.csv")),
                    "pred_model": os.path.exists(os.path.join(eid_sync_dir, "Prediction", "Classifier_Performance_GANomics.csv")),
                }
        
        # Internal Test Meta (Heuristic)
        internal_meta = {"description": "Standard Internal Test Set", "note": "Original Test Set from Unseen data points", "samples": 0, "genes": 0}
        if os.path.exists(sync_path):
            try:
                # Use pandas to peek dimensions
                df_peek = pd.read_csv(sync_path, index_col=0, nrows=0)
                internal_meta["genes"] = len(df_peek.columns)
                real_test_path = os.path.join(SYNC_DATA_DIR, run_id, "test", "microarray_real.csv")
                if os.path.exists(real_test_path):
                    # Count rows without loading full file
                    with open(real_test_path, 'rb') as f:
                        internal_meta["samples"] = sum(1 for _ in f) - 1
            except: pass

        run_statuses[run_id] = {
            "training": "running" if is_running else ("completed" if os.path.exists(checkpoint_latest) else "idle"),
            "stopped": running_processes.get(run_id, {}).get('stopped', False) or (run_id == 'training_session' and running_processes.get('training_session', {}).get('stopped', False)),
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "sync": os.path.exists(sync_path),
            "comparative": os.path.exists(os.path.join(COMPARATIVE_DIR, run_id, "Test_performance.csv")),
            "deg": os.path.exists(os.path.join(BIOMARKERS_DIR, "DEG", run_id, "Jaccard_Curve_GANomics.csv")),
            "pathway": os.path.exists(os.path.join(BIOMARKERS_DIR, "Pathway", run_id, "Pathway_Concordance_GANomics.csv")),
            "pred_model": os.path.exists(os.path.join(BIOMARKERS_DIR, "Prediction", run_id, "Classifier_Performance_GANomics.csv")),
            "metadata": internal_meta,
            # External Branch Tracking
            "ext_ids": ext_ids,
            "ext_statuses": ext_statuses,
            "sync_ext": len(ext_ids) > 0,
        }

    return {
        "checkpoints": checkpoints,
        "logs": logs,
        "run_statuses": run_statuses
    }

@app.post("/api/train")
async def start_training(req: TrainRequest):
    run_ablation_script = os.path.join(SCRIPTS_DIR, "run_ablation.py")
    cmd = [sys.executable, run_ablation_script, "--config", req.config_path, "--repeats", str(req.repeats)]
    if req.sizes: cmd += ["--sizes"] + [str(s) for s in req.sizes]
    if req.betas: cmd += ["--betas"] + [str(b) for b in req.betas]
    if req.lambdas: cmd += ["--lambdas"] + [str(l) for l in req.lambdas]
    
    if req.use_gpu:
        gpu_ids = get_available_gpus()
        if gpu_ids:
            cmd += ["--gpu_ids", gpu_ids]
            print(f"Using GPUs: {gpu_ids}")
        else:
            print("GPU requested but none found. Falling back to CPU.")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{BACKEND_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"
    
    try:
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        
        process = subprocess.Popen(cmd, cwd=BACKEND_DIR, env=env, **kwargs)
        
        # For training, we track it by config name or a generic training key
        # Since run_ablation starts multiple runs, we'll use 'training_session'
        running_processes['training_session'] = {
            "pid": process.pid,
            "cmd": cmd,
            "env": env,
            "cwd": BACKEND_DIR,
            "type": "training"
        }
        return {"message": "Training session started", "pid": process.pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/runs/{run_id}/stop")
async def stop_task(run_id: str):
    # 1. Check if it's explicitly in our tracker
    proc_info = running_processes.get(run_id)
    
    # 2. If not found, and it looks like a training run, check if we have a generic training session
    if not proc_info and ("Ablation" in run_id or "Sensitivity" in run_id):
        proc_info = running_processes.get('training_session')
        # verify it's the right project (heuristic)
        if proc_info:
            cmd_str = " ".join(proc_info.get('cmd', []))
            project_id = run_id.split('_')[0]
            if project_id.lower() not in cmd_str.lower():
                proc_info = None

    if proc_info:
        try:
            kill_proc_tree(proc_info['pid'])
            # Mark as stopped in memory
            proc_info['stopped'] = True
            # Keep proc_info for restart!
        except: pass
        
        # User requested: "clicking stop, it should update the status of the ongoing_tasks to be stopped"
        # We don't remove it from ongoing_tasks.txt so it stays in "Incomplete" or "Task Monitor"
        # but we need to know it's not running.
            
        return {"message": f"Task {run_id} stopped"}
    
    # 3. Fallback: find by name in psutil (essential if backend restarted)
    stopped_any = False
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            # Search for run_id in any of the command line arguments
            if cmdline and any(run_id in arg for arg in cmdline):
                kill_proc_tree(proc.info['pid'])
                stopped_any = True
        except (psutil.NoSuchProcess, psutil.AccessDenied): continue
    
    if stopped_any:
        # Cleanup ongoing_tasks.txt even in fallback
        ongoing_file = os.path.join(TRAINING_DIR, "ongoing_tasks.txt")
        if os.path.exists(ongoing_file):
            try:
                with open(ongoing_file, 'r') as f:
                    tasks = [line.strip() for line in f if line.strip()]
                new_tasks = [t for t in tasks if t != run_id]
                if len(new_tasks) < len(tasks):
                    with open(ongoing_file, 'w') as f:
                        for t in new_tasks: f.write(f"{t}\n")
            except: pass
        return {"message": f"Task {run_id} stopped (found via psutil)"}
        
    raise HTTPException(status_code=404, detail=f"Process for {run_id} not found. It may have already finished or the PID is lost.")

@app.post("/api/runs/{run_id}/restart")
async def restart_task(run_id: str):
    proc_info = running_processes.get(run_id)
    if not proc_info and run_id == 'training_session':
        proc_info = running_processes.get('training_session')
        
    cmd = None
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{BACKEND_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"
    cwd = BACKEND_DIR

    if proc_info:
        # Stop first if running
        try: kill_proc_tree(proc_info['pid'])
        except: pass
        cmd = proc_info['cmd']
        cwd = proc_info.get('cwd', BACKEND_DIR)
        env = proc_info.get('env', env)
    else:
        # Heuristic reconstruction for training runs
        # run_id format: PROJECT_Ablation_Size_10_Run_0
        if "Ablation" in run_id or "Sensitivity" in run_id or "Run_" in run_id:
            project_id = run_id.split('_')[0]
            # Try to find config path
            config_path = None
            for root, dirs, files in os.walk(DATASET_DIR):
                if project_id.upper() in root.upper():
                    for f in files:
                        if f.endswith("_config.yaml"):
                            config_path = os.path.join(root, f)
                            break
                if config_path: break
            
            if not config_path:
                raise HTTPException(status_code=404, detail=f"Config not found for project {project_id}")
            
            train_script = os.path.join(SCRIPTS_DIR, "train.py")
            cmd = [sys.executable, train_script, "--config", config_path, "--name", run_id]
            
            # Extract params from run_id
            size_match = re.search(r'Size_(\d+)', run_id)
            if size_match: cmd += ["--max_samples", size_match.group(1)]
            
            beta_match = re.search(r'Beta_([\d.]+)', run_id)
            if beta_match: cmd += ["--lambda_feedback", beta_match.group(1)]
            
            lambda_match = re.search(r'Lambda_([\d.]+)', run_id)
            if lambda_match: cmd += ["--lambda_cycle", lambda_match.group(1)]
            
            if "AtoB" in run_id: cmd += ["--direction", "AtoB"]
            elif "BtoA" in run_id: cmd += ["--direction", "BtoA"]
            
            # Check GPU
            gpu_ids = get_available_gpus()
            if gpu_ids: cmd += ["--device", f"cuda:0"] # Default to first GPU for single run restart
        else:
            raise HTTPException(status_code=404, detail="Restart info not found and cannot be reconstructed")
    
    try:
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        
        process = subprocess.Popen(cmd, cwd=cwd, env=env, **kwargs)
        
        # Update tracker
        running_processes[run_id] = {
            "pid": process.pid,
            "cmd": cmd,
            "env": env,
            "cwd": cwd,
            "type": "training",
            "stopped": False
        }
        return {"message": f"Task {run_id} restarted", "pid": process.pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart: {str(e)}")

@app.post("/api/runs/{run_id}/run_step")
async def run_analysis_step(run_id: str, step: int, config_path: Optional[str] = None, ext_id: Optional[str] = None):
    # Map step numbers to scripts
    scripts = {
        2: "test_sync.py",
        3: "comparative_analysis.py",
        4: "biomarker.py" # Covers both DEG and Prediction
    }
    
    if step not in scripts:
        raise HTTPException(status_code=400, detail=f"Invalid step: {step}")
        
    script_path = os.path.join(SCRIPTS_DIR, scripts[step])
    if not os.path.exists(script_path):
        raise HTTPException(status_code=404, detail=f"Script not found: {scripts[step]}")
        
    cmd = [sys.executable, script_path, "--run_id", run_id]
    
    if ext_id:
        cmd += ["--ext_id", ext_id]
    
    # Step 2 needs config
    if step == 2:
        if not config_path:
            raise HTTPException(status_code=400, detail="Step 2 requires config_path")
        cmd += ["--config", config_path]
    
    # Step 4 might need labels (defaults to NB label.txt)
    if step == 4:
        # Check if project is NB or other to set labels
        project_id = run_id.split('_')[0]
        label_path = os.path.join(DATASET_DIR, project_id, "label.txt")
        if os.path.exists(label_path):
            cmd += ["--labels", label_path]
            
    print(f"Running step {step} command: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{BACKEND_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"
    
    try:
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        
        process = subprocess.Popen(
            cmd, 
            cwd=BACKEND_DIR, 
            env=env, 
            stdout=None, 
            stderr=None, 
            **kwargs
        )
        
        running_processes[run_id] = {
            "pid": process.pid,
            "cmd": cmd,
            "env": env,
            "cwd": BACKEND_DIR,
            "type": "step",
            "step": step
        }
        return {"message": f"Step {step} started", "pid": process.pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start step {step}: {str(e)}")

@app.get("/api/runs/{run_id}/logs")
async def stream_run_logs(run_id: str):
    log_path = os.path.join(LOGS_DIR, run_id)
    if not os.path.exists(log_path):
        if not run_id.endswith("_log.txt"): log_path = os.path.join(LOGS_DIR, f"{run_id}_log.txt")
    if not os.path.exists(log_path):
        clean_id = run_id.replace("_log.txt", "")
        for f in os.listdir(LOGS_DIR):
            if clean_id in f and f.endswith(".txt"):
                log_path = os.path.join(LOGS_DIR, f)
                break
    if not os.path.exists(log_path): raise HTTPException(status_code=404, detail="Log file not found")

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
            structured = []
            for line in all_lines:
                parsed = parse_log_line(line)
                if parsed: structured.append(parsed)
            
            summary = {}
            if structured:
                last = structured[-1]
                # Map internal names to UI-friendly summary names
                summary = {
                    "epoch": last.get("epoch"),
                    "iters": last.get("iters"),
                    "G_A": last.get("G_A", 0),
                    "G_B": last.get("G_B", 0),
                    "D_A": last.get("D_A", 0),
                    "D_B": last.get("D_B", 0),
                    "Cycle": (last.get("cycle_A", 0) + last.get("cycle_B", 0)),
                    "Feedback": (last.get("feedback_A", 0) + last.get("feedback_B", 0)),
                    "IDT": (last.get("idt_A", 0) + last.get("idt_B", 0)),
                }
            return {"run_id": run_id, "summary": summary, "structured": structured, "total_lines": len(all_lines)}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/{project_id}")
async def get_project_metrics(project_id: str):
    table_path = os.path.join(COMPARATIVE_DIR, f"Table_2_Comparison_{project_id.upper()}.csv")
    if not os.path.exists(table_path): return []
    try:
        df = pd.read_csv(table_path)
        return df.to_dict(orient="records")
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/runs/{run_id}/comparative")
async def get_run_comparative_metrics(run_id: str, ext_id: Optional[str] = None):
    run_id = run_id.strip()
    # Try multiple possible paths
    if ext_id:
        possible_paths = [
            os.path.join(SYNC_DATA_DIR, run_id, ext_id, "Test_performance.csv"),
            os.path.join(SYNC_DATA_DIR, run_id, f"algorithms_{ext_id}", "Test_performance.csv")
        ]
    else:
        possible_paths = [
            os.path.join(COMPARATIVE_DIR, run_id, "Test_performance.csv"),
            os.path.join(COMPARATIVE_DIR, run_id, "Table_2_Comparison.csv"),
            os.path.join(RESULTS_DIR, "tables", "biomarkers", "CycleGAN", "Table_2_Comparison.csv") # legacy fallback
        ]
    
    table_path = None
    for p in possible_paths:
        if os.path.exists(p):
            table_path = p
            break
            
    if not table_path:
        print(f"Comparative metrics NOT FOUND for {run_id}. Checked: {possible_paths}")
        raise HTTPException(status_code=404, detail=f"Comparative metrics not found for run {run_id}")
    
    try:
        # Use utf-8 to handle ± symbols
        df = pd.read_csv(table_path, encoding='utf-8')
        return df.to_dict(orient="records")
    except Exception as e: 
        print(f"Error reading comparative metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/runs/{run_id}/sync")
async def get_run_sync_status(run_id: str, ext_id: Optional[str] = None):
    # ... (preserving previous logic)
    pass

@app.get("/api/runs/{run_id}/tsne")
async def get_run_tsne_coords(run_id: str, ext_id: Optional[str] = None):
    if ext_id:
        tsne_path = os.path.join(SYNC_DATA_DIR, run_id, ext_id, "tsne_coords.csv")
    else:
        tsne_path = os.path.join(SYNC_DATA_DIR, run_id, "test", "tsne_coords.csv")

    if not os.path.exists(tsne_path):
        raise HTTPException(status_code=404, detail="t-SNE coordinates not found. Run sync step first.")

    try:
        df = pd.read_csv(tsne_path)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))@app.get("/api/runs/{run_id}/deg")
async def get_run_deg_metrics(run_id: str, ext_id: Optional[str] = None):
    if ext_id:
        deg_dir = os.path.join(SYNC_DATA_DIR, run_id, ext_id, "DEG")
    else:
        deg_dir = os.path.join(BIOMARKERS_DIR, "DEG", run_id)
    
    if not os.path.exists(deg_dir):
        raise HTTPException(status_code=404, detail="DEG results not found")
    
    results = {}
    for f in os.listdir(deg_dir):
        if f.startswith("Jaccard_Curve_") and f.endswith(".csv"):
            algo = f.replace("Jaccard_Curve_", "").replace(".csv", "")
            try:
                df = pd.read_csv(os.path.join(deg_dir, f))
                # Ensure it has threshold and jaccard
                if 'threshold' in df.columns and 'jaccard' in df.columns:
                    # Convert NaN to None for JSON compatibility
                    results[algo] = df.replace({np.nan: None}).to_dict(orient="records")
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
    return results

@app.get("/api/runs/{run_id}/prediction")
async def get_run_prediction_metrics(run_id: str, ext_id: Optional[str] = None):
    if ext_id:
        pred_dir = os.path.join(SYNC_DATA_DIR, run_id, ext_id, "Prediction")
    else:
        pred_dir = os.path.join(BIOMARKERS_DIR, "Prediction", run_id)
        
    if not os.path.exists(pred_dir):
        raise HTTPException(status_code=404, detail="Prediction results not found")
    
    results = {}
    for f in os.listdir(pred_dir):
        if f.startswith("Classifier_Performance_") and f.endswith(".csv"):
            algo = f.replace("Classifier_Performance_", "").replace(".csv", "")
            try:
                df = pd.read_csv(os.path.join(pred_dir, f))
                # Convert NaN to None for JSON compatibility
                results[algo] = df.replace({np.nan: None}).to_dict(orient="records")
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
    return results

@app.get("/api/runs/{run_id}/pathway")
async def get_run_pathway_metrics(run_id: str, ext_id: Optional[str] = None):
    if ext_id:
        pathway_dir = os.path.join(SYNC_DATA_DIR, run_id, ext_id, "Pathway")
    else:
        pathway_dir = os.path.join(BIOMARKERS_DIR, "Pathway", run_id)
        
    if not os.path.exists(pathway_dir):
        raise HTTPException(status_code=404, detail="Pathway results not found")
    
    # Structure: { library_name: { algo_name: { rho, p, null_dist } } }
    results = {}
    for f in os.listdir(pathway_dir):
        if f.startswith("Pathway_Concordance_") and f.endswith(".csv"):
            # Format: Pathway_Concordance_{algo}_{library}.csv
            parts = f.replace("Pathway_Concordance_", "").replace(".csv", "").split("_")
            if len(parts) < 2: continue
            algo = parts[0]
            library = "_".join(parts[1:])
            
            if library not in results: results[library] = {"concordance": {}, "null_distributions": {}}
            
            try:
                df = pd.read_csv(os.path.join(pathway_dir, f))
                # Convert NaN to None for JSON compatibility
                record = df.replace({np.nan: None}).to_dict(orient="records")[0]
                results[library]["concordance"][algo] = record
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
        if f.startswith("Null_Dist_") and f.endswith(".csv"):
            # Format: Null_Dist_{algo}_{library}.csv
            parts = f.replace("Null_Dist_", "").replace(".csv", "").split("_")
            if len(parts) < 2: continue
            algo = parts[0]
            library = "_".join(parts[1:])
            
            if library not in results: results[library] = {"concordance": {}, "null_distributions": {}}
            
            try:
                df = pd.read_csv(os.path.join(pathway_dir, f), header=None)
                # Convert NaN to None for JSON compatibility
                null_dist = [x if not (isinstance(x, float) and np.isnan(x)) else None for x in df[0].tolist()]
                results[library]["null_distributions"][algo] = null_dist
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
    return results

@app.get("/api/projects/{project_id}/ablation")
async def get_project_ablation_metrics(project_id: str):
    # Search all comparative results for this project
    results = []
    if not os.path.exists(COMPARATIVE_DIR): return []
    
    for run_id in os.listdir(COMPARATIVE_DIR):
        if run_id.startswith(project_id):
            perf_path = os.path.join(COMPARATIVE_DIR, run_id, "Test_performance.csv")
            if os.path.exists(perf_path):
                try:
                    df = pd.read_csv(perf_path)
                    # Get GANomics performance
                    row = df[df['Algorithm'] == 'GANomics'].iloc[0].to_dict()
                    row['run_id'] = run_id
                    results.append(row)
                except: pass
    return results

@app.get("/api/projects/{project_id}/ablation_logs")
async def get_project_ablation_logs(project_id: str, category: str):
    # category: architecture, size, sensitivity
    results = []
    if not os.path.exists(LOGS_DIR): return []
    
    for f in os.listdir(LOGS_DIR):
        if f.startswith(project_id) and f.endswith("_log.txt"):
            run_id = f.replace("_log.txt", "")
            
            # Filter by category
            is_match = False
            if category == 'architecture':
                if "Architecture" in run_id: is_match = True
                # Also include baseline (Size 50, but not Architecture)
                elif "Size_50" in run_id and "Architecture" not in run_id: is_match = True
            elif category == 'size':
                if "Size" in run_id and "Architecture" not in run_id: is_match = True
            elif category == 'sensitivity':
                if "Sensitivity" in run_id: is_match = True
            
            if not is_match: continue
            
            log_path = os.path.join(LOGS_DIR, f)
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as lf:
                    lines = lf.readlines()
                    if not lines: continue
                    
                    # Find first and last parsed lines
                    structured = []
                    for line in lines:
                        parsed = parse_log_line(line)
                        if parsed: structured.append(parsed)
                    
                    if len(structured) < 2: continue
                    
                    first = structured[0]
                    last = structured[-1]
                    
                    results.append({
                        "run_id": run_id,
                        "first": first,
                        "last": last
                    })
            except: pass
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8832)
