import os
import yaml
import subprocess
import re
import time
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="GANomics API")

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
    genes: int
    samples: int
    config_path: str
    config: Optional[dict] = None

class TrainRequest(BaseModel):
    config_path: str
    sizes: List[int]
    betas: List[float]
    lambdas: List[float]
    repeats: int = 1
    epochs: Optional[int] = 250
    lr: Optional[float] = 0.0002

def get_project_stats(config_path):
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        path_a = cfg['dataset']['path_A']
        
        # Resolve path relative to backend or root
        if not os.path.isabs(path_a):
            # Try root relative
            alt_path = os.path.abspath(os.path.join(ROOT_DIR, path_a))
            if os.path.exists(alt_path): 
                path_a = alt_path
            else:
                # Try backend relative
                path_a = os.path.abspath(os.path.join(BACKEND_DIR, path_a))
        
        if os.path.exists(path_a):
            # Efficiently read columns and row count
            df_headers = pd.read_csv(path_a, index_col=0, nrows=0, sep=None, engine='python')
            # For row count, we can use a fast line count if it's really large, 
            # but for now let's just use pandas with only the index column if possible
            # or just skip full load.
            row_count = 0
            with open(path_a, 'rb') as f:
                row_count = sum(1 for _ in f) - 1 # Subtract header
            return len(df_headers.columns), row_count
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
                pid = os.path.basename(root).upper()
                genes, samples = get_project_stats(full_path)
                
                config_data = {}
                try:
                    with open(full_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                except: pass
                
                projects.append(ProjectInfo(id=pid, name=pid, genes=genes, samples=samples, config_path=full_path, config=config_data))
    return projects

@app.get("/api/results")
async def get_results_status():
    checkpoints = os.listdir(CHECKPOINTS_DIR) if os.path.exists(CHECKPOINTS_DIR) else []
    logs = [l.replace("_log.txt", "") for l in os.listdir(LOGS_DIR) if l.endswith("_log.txt")]
    
        # Define status for each log (run)
    run_statuses = {}
    now = time.time()
    for run_id in logs:
        sync_folder = run_id
        sync_path = os.path.join(SYNC_DATA_DIR, sync_folder, "test", "microarray_fake.csv")
        log_path = os.path.join(LOGS_DIR, f"{run_id}_log.txt")
        checkpoint_latest = os.path.join(CHECKPOINTS_DIR, run_id, "net_latest.pth")
        
        # Check if log was updated in the last 60 seconds
        is_running = False
        if os.path.exists(log_path):
            if now - os.path.getmtime(log_path) < 60:
                is_running = True
        
        run_statuses[run_id] = {
            "training": "running" if is_running else ("completed" if os.path.exists(checkpoint_latest) else "idle"),
            "sync": os.path.exists(sync_path),
            "comparative": os.path.exists(os.path.join(COMPARATIVE_DIR, run_id, "Test_performance.csv")),
            "deg": os.path.exists(os.path.join(BIOMARKERS_DIR, "DEG", run_id, "Jaccard_Curve_GANomics.csv")),
            "pathway": os.path.exists(os.path.join(BIOMARKERS_DIR, "Pathway", run_id, "Pathway_Concordance_GANomics.csv")),
            "pred_model": os.path.exists(os.path.join(BIOMARKERS_DIR, "Prediction", run_id, "Classifier_Performance_GANomics.csv")),
        }

    return {
        "checkpoints": checkpoints,
        "logs": logs,
        "run_statuses": run_statuses
    }

@app.post("/api/train")
async def start_training(req: TrainRequest):
    run_ablation_script = os.path.join(SCRIPTS_DIR, "run_ablation.py")
    # Use sys.executable to ensure we use the same environment's python
    cmd = [sys.executable, run_ablation_script, "--config", req.config_path, "--repeats", str(req.repeats)]
    if req.sizes: cmd += ["--sizes"] + [str(s) for s in req.sizes]
    if req.betas: cmd += ["--betas"] + [str(b) for b in req.betas]
    if req.lambdas: cmd += ["--lambdas"] + [str(l) for l in req.lambdas]
    
    print(f"Starting training command: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{BACKEND_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"
    
    # Use Popen to start a completely detached process
    # creationflags=subprocess.CREATE_NEW_CONSOLE opens a new terminal window on Windows
    try:
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        
        process = subprocess.Popen(
            cmd, 
            cwd=BACKEND_DIR, 
            env=env, 
            stdout=None, # Let it print to its own console
            stderr=None, 
            **kwargs
        )
        return {"message": "Training session started in separate process", "pid": process.pid, "command": " ".join(cmd)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

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
async def get_run_comparative_metrics(run_id: str):
    run_id = run_id.strip()
    # Try multiple possible paths
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
async def get_run_sync_status(run_id: str):
    run_dir = os.path.join(SYNC_DATA_DIR, run_id)
    if not os.path.exists(run_dir):
        return {"exists": False, "details": {}}
    
    status = {
        "exists": True,
        "details": {
            "train": {
                "Microarray": {
                    "Real": os.path.exists(os.path.join(run_dir, "train", "microarray_real.csv")),
                    "Fake": os.path.exists(os.path.join(run_dir, "train", "microarray_fake.csv")),
                },
                "RNA-Seq": {
                    "Real": os.path.exists(os.path.join(run_dir, "train", "rnaseq_real.csv")),
                    "Fake": os.path.exists(os.path.join(run_dir, "train", "rnaseq_fake.csv")),
                }
            },
            "test": {
                "Microarray": {
                    "Real": os.path.exists(os.path.join(run_dir, "test", "microarray_real.csv")),
                    "Fake": os.path.exists(os.path.join(run_dir, "test", "microarray_fake.csv")),
                },
                "RNA-Seq": {
                    "Real": os.path.exists(os.path.join(run_dir, "test", "rnaseq_real.csv")),
                    "Fake": os.path.exists(os.path.join(run_dir, "test", "rnaseq_fake.csv")),
                }
            },
            "algorithms": {
                "Microarray": {
                    "ComBat": os.path.exists(os.path.join(run_dir, "algorithms", "microarray_fake_combat.csv")),
                    "CuBlock": os.path.exists(os.path.join(run_dir, "algorithms", "microarray_fake_cublock.csv")),
                    "QN": os.path.exists(os.path.join(run_dir, "algorithms", "microarray_fake_qn.csv")),
                    "TDM": os.path.exists(os.path.join(run_dir, "algorithms", "microarray_fake_tdm.csv")),
                    "YuGene": os.path.exists(os.path.join(run_dir, "algorithms", "microarray_fake_yugene.csv")),
                },
                "RNA-Seq": {
                    "ComBat": os.path.exists(os.path.join(run_dir, "algorithms", "rnaseq_fake_combat.csv")),
                    "CuBlock": os.path.exists(os.path.join(run_dir, "algorithms", "rnaseq_fake_cublock.csv")),
                    "QN": os.path.exists(os.path.join(run_dir, "algorithms", "rnaseq_fake_qn.csv")),
                    "TDM": os.path.exists(os.path.join(run_dir, "algorithms", "rnaseq_fake_tdm.csv")),
                    "YuGene": os.path.exists(os.path.join(run_dir, "algorithms", "rnaseq_fake_yugene.csv")),
                }
            }
        }
    }
    return status

@app.get("/api/runs/{run_id}/deg")
async def get_run_deg_metrics(run_id: str):
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
                    results[algo] = df.to_dict(orient="records")
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
    return results

@app.get("/api/runs/{run_id}/prediction")
async def get_run_prediction_metrics(run_id: str):
    pred_dir = os.path.join(BIOMARKERS_DIR, "Prediction", run_id)
    if not os.path.exists(pred_dir):
        raise HTTPException(status_code=404, detail="Prediction results not found")
    
    results = {}
    for f in os.listdir(pred_dir):
        if f.startswith("Classifier_Performance_") and f.endswith(".csv"):
            algo = f.replace("Classifier_Performance_", "").replace(".csv", "")
            try:
                df = pd.read_csv(os.path.join(pred_dir, f))
                results[algo] = df.to_dict(orient="records")
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
    return results

@app.post("/api/runs/{run_id}/run_step")
async def run_analysis_step(run_id: str, step: int, config_path: Optional[str] = None):
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
    
    # Step 2 needs config
    if step == 2:
        if not config_path:
            # Try to find config from run_id or projects
            raise HTTPException(status_code=400, detail="Step 2 requires config_path")
        cmd += ["--config", config_path]
    
    # Step 4 might need labels (defaults to NB clinical_info)
    if step == 4:
        # Check if project is NB or other to set labels
        if "NB" in run_id:
            cmd += ["--labels", os.path.join(DATASET_DIR, "NB", "clinical_info.tsv")]
        elif "METSIM" in run_id:
            cmd += ["--labels", os.path.join(DATASET_DIR, "METSIM", "clinical_info.tsv")]
            
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
        return {"message": f"Step {step} started", "pid": process.pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start step {step}: {str(e)}")

@app.get("/api/runs/{run_id}/pathway")
async def get_run_pathway_metrics(run_id: str):
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
                results[library]["concordance"][algo] = df.to_dict(orient="records")[0]
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
                results[library]["null_distributions"][algo] = df[0].tolist()
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8832)
