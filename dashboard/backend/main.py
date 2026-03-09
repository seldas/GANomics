import os
import yaml
import subprocess
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="GANomics API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Path Definitions (Consolidated in Backend) ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BACKEND_DIR, "scripts")
SRC_DIR = os.path.join(BACKEND_DIR, "src")
DATASET_DIR = os.path.join(BACKEND_DIR, "dataset")
RESULTS_DIR = os.path.join(BACKEND_DIR, "results")

LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")

# --- Models ---
class ProjectInfo(BaseModel):
    id: str
    name: str
    genes: int
    samples: int
    config_path: str # Absolute path to the yaml file

class TrainRequest(BaseModel):
    config_path: str # Now we use the full path found during discovery
    sizes: List[int]
    betas: List[float]
    lambdas: List[float]
    repeats: int = 1
    epochs: Optional[int] = 250
    lr: Optional[float] = 0.0002

# --- Helpers ---
def get_project_stats(config_path):
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # In the new design, dataset paths in YAML might be relative to the PROJECT folder
        # or relative to the original ROOT. We need to handle this.
        # Let's assume for now the dashboard discovery handles finding the file.
        # We need the actual data file to count genes/samples.
        
        path_a = cfg['dataset']['path_A']
        # If it's a relative path like 'dataset/NB/NB_AG.csv', we need to check if it exists relative to BACKEND_DIR
        if not os.path.isabs(path_a):
            # Check if it's relative to root or relative to the new dataset dir
            alt_path = os.path.abspath(os.path.join(BACKEND_DIR, "..", "..", path_a))
            if os.path.exists(alt_path):
                path_a = alt_path
            else:
                path_a = os.path.abspath(os.path.join(BACKEND_DIR, path_a))

        if os.path.exists(path_a):
            df = pd.read_csv(path_a, index_col=0, nrows=1)
            full_df = pd.read_csv(path_a, index_col=0)
            return len(full_df.columns), len(full_df)
    except:
        pass
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

# --- Endpoints ---

@app.get("/api/projects", response_model=List[ProjectInfo])
async def list_projects():
    """Discover projects by looking for yaml configs inside dashboard/backend/dataset/ subfolders."""
    projects = []
    if not os.path.exists(DATASET_DIR):
        return []
    
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith("_config.yaml"):
                full_path = os.path.join(root, file)
                # Use the folder name as Project ID
                pid = os.path.basename(root).upper()
                genes, samples = get_project_stats(full_path)
                projects.append(ProjectInfo(
                    id=pid,
                    name=pid,
                    genes=genes,
                    samples=samples,
                    config_path=full_path
                ))
    return projects

@app.get("/api/results")
async def get_results_status():
    status = {
        "checkpoints": os.listdir(CHECKPOINTS_DIR) if os.path.exists(CHECKPOINTS_DIR) else [],
        "logs": os.listdir(LOGS_DIR) if os.path.exists(LOGS_DIR) else []
    }
    return status

@app.post("/api/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Trigger run_ablation.py with the new discovery logic."""
    run_ablation_script = os.path.join(SCRIPTS_DIR, "run_ablation.py")
    
    # run_ablation.py expects --config. We pass the absolute path discovered.
    cmd = [
        "python", run_ablation_script,
        "--config", req.config_path,
        "--repeats", str(req.repeats)
    ]
    
    if req.sizes:
        cmd += ["--sizes"] + [str(s) for s in req.sizes]
    if req.betas:
        cmd += ["--betas"] + [str(b) for b in req.betas]
    if req.lambdas:
        cmd += ["--lambdas"] + [str(l) for l in req.lambdas]
        
    def run_cmd():
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{BACKEND_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}"
        # Execute from BACKEND_DIR so results/ and dataset/ are reachable if scripts use relative paths
        subprocess.run(cmd, cwd=BACKEND_DIR, env=env)

    background_tasks.add_task(run_cmd)
    return {"message": "Training session started", "command": " ".join(cmd)}

@app.get("/api/runs/{run_id}/logs")
async def stream_run_logs(run_id: str, lines: int = 500):
    log_path = os.path.join(LOGS_DIR, run_id)
    if not os.path.exists(log_path):
        if not run_id.endswith("_log.txt"):
            log_path = os.path.join(LOGS_DIR, f"{run_id}_log.txt")
    
    if not os.path.exists(log_path):
        clean_id = run_id.replace("_log.txt", "")
        for f in os.listdir(LOGS_DIR):
            if clean_id in f and f.endswith(".txt"):
                log_path = os.path.join(LOGS_DIR, f)
                break

    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Log file not found")

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
            subset = all_lines[-lines:] if lines > 0 else all_lines
            structured = []
            raw = []
            for line in subset:
                parsed = parse_log_line(line)
                if parsed: structured.append(parsed)
                else:
                    cl = line.strip()
                    if cl: raw.append(cl)
            
            summary = {}
            if structured:
                last = structured[-1]
                summary = {
                    "epoch": last.get("epoch"),
                    "iters": last.get("iters"),
                    "loss_g": (last.get("G_A", 0) + last.get("G_B", 0)) / 2,
                    "loss_d": (last.get("D_A", 0) + last.get("D_B", 0)) / 2,
                }
            
            return {"run_id": run_id, "summary": summary, "structured": structured, "raw": raw[-100:], "total_lines": len(all_lines)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/{project_id}")
async def get_project_metrics(project_id: str):
    table_path = os.path.join(RESULTS_DIR, "tables", f"Table_2_Comparison_{project_id.upper()}.csv")
    if not os.path.exists(table_path):
        return []
    try:
        df = pd.read_csv(table_path)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
