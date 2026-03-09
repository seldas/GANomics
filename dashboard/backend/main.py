import os
import yaml
import subprocess
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

# Base Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")

# --- Models ---
class ProjectInfo(BaseModel):
    id: str
    name: str
    genes: int
    samples: int
    config_path: str

class TrainRequest(BaseModel):
    config: str
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
        path_a = os.path.join(ROOT_DIR, cfg['dataset']['path_A'])
        if os.path.exists(path_a):
            df = pd.read_csv(path_a, index_col=0, nrows=5)
            # This is a bit slow for large files, but for the summary it's ok
            # Better to use a precomputed stat or just read the full file if it's small
            full_df = pd.read_csv(path_a, index_col=0)
            return len(full_df.columns), len(full_df)
    except:
        pass
    return 0, 0

# --- Endpoints ---

@app.get("/api/projects", response_model=List[ProjectInfo])
async def list_projects():
    projects = []
    for file in os.listdir(CONFIG_DIR):
        if file.endswith("_config.yaml"):
            pid = file.replace("_config.yaml", "").upper()
            path = os.path.join(CONFIG_DIR, file)
            genes, samples = get_project_stats(path)
            projects.append(ProjectInfo(
                id=pid,
                name=pid,
                genes=genes,
                samples=samples,
                config_path=file
            ))
    return projects

@app.get("/api/results")
async def get_results_status():
    """Scan results folder to see what's finished."""
    status = {
        "checkpoints": os.listdir(CHECKPOINTS_DIR) if os.path.exists(CHECKPOINTS_DIR) else [],
        "logs": os.listdir(LOGS_DIR) if os.path.exists(LOGS_DIR) else []
    }
    return status

@app.post("/api/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Trigger run_ablation.py in the background."""
    cmd = [
        "python", "scripts/run_ablation.py",
        "--config", f"configs/{req.config}",
        "--repeats", str(req.repeats)
    ]
    
    if req.sizes:
        cmd += ["--sizes"] + [str(s) for s in req.sizes]
    if req.betas:
        cmd += ["--betas"] + [str(b) for b in req.betas]
    if req.lambdas:
        cmd += ["--lambdas"] + [str(l) for l in req.lambdas]
        
    # Start process
    def run_cmd():
        env = os.environ.copy()
        env["PYTHONPATH"] = ROOT_DIR
        subprocess.run(cmd, cwd=ROOT_DIR, env=env)

    background_tasks.add_task(run_cmd)
    return {"message": "Training session started", "command": " ".join(cmd)}

@app.get("/api/logs/{filename}")
async def get_log(filename: str):
    log_path = os.path.join(LOGS_DIR, filename)
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Log not found")
    
    with open(log_path, 'r') as f:
        # Return last 100 lines for efficiency
        lines = f.readlines()
        return {"content": "".join(lines[-100:])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
