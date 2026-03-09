import subprocess
import sys
import os
import time

def main():
    root_dir = os.path.abspath(os.path.dirname(__file__))
    
    # 1. Start Backend (FastAPI)
    # The script uses uvicorn internally on port 8000
    print("\n[BACKEND] Starting FastAPI on http://localhost:8000 ...")
    backend_proc = subprocess.Popen(
        [sys.executable, "dashboard/backend/main.py"],
        cwd=root_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # 2. Start Frontend (Vite)
    # Vite defaults to port 5173
    print("[FRONTEND] Starting Vite on http://localhost:5173 ...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=os.path.join(root_dir, "dashboard", "frontend"),
        shell=True, # Required for npm on Windows
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    print("\n✅ Dashboard is launching. Press Ctrl+C to stop both processes.\n")

    try:
        # Simple loop to stream output from both (first few lines to confirm startup)
        # In a real tool, you might want a more sophisticated multiplexer
        while True:
            # Check if processes are still alive
            if backend_proc.poll() is not None:
                print("\n[!] Backend process exited.")
                break
            if frontend_proc.poll() is not None:
                print("\n[!] Frontend process exited.")
                break
            
            # Print any available output (optional, but good for debugging)
            # For simplicity in this "start" script, we'll just wait
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping Dashboard...")
    finally:
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Done.")

if __name__ == "__main__":
    main()
