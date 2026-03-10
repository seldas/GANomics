import subprocess
import sys
import os
import time
import threading

def stream_output(pipe, prefix, log_path):
    """
    Reads from a pipe line by line, prints with a prefix, and saves to a log file.
    """
    try:
        # Open in append mode
        with open(log_path, 'a', encoding='utf-8') as f:
            for line in iter(pipe.readline, ''):
                if line:
                    stripped_line = line.strip()
                    # Print to console
                    print(f"{prefix} {stripped_line}")
                    # Write to file with timestamp
                    f.write(f"{time.ctime()} | {stripped_line}\n")
                    f.flush() # Ensure it's written immediately
    except Exception:
        pass
    finally:
        pipe.close()

def main():
    root_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Create logs directory
    logs_dir = os.path.join(root_dir, "app_logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    backend_log = os.path.join(logs_dir, "backend.log")
    frontend_log = os.path.join(logs_dir, "frontend.log")
    
    # 1. Start Backend (FastAPI)
    print(f"\n[SYSTEM] Starting Backend on http://localhost:8832 (Logging to {backend_log})...")
    backend_proc = subprocess.Popen(
        ['./venv/bin/python', "dashboard/backend/main.py"],
        cwd=root_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Start thread to stream backend logs
    backend_thread = threading.Thread(
        target=stream_output, 
        args=(backend_proc.stdout, "[BACKEND]", backend_log), 
        daemon=True
    )
    backend_thread.start()
    
    # 2. Start Frontend (Vite)
    print(f"[SYSTEM] Starting Frontend on http://localhost:8831 (Logging to {frontend_log})...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=os.path.join(root_dir, "dashboard", "frontend"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Start thread to stream frontend logs
    frontend_thread = threading.Thread(
        target=stream_output, 
        args=(frontend_proc.stdout, "[FRONTEND]", frontend_log), 
        daemon=True
    )
    frontend_thread.start()
    
    print("\n✅ Dashboard is launching. Press Ctrl+C to stop both processes.\n")

    try:
        while True:
            # Check if processes are still alive
            if backend_proc.poll() is not None:
                print("\n[!] Backend process exited.")
                break
            if frontend_proc.poll() is not None:
                print("\n[!] Frontend process exited.")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping Dashboard...")
    finally:
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Done.")

if __name__ == "__main__":
    main()
