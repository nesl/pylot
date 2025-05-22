import multiprocessing
import subprocess
from datetime import datetime
import os
import signal

EXTRA_CLIENTS_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MODEL = "yolo11x.pt"
IMAGE_DIR = "images"
LOG_DIR = "results"
GPU_LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(GPU_LOG_DIR, exist_ok=True)

def start_gpu_logger(extra_clients):
    log_path = os.path.join(GPU_LOG_DIR, f"gpu_util_{extra_clients}_clients.csv")
    proc = subprocess.Popen([
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu",
        "--format=csv,noheader,nounits",
        "-l", "1"
    ], stdout=open(log_path, "w"))
    return proc, log_path

def stop_gpu_logger(proc):
    proc.send_signal(signal.SIGINT)
    proc.wait()

def run_main_server(extra_clients):
    out_file = os.path.join(LOG_DIR, f"main_client_{extra_clients}_clients.csv")
    cmd = [
        "python3", "run_yolo_main_server.py",
        f"--model={MODEL}",
        f"--images={IMAGE_DIR}",
        f"--out={out_file}"
    ]
    subprocess.run(cmd)

def run_extra_client():
    cmd = [
        "python3", "run_yolo_extra_server.py",
        f"--model={MODEL}",
        f"--images={IMAGE_DIR}"
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    for extra_clients in EXTRA_CLIENTS_LIST:
        print(f"\n[INFO] ==== RUN: 1 main server + {extra_clients} extra clients ====")
        start_time = datetime.now()

        # Start GPU logger
        gpu_proc, gpu_log_path = start_gpu_logger(extra_clients)
        print(f"[INFO] GPU logging to: {gpu_log_path}")

        procs = []

        # Launch background clients FIRST
        for _ in range(extra_clients):
            p = multiprocessing.Process(target=run_extra_client)
            procs.append(p)
            p.start()

        # Small delay to let clients occupy GPU
        if extra_clients > 0:
            print(f"[INFO] Waiting briefly for {extra_clients} clients to initialize...")
            import time; time.sleep(2)

        # Launch main server LAST
        p_main = multiprocessing.Process(target=run_main_server, args=(extra_clients,))
        procs.append(p_main)
        p_main.start()

        for p in procs:
            p.join()

        # Stop GPU logger
        stop_gpu_logger(gpu_proc)

        end_time = datetime.now()
        print(f"[INFO] Run with {extra_clients} extra clients completed in {end_time - start_time}")
