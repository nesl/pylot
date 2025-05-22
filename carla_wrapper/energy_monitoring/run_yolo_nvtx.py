# run_yolo_nvtx.py
import argparse
import time
import torch
import nvtx
from ultralytics import YOLO
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--img_dir", type=str, default="images")
args = parser.parse_args()

print(f"[INFO] Running inference with model: {args.model}")
model = YOLO(args.model).to("cuda:0")
model(torch.zeros(1, 3, 640, 640).cuda())  # warm-up

images = sorted(Path(args.img_dir).glob("*.png"))
print(f"[INFO] Found {len(images)} images")

timings = []
with nvtx.annotate("YOLO Inference Loop", color="blue"):
    for i, img_path in enumerate(images):
        start = time.perf_counter()
        _ = model(img_path)
        end = time.perf_counter()
        timings.append(end - start)
        print(f"[INFO] Frame {i+1:03d}: {end - start:.4f} sec")

# Save timing
out_path = Path(args.output)
out_path.parent.mkdir(exist_ok=True, parents=True)
pd.DataFrame(timings, columns=["detection_time_sec"]).to_csv(out_path, index=False)
print(f"[INFO] Timing saved to {out_path}")
