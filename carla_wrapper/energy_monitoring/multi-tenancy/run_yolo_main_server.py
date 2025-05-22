import torch
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--images", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

model = YOLO(args.model).to("cuda:0")
model(torch.zeros(1, 3, 640, 640).cuda())  # Warm-up

images = sorted(Path(args.images).glob("*.png"))
if len(images) == 0:
    raise RuntimeError("No images found!")

timings = []
for i, img_path in enumerate(images):
    start = time.perf_counter()
    results = model(img_path)
    end = time.perf_counter()
    duration = end - start
    timings.append(duration)
    print(f"[INFO][Main] Frame {i+1:03d}: {duration:.4f} sec")

df = pd.DataFrame(timings, columns=["detection_time_sec"])
df.to_csv(args.out, index=False)
print(f"[INFO][Main] Output saved to {args.out}. Avg time: {np.mean(timings):.4f} sec")
