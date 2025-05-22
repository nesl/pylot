import torch
import time
import argparse
from pathlib import Path
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--images", required=True)
args = parser.parse_args()

model = YOLO(args.model).to("cuda:0")
model(torch.zeros(1, 3, 640, 640).cuda())

images = sorted(Path(args.images).glob("*.png"))
if len(images) == 0:
    raise RuntimeError("No images found!")

for img_path in images:
    start = time.perf_counter()
    _ = model(img_path)
    end = time.perf_counter()
