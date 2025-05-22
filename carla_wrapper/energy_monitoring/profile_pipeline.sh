#!/bin/bash
MODELS=("yolo11n.pt" "yolo11s.pt" "yolo11m.pt" "yolo11l.pt" "yolo11x.pt"
        "yolov8n.pt" "yolov8s.pt" "yolov8m.pt" "yolov8l.pt" "yolov8x.pt")

IMAGEDIR="images"
RESULTDIR="results"

for model in "${MODELS[@]}"; do
    base=$(basename $model .pt)
    echo -e "\n=== Profiling model: $base ==="

    DETECTION_CSV="$RESULTDIR/detection_timings_${base}.csv"
    POWER_LOG="$RESULTDIR/power_log_${base}.csv"
    NSYS_REPORT="$RESULTDIR/profile_${base}"

    echo "[1/5] Clean previous logs for $base"
    rm -f "$POWER_LOG" "$DETECTION_CSV" "$NSYS_REPORT.qdrep"

    echo "[2/5] Start power logging..."
    nvidia-smi --query-gpu=timestamp,power.draw --format=csv -lms 10 > "$POWER_LOG" &
    POWER_PID=$!

    echo "[3/5] Run model under Nsight..."
    nsys profile --trace=cuda,nvtx,cudnn \
        --output="$NSYS_REPORT" \
        python run_yolo_nvtx.py --model models/$model --output "$DETECTION_CSV" --img_dir "$IMAGEDIR"

    echo "[4/5] Stop power logging..."
    kill $POWER_PID
done

echo -e "\n[✔] All models processed. Results in $RESULTDIR/"
