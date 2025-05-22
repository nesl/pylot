import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def compute_energy_per_frame(timing_file: Path, power_file: Path):
    if not timing_file.exists() or not power_file.exists():
        print(f"[WARN] Missing file(s): {timing_file}, {power_file}")
        return None, None, None, None  # include raw power values

    try:
        timings = pd.read_csv(timing_file)

        try:
            power = pd.read_csv(power_file, skiprows=1)
            power.columns = [col.strip().lower() for col in power.columns]
            power_column = next((col for col in power.columns if 'power.draw' in col and '[w]' in col), None)
            if power_column is None:
                raise ValueError("Power column not found in header mode")
            print(f"[INFO] Parsed {power_file.name} with header: using column '{power_column}'")
        except Exception:
            power = pd.read_csv(power_file, names=["timestamp", "power"], header=None)
            power_column = "power"
            print(f"[INFO] Parsed {power_file.name} without header")

        # Extract numeric power values
        power_values = power[power_column].astype(str).str.extract(r'(\d+\.\d+|\d+)').astype(float).dropna()
        avg_power = power_values.mean().values[0]

        latencies = timings["detection_time_sec"]
        energies = latencies * avg_power

        return latencies, [avg_power] * len(latencies), energies, power_values

    except Exception as e:
        print(f"[ERROR] Failed to process {timing_file.name} or {power_file.name}: {e}")
        return None, None, None, None

def plot_energy_per_frame(results_dir: Path, model_names: list,
                          output_plot: Path, output_csv: Path, summary_csv: Path):
    model_energy_data = {}
    per_frame_records = []
    summary_records = []

    for model in model_names:
        timing_path = results_dir / f"detection_timings_{model}.csv"
        power_path = results_dir / f"power_log_{model}.csv"

        latencies, avg_powers, energies, raw_power = compute_energy_per_frame(timing_path, power_path)

        if energies is not None:
            model_energy_data[model] = energies
            for i, (lat, power, energy) in enumerate(zip(latencies, avg_powers, energies), start=1):
                per_frame_records.append({
                    "model": model,
                    "frame_number": i,
                    "latency_sec": lat,
                    "avg_power_w": power,
                    "energy_joule": energy
                })

            # Summary statistics
            summary_records.append({
                "model": model,
                "avg_latency_sec": latencies.mean(),
                "min_latency_sec": latencies.min(),
                "max_latency_sec": latencies.max(),

                "avg_power_w": raw_power.mean().values[0],
                "min_power_w": raw_power.min().values[0],
                "max_power_w": raw_power.max().values[0],

                "avg_energy_joule": energies.mean(),
                "min_energy_joule": energies.min(),
                "max_energy_joule": energies.max()
            })

    if not model_energy_data:
        print("[ERROR] No valid model data found. Plotting aborted.")
        return

    # Save full per-frame table
    pd.DataFrame(per_frame_records).to_csv(output_csv, index=False)
    print(f"[INFO] Saved per-frame data to {output_csv}")

    # Save summary statistics table
    pd.DataFrame(summary_records).to_csv(summary_csv, index=False)
    print(f"[INFO] Saved summary data to {summary_csv}")

    # Plot
    plt.figure(figsize=(12, 6))
    for model, energy in model_energy_data.items():
        plt.plot(range(1, len(energy) + 1), energy, label=model)

    plt.xlabel("Frame Number")
    plt.ylabel("Energy per Detection (Joules)")
    plt.title("Frame-wise Energy per Detection for YOLO Models")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"[INFO] Plot saved to {output_plot}")

if __name__ == "__main__":
    results_path = Path("results")
    models = [
        "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
        "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"
    ]
    plot_path = results_path / "energy_per_frame_plot.png"
    csv_path = results_path / "energy_latency_table.csv"
    summary_path = results_path / "model_energy_latency_averages.csv"

    plot_energy_per_frame(results_path, models, plot_path, csv_path, summary_path)
