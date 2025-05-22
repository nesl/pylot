import os
import pandas as pd

def extract_brake_true_rows(logs_dir, output_file='brake_summary.csv'):
    summary_rows = []
    files = [f for f in os.listdir(logs_dir) if f.endswith('_latency_location_log.csv')]

    for file in files:
        file_path = os.path.join(logs_dir, file)
        try:
            df = pd.read_csv(file_path)
            brake_true_rows = df[df['brake_issued'] == True]
            if not brake_true_rows.empty:
                row = brake_true_rows.iloc[0].copy()
                row['filename'] = file
                summary_rows.append(row)
            else:
                print(f"[!] No 'brake_issued == True' row found in: {file}")
        except Exception as e:
            print(f"[!] Failed to process {file}: {e}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(logs_dir, output_file), index=False)
        print(f"[✓] Summary written to {output_file} in {logs_dir}")
    else:
        print("[!] No valid brake rows found across all files.")

# Example usage
if __name__ == "__main__":
    logs_directory = "latency_logs"  # Replace with your actual directory
    extract_brake_true_rows(logs_directory)
