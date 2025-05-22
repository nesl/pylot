import pandas as pd
import re
import os
import matplotlib.pyplot as plt

# Assign consistent platform → color and vehicle → linestyle
platform_colors = {
    "cloud": "blue",
    "edge": "green",
    "local": "orange"
}
vehicle_styles = {
    "kawasaki.ninja": "solid",
    "audi.tt": "dashed",
    "carlamotors.carlacola": "dotted"
}

def parse_filename(filename):
    match = re.match(r"(.*?)_(.*?)_(.*?)_(.*?)_latency_location_log\.csv", filename)
    if match:
        platform, vehicle, speed, obstacle = match.groups()
        return platform, vehicle, speed, obstacle
    return None, None, None, None

def extract_y(loc_str):
    """Extract Y coordinate (2nd value) from string like '(x, y, z)'"""
    try:
        return float(re.findall(r'-?\d+\.\d+', loc_str)[1])
    except:
        return None

def transform_summary(input_path="latency_logs/brake_summary.csv", output_path="latency_logs/parsed_braking_summary.csv"):
    print(f"Loading from {input_path}")
    df = pd.read_csv(input_path)

    parsed_data = df['filename'].apply(parse_filename)
    df[['platform', 'ego_vehicle', 'speed', 'obstacle']] = pd.DataFrame(parsed_data.tolist(), index=df.index)

    df = df.rename(columns={
        'frame_capture_time': 'capture_time',
        'frame_capture_loc': 'capture_loc',
        'server_response_receive_time': 'brake_rxvd_time',
        'server_response_loc': 'brake_rxvd_loc',
        'stopping_time': 'brake_applied_time',
        'stopping_loc': 'brake_applied_loc'
    })

    # Extract Y-axis distances
    df['capture_y'] = df['capture_loc'].apply(extract_y)
    df['brake_rxvd_y'] = df['brake_rxvd_loc'].apply(extract_y)
    df['brake_applied_y'] = df['brake_applied_loc'].apply(extract_y)

    # Output CSV
    selected_cols = [
        'platform', 'ego_vehicle', 'obstacle', 'speed', 'vehicle_speed_mph',
        'capture_time', 'capture_y',
        'brake_rxvd_time', 'brake_rxvd_y',
        'brake_applied_time', 'brake_applied_y'
    ]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[selected_cols].to_csv(output_path, index=False)
    print(f"[✓] Parsed summary saved to: {output_path}")

    return df

def plot_braking_behavior(df):
    from matplotlib.lines import Line2D

    grouped = df.groupby("obstacle")

    for obstacle, group in grouped:
        fig, ax = plt.subplots(figsize=(12, 6))

        for _, row in group.iterrows():
            x_vals = [row['capture_y'], row['brake_rxvd_y'], row['brake_applied_y']]
            y_val = row['vehicle_speed_mph']
            y_vals = [y_val] * 3

            platform = row['platform']
            vehicle = row['ego_vehicle']

            color = platform_colors.get(platform, "gray")
            linestyle = vehicle_styles.get(vehicle, "solid")

            # Plot line and markers
            ax.plot(x_vals, y_vals, linestyle=linestyle, color=color, alpha=0.7)
            ax.plot(x_vals[0], y_val, marker='*', color=color)
            ax.plot(x_vals[1], y_val, marker='o', color=color)
            ax.plot(x_vals[2], y_val, marker='s', color=color)

        # Obstacle marker line
        ax.axvline(-122, color='red', linestyle='--')

        ax.set_title(f"Braking Response for Obstacle: {obstacle.capitalize()}")
        ax.set_xlabel("Distance from Obstacle (meters)")
        ax.set_ylabel("Speed (mph)")
        ax.invert_xaxis()
        ax.set_xlim(-25, -155)
        ax.grid(True)

        # Cleaned-up legend
        legend_elements = [
            # Platform → Color
            Line2D([0], [0], color='blue', lw=2, label='Cloud'),
            Line2D([0], [0], color='green', lw=2, label='Edge'),
            Line2D([0], [0], color='orange', lw=2, label='Local'),

            # Vehicle → Line style
            Line2D([0], [0], color='black', lw=2, linestyle='solid', label='Bike'),
            Line2D([0], [0], color='black', lw=2, linestyle='dashed', label='Car'),
            Line2D([0], [0], color='black', lw=2, linestyle='dotted', label='Truck'),

            # Marker meanings
            Line2D([0], [0], marker='*', color='black', lw=0, label='Frame Captured'),
            Line2D([0], [0], marker='o', color='black', lw=0, label='Brake Received'),
            Line2D([0], [0], marker='s', color='black', lw=0, label='Brake Applied'),

            # Obstacle line
            Line2D([0], [0], color='red', lw=2, linestyle='--', label='Obstacle')
        ]

        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

        plot_path = f"latency_logs/{obstacle}_braking_plot_cleaned.png"
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"[✓] Plot saved: {plot_path}")
        plt.close()

if __name__ == "__main__":
    df = transform_summary()
    plot_braking_behavior(df)
