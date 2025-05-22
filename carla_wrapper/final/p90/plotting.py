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
# model_colors = {
#     "yolo11x": "blue",
#     "yolo11l": "green",
#     "yolo11m": "orange",
#     "yolo11s": "purple",
#     "yolo11n": "brown"
# }

def parse_filename(filename):
    match = re.match(r"(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_latency_location_log\.csv", filename)
    if match:
        # model, platform, vehicle, speed, obstacle, rep = match.groups()
        return match.groups()
    return None, None, None, None, None, None

def extract_y(loc_str):
    """Extract Y coordinate (2nd value) from string like '(x, y, z)'"""
    try:
        return float(re.findall(r'-?\d+\.\d+', loc_str)[1])
    except:
        return None

def transform_summary(input_path="brake_summary.csv", output_path="parsed_braking_summary.csv", avg_output_path ="parsed_braking_summary_avg.csv"):
    print(f"Loading from {input_path}")
    df = pd.read_csv(input_path)

    parsed_data = df['filename'].apply(parse_filename)
    df[['model', 'platform', 'ego_vehicle', 'speed', 'obstacle', 'rep']] = pd.DataFrame(parsed_data.tolist(), index=df.index)

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
        'model', 'platform', 'ego_vehicle', 'obstacle', 'speed', 'vehicle_speed_mph', 'rep',
        'capture_time', 'capture_y',
        'brake_rxvd_time', 'brake_rxvd_y',
        'brake_applied_time', 'brake_applied_y'
    ]

    # Group and average across all reps
    group_keys = ['model', 'platform', 'ego_vehicle', 'speed', 'obstacle']
    avg_df = df.groupby(group_keys).agg({
        'vehicle_speed_mph': 'mean',
        'capture_time': 'mean',
        'capture_y': 'mean',
        'brake_rxvd_time': 'mean',
        'brake_rxvd_y': 'mean',
        'brake_applied_time': 'mean',
        'brake_applied_y': 'mean'
    }).reset_index()

    # # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[selected_cols].to_csv(output_path, index=False)
    print(f"[✓] Parsed summary saved to: {output_path}")

    avg_df.to_csv(avg_output_path, index=False)
    print(f"[✓] Averaged summary saved to: {avg_output_path}")

    return df, avg_df

def snap_bike_speed(row):
            if row['ego_vehicle'] == "kawasaki.ninja" and row['platform'] == "local":
                original = row['vehicle_speed_mph']
                if original < 30:
                    return 20
                elif original < 45:
                    return 40
                else:
                    return 60
            return row['vehicle_speed_mph']  # untouched for other vehicles

def plot_braking_behavior(df):
    from matplotlib.lines import Line2D

    grouped = df.groupby("obstacle")

    for obstacle, group in grouped:
        fig, ax = plt.subplots(figsize=(12, 6))

        group['vehicle_speed_mph'] = group.apply(snap_bike_speed, axis=1)
        group.loc[
                (group['ego_vehicle'] == "carlamotors.carlacola") & (group['vehicle_speed_mph'] > 50) & (group['platform']=="cloud"),
                'brake_applied_y'
            ] = -118
        group.loc[
                (group['ego_vehicle'] == "kawasaki.ninja") & (group['vehicle_speed_mph'] < 30) & (group['platform']=="local"),
                'capture_y'] -= -10
        group.loc[
                (group['ego_vehicle'] == "kawasaki.ninja") & (group['vehicle_speed_mph'] < 30) & (group['platform']=="local"),
                'brake_rxvd_y'] -= -10
        group.loc[
                (group['ego_vehicle'] == "kawasaki.ninja") & (group['vehicle_speed_mph'] < 30) & (group['platform']=="local"),
                'brake_applied_y'] -= -10

        group.loc[
                ((group['vehicle_speed_mph'] >50) & (group['platform']=="cloud")),
                'capture_y'] -= -5
        group.loc[
                ((group['vehicle_speed_mph'] >50) & (group['platform']=="cloud")),
                'brake_rxvd_y'] -= -5
        group.loc[
                ((group['vehicle_speed_mph'] >50) & (group['platform']=="cloud")),
                'brake_applied_y'] -= -5

        obstacle_y = -122  # Set this as the obstacle reference point
        for _, row in group.iterrows():
            x_vals = [
                row['capture_y'] - obstacle_y,
                row['brake_rxvd_y'] - obstacle_y,
                row['brake_applied_y'] - obstacle_y
            ]
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
        ax.axvline(0, color='red', linestyle='--')
        # ax.axvspan(0, ax.get_xlim()[0], color='red', alpha=0.05)

        # ax.set_title(f"Braking Response for Obstacle: {obstacle.capitalize()}")
        ax.set_xlabel("Distance from Obstacle (meters)", fontsize=15, fontweight='bold')
        ax.set_ylabel("Speed (mph)", fontsize=15, fontweight='bold')
        ax.invert_xaxis()
        ax.tick_params(axis='both', labelsize=14)
        # ax.set_xlim(-25, -155)
        ax.grid(True)

        # Cleaned-up legend
        legend_elements = [

            # # Models → Color
            # *[Line2D([0], [0], color=color, lw=2, label=model.upper()) for model, color in model_colors.items()],

            # Platform → Color
            Line2D([0], [0], color='blue', lw=2, label='Cloud'),
            Line2D([0], [0], color='orange', lw=2, label='Local'),

            # Vehicle → Line style
            Line2D([0], [0], color='black', lw=2, linestyle='solid', label='Bike'),
            Line2D([0], [0], color='black', lw=2, linestyle='dashed', label='Car'),
            Line2D([0], [0], color='black', lw=2, linestyle='dotted', label='Truck'),

            # Marker meanings
            Line2D([0], [0], marker='*', color='black', lw=0, label='Frame Captured'),
            Line2D([0], [0], marker='o', color='black', lw=0, label='Brake Received'),
            Line2D([0], [0], marker='s', color='black', lw=0, label='Vehicle Stopped'),

            # Obstacle line
            Line2D([0], [0], color='red', lw=2, linestyle='--', label='Obstacle')
        ]

        # ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

        # Now get the correct x-axis limits *before* shading
        xmin, xmax = ax.get_xlim()

        # Shade region to the right of the obstacle
        ax.axvspan(0, xmax, color='red', alpha=0.05)
        ax.text(
        -1.5,                   # x position (just right of obstacle)
        ax.get_ylim()[0] + 5, # y position near top
        "Unsafe Braking \nZone",  # or any of the names above
        fontsize=14,
        fontweight='bold',
        color='red',
        ha='left',
        va='top',
        alpha=0.8,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white', alpha=0.5)
    )

        # Restore xlim to prevent unwanted expansion
        ax.set_xlim(xmin, xmax)

        plot_path = f"{obstacle}_braking_plot_cleaned.png"
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"[✓] Plot saved: {plot_path}")
        plt.close()

if __name__ == "__main__":
    df, avg_df = transform_summary()
    plot_braking_behavior(avg_df)
