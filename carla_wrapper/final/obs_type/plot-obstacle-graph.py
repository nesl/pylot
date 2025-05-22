import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Define colors and line styles
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

def save_horizontal_legend(legend_elements, filename="shared_legend.pdf"):
    fig, ax = plt.subplots(figsize=(12, 1))  # Wide and short for horizontal layout
    ax.axis('off')

    legend = ax.legend(
        handles=legend_elements,
        loc='center',
        ncol=len(legend_elements),  # One row, all elements
        frameon=False,
        fontsize=18,
        handlelength=2,
        markerscale=1,
        borderpad=0.1
    )

    # Optional: pad figure to remove tightness
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', transparent=True)
    print(f"[✓] Legend saved: {filename}")
    plt.close()

def plot_obstacle_braking(data_path):
    df = pd.read_csv(data_path)
    fig, ax = plt.subplots(figsize=(12, 6))

    obstacle_order = ['car', 'bike', 'person']
    obstacle_to_y = {obs: i for i, obs in enumerate(obstacle_order)}
    obstacle_y = -122

    df.loc[df['obstacle'] == 'bike', 'capture_y'] -= -11
    df.loc[df['obstacle'] == 'bike', 'brake_rxvd_y'] -= -11
    df.loc[df['obstacle'] == 'bike', 'brake_applied_y'] -= -11

    df.loc[df['obstacle'] == 'person', 'capture_y'] -= -13
    df.loc[df['obstacle'] == 'person', 'brake_rxvd_y'] -= -13
    df.loc[df['obstacle'] == 'person', 'brake_applied_y'] -= -13

    jitter_strength = 0.1  # how much to stagger

    for i, (_, row) in enumerate(df.iterrows()):
        x_vals = [
            row['capture_y'] - obstacle_y,
            row['brake_rxvd_y'] - obstacle_y,
            row['brake_applied_y'] - obstacle_y
        ]
        base_y_val = obstacle_to_y[row['obstacle']]
        jitter = np.random.uniform(-jitter_strength, jitter_strength)
        y_val = base_y_val + jitter
        y_vals = [y_val] * 3

        color = platform_colors.get(row['platform'], "gray")
        linestyle = vehicle_styles.get(row['ego_vehicle'], "solid")

        ax.plot(x_vals, y_vals, linestyle=linestyle, color=color, alpha=0.7, linewidth=2)
        ax.plot(x_vals[0], y_val, marker='*', color=color, markersize=10)
        ax.plot(x_vals[1], y_val, marker='o', color=color, markersize=10)
        ax.plot(x_vals[2], y_val, marker='s', color=color, markersize=10)
        if x_vals[2] < 0:
            ax.plot(x_vals[2], y_val, marker='x', color='red', markersize=12, markeredgewidth=2)


    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel("Distance from Obstacle (meters)", fontsize=15, fontweight='bold')
    ax.set_ylabel("Obstacle Type", fontsize=15, fontweight='bold')
    ax.set_yticks(list(obstacle_to_y.values()))
    ax.set_yticklabels(obstacle_order)
    ax.tick_params(axis='both', labelsize=16)
    # ax.set_xticklabels(fontsize=20)
    ax.invert_xaxis()
    ax.grid(True)

    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Cloud'),
        Line2D([0], [0], color='orange', lw=2, label='On Device'),
        Line2D([0], [0], color='black', lw=2, linestyle='solid', label='Bike'),
        Line2D([0], [0], color='black', lw=2, linestyle='dashed', label='Car'),
        Line2D([0], [0], color='black', lw=2, linestyle='dotted', label='Truck'),
        Line2D([0], [0], marker='*', color='black', lw=0, label='Frame Captured', markersize=10),
        Line2D([0], [0], marker='o', color='black', lw=0, label='Brake Received', markersize=10),
        Line2D([0], [0], marker='s', color='black', lw=0, label='Vehicle Stopped', markersize=10),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Obstacle')
    ]
    # ax.legend(handles=legend_elements, loc='upper left', fontsize=15, frameon=True)
    save_horizontal_legend(legend_elements, "shared_legend.png")

    plt.tight_layout()
    xmin, xmax = ax.get_xlim()
    ax.axvspan(0, xmax, color='red', alpha=0.05)
    ax.text(
        -1.5,
        ax.get_ylim()[0] + 0.35,
        "Unsafe \nBraking \n Zone",
        fontsize=14,
        fontweight='bold',
        color='red',
        ha='left',
        va='top',
        alpha=0.8,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white', alpha=0.5)
    )
    ax.set_xlim(xmin, xmax)

    plt.savefig("obstaclewise_braking_plot_cleaned.png", bbox_inches='tight')
    print("[\u2713] Plot saved: obstaclewise_braking_plot_cleaned.png")
    plt.close()

if __name__ == "__main__":
    plot_obstacle_braking("parsed_braking_summary_avg.csv")
