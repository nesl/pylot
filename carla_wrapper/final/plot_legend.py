import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Define dummy handles with the same styles as your plots
cloud = mlines.Line2D([], [], color='blue', label='Cloud')
local = mlines.Line2D([], [], color='orange', label='Local')
bike = mlines.Line2D([], [], color='black', linestyle='--', label='Bike')
car = mlines.Line2D([], [], color='black', linestyle='-', label='Car')
truck = mlines.Line2D([], [], color='black', linestyle=':', label='Truck')

frame_cap = mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='Frame Captured')
brake_rx = mlines.Line2D([], [], color='black', marker='s', linestyle='None', label='Brake Received')
stopped = mlines.Line2D([], [], color='black', marker='*', linestyle='None', label='Vehicle Stopped')
obstacle = mlines.Line2D([], [], color='red', linestyle='--', label='Obstacle')

# Create figure for legend
fig_legend = plt.figure(figsize=(10, 1))  # Adjust width/height as needed
fig_legend.legend(
    handles=[cloud, local, bike, car, truck, frame_cap, brake_rx, stopped, obstacle],
    loc='center',
    ncol=5,  # Adjust column count
    frameon=False,
    fontsize='medium'
)
plt.axis('off')
plt.tight_layout()
fig_legend.savefig("shared_legend.png", bbox_inches='tight', transparent=True)
