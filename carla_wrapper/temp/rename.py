import os

# Directory where files are located
directory = "."

for fname in os.listdir(directory):
    if "_cloud_" in fname:
        new_fname = fname.replace("_cloud_", "_edge_")
        os.rename(os.path.join(directory, fname), os.path.join(directory, new_fname))
        print(f"Renamed: {fname} -> {new_fname}")
