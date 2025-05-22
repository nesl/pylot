import os

# Directory where files are located
directory = "."

for fname in os.listdir(directory):
    if "edge" in fname:
        new_fname = fname.replace("edge", "local")
        os.rename(os.path.join(directory, fname), os.path.join(directory, new_fname))
        print(f"Renamed: {fname} -> {new_fname}")
