import h5py
import pandas as pd

# Load the HDF5 file
file_path = "./sample-dataset/10t-10n-DOS2019-dataset-val.hdf5"

with h5py.File(file_path, "r") as f:
    # List available datasets inside the file
    print("Datasets in the HDF5 file:", list(f.keys()))
with h5py.File(file_path, "r") as f:
    # Load dataset (modify "X_val" and "Y_val" if different)
    X_val = f["set_x"][:]
    Y_val = f["set_y"][:]
X_val_flattened = X_val.reshape(X_val.shape[0], -1)
# Convert to DataFrame
df_X = pd.DataFrame(X_val_flattened)
df_Y = pd.DataFrame(Y_val, columns=["Label"])  # Name the column if applicable

# Combine into one CSV if needed
df_combined = pd.concat([df_X, df_Y], axis=1)

# Save to CSV
df_combined.to_csv("val.csv", index=False)


print(df_combined.describe())