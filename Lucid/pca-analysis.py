import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import h5py
import os

# === Step 1: Load the dataset (supports .data and .hdf5 files) ===
dataset_path = "./sample-dataset/10t-10n-DOS2019-preprocess.data" 

def load_dataset(file_path, max_length=50):
    """Loads dataset from either a .data (pickled) or .hdf5 file, with padding and flattening."""
    if file_path.endswith('.data'):
        print("🔹 Loading pickled .data file...")
        with open(file_path, 'rb') as filehandle:
            preprocessed_flows = pickle.load(filehandle)
        
        # Convert to list of fragments
        from lucid_dataset_parser import dataset_to_list_of_fragments
        X, y, _ = dataset_to_list_of_fragments(preprocessed_flows)

        # Find maximum feature size per packet
        max_features = max([seq.shape[1] if len(seq.shape) > 1 else len(seq) for seq in X])

        # Adjust padding size
        feature_vector_length = max_length * max_features  # Total length after flattening

        # Pad and flatten sequences
        X_padded = np.zeros((len(X), feature_vector_length))  # Create uniform shape array
        for i, seq in enumerate(X):
            seq = np.array(seq).flatten()  # Flatten each sequence (convert 2D to 1D)
            seq_length = min(len(seq), feature_vector_length)  # Truncate if too long
            X_padded[i, :seq_length] = seq[:seq_length]  # Store in padded array
        
        X = np.array(X_padded)
        y = np.array(y)

    elif file_path.endswith('.hdf5'):
        print("🔹 Loading HDF5 dataset...")
        with h5py.File(file_path, 'r') as f:
            X = np.array(f['set_x'])  # Feature data
            y = np.array(f['set_y'])  # Labels
    else:
        raise ValueError("Unsupported file format. Use either .data or .hdf5")

    return X, y

# Check if file exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

print("📥 Loading dataset...")
X, y = load_dataset(dataset_path)
print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")

# === Step 2: Normalize the features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Features normalized.")

# === Step 3: Apply PCA ===
n_components = min(10, X_scaled.shape[1])  # Use up to 10 components or the max available
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# === Step 4: Explained Variance Plot ===
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_components + 1), explained_variance.cumsum(), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid()
plt.show()

# === Step 5: Identify Most Important Features ===
feature_names = [f'Feature {i}' for i in range(X.shape[1])] 
pca_components = pd.DataFrame(pca.components_, columns=feature_names, index=[f'PC{i+1}' for i in range(pca.n_components)])

# Show top 5 most important features for each principal component
for i in range(5):  
    print(f"\n🔹 Top features for Principal Component {i+1}:")
    print(pca_components.iloc[i].abs().nlargest(10))  
    
# === Step 6: Visualize PCA Projection ===
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection')
plt.colorbar(label='Class')
plt.show()

print("\n✅ PCA analysis completed!")
