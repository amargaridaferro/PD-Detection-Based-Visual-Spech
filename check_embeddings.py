import numpy as np

# Load the .npz file
data = np.load('Tese/PD/features/lipread_emb/p_01/p_01-00001-00004.npz')

# Access the 'embeddings' array
embeddings = data['embeddings']

# Print the shape, data type, and values of the 'embeddings' array
print('Shape:', embeddings.shape)
print('Data Type:', embeddings.dtype)
print('Values:', embeddings)