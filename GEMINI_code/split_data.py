
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Define the directory path
data_dir = os.path.join('dataset', 'real_doppler_RAD_DAR_database', 'processed')

# Load the data
try:
    input_data = np.load(os.path.join(data_dir, 'input_data.npy'))
    targets = np.load(os.path.join(data_dir, 'targets.npy'))
except FileNotFoundError:
    print(f"Error: Make sure 'input_data.npy' and 'targets.npy' exist in {data_dir}")
    exit()

# Get the total number of samples
n_samples = input_data.shape[0]
indices = np.arange(n_samples)

# Shuffle indices
np.random.shuffle(indices)

# Apply shuffled indices to data
input_data = input_data[indices]
targets = targets[indices]

# Split ratio
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Calculate split indices
train_end = int(n_samples * train_ratio)
val_end = int(n_samples * (train_ratio + val_ratio))

# Split the data
input_data_train = input_data[:train_end]
targets_train = targets[:train_end]

input_data_val = input_data[train_end:val_end]
targets_val = targets[train_end:val_end]

input_data_test = input_data[val_end:]
targets_test = targets[val_end:]

# Save the split datasets
output_dir = data_dir
np.save(os.path.join(output_dir, 'input_data_TRAIN.npy'), input_data_train)
np.save(os.path.join(output_dir, 'targets_TRAIN.npy'), targets_train)

np.save(os.path.join(output_dir, 'input_data_VAL.npy'), input_data_val)
np.save(os.path.join(output_dir, 'targets_VAL.npy'), targets_val)

np.save(os.path.join(output_dir, 'input_data_TEST.npy'), input_data_test)
np.save(os.path.join(output_dir, 'targets_TEST.npy'), targets_test)

print("Data splitting and saving completed successfully.")
print(f"Train set size: {len(input_data_train)}")
print(f"Validation set size: {len(input_data_val)}")
print(f"Test set size: {len(input_data_test)}")

