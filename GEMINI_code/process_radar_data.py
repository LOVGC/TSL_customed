import os
import pandas as pd
import numpy as np

def process_radar_data(base_path="dataset/real_doppler_RAD_DAR_database"):
    raw_data_path = os.path.join(base_path, "raw")
    processed_data_path = os.path.join(base_path, "processed")

    # Ensure the processed data directory exists
    os.makedirs(processed_data_path, exist_ok=True)

    # Define class labels
    class_labels = {
        "Cars": 0,
        "Drones": 1,
        "People": 2
    }
    
    all_input_data = []
    all_targets = []

    for class_name, label in class_labels.items():
        class_folder = os.path.join(raw_data_path, class_name)
        if not os.path.exists(class_folder):
            print(f"Warning: Class folder '{class_folder}' not found. Skipping.")
            continue

        for root, _, files in os.walk(class_folder):
            print(f"processing {root}")
            for file_name in files:
                if file_name.endswith(".csv"):
                    file_path = os.path.join(root, file_name)
                    try:
                        # Read the CSV file
                        df = pd.read_csv(file_path, header=None)
                        # Convert to numpy array
                        matrix_data = df.to_numpy(dtype=np.float32)
                        
                        # Ensure the matrix is 11x61 as expected
                        if matrix_data.shape != (11, 61):
                            print(f"Warning: Skipping {file_name} due to unexpected shape {matrix_data.shape}. Expected (11, 61).")
                            continue
                        
                        # Reshape to (seq_len, features) which is (61, 11)
                        reshaped_data = matrix_data.T
                        
                        # target 就是 label, i.e. {0, 1, 2}
                        
                        target = label

                        all_input_data.append(reshaped_data)
                        all_targets.append(target)
                    except Exception as e:
                        print(f"Error processing {file_name}: {e}")
    
    if not all_input_data:
        print("No data processed. Exiting.")
        return

    # Concatenate all data into final NumPy arrays
    input_data_array = np.array(all_input_data)
    targets_array = np.array(all_targets)

    # Save the processed data
    np.save(os.path.join(processed_data_path, "input_data.npy"), input_data_array)
    np.save(os.path.join(processed_data_path, "targets.npy"), targets_array)

    print(f"Processed {len(all_input_data)} samples.")
    print(f"Input data shape: {input_data_array.shape}")  # (N, seq_len, features)
    print(f"Targets data shape: {targets_array.shape}")   # (N, 3), here, we use one-hot encoding for classes
    print(f"Processed data saved to {processed_data_path}")

if __name__ == "__main__":
    process_radar_data()
