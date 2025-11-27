import torch
from data_provider.data_factory import data_provider
import argparse

def verify_real_doppler_kaggle_dataloader():
    """
    This function verifies the implementation of the Dataset_Real_Doppler_Kaggle
    by creating a data_loader and printing a sample batch.
    """
    # 1. Create a mock args object
    args = argparse.Namespace()
    args.data = 'real_doppler_kaggle'
    args.task_name = 'classification'
    args.batch_size = 4
    args.num_workers = 0
    # These attributes are not used by the new dataset loader, but we define them for completeness
    # to avoid potential errors if the factory logic changes in the future.
    args.embed = 'timeF' 
    args.freq = 's' 

    print("--- Verifying 'train' data ---")
    
    # 2. Get the dataset and dataloader for the 'train' flag
    try:
        train_data_set, train_data_loader = data_provider(args, flag='train')
    except Exception as e:
        print(f"Error creating data provider for 'train' flag: {e}")
        return

    # 3. Get one batch of data
    try:
        first_batch = next(iter(train_data_loader))
    except StopIteration:
        print("Data loader is empty. Cannot retrieve a batch.")
        return
    except Exception as e:
        print(f"Error retrieving a batch from the data loader: {e}")
        return


    # 4. Print shapes and samples
    batch_x, batch_y = first_batch
    print(f"Successfully loaded a batch from the 'train' data_loader.")
    print(f"Shape of the input data (batch_x): {batch_x.shape}")
    print(f"Shape of the targets (batch_y): {batch_y.shape}")
    
    print("\n--- Sample Data ---")
    print("Input data sample (first sample in batch, first 5 timesteps):")
    print(batch_x[0, :5, :])
    
    print("\nTarget sample (first sample in batch):")
    print(batch_y[0])

    print("\nVerification successful!")

if __name__ == '__main__':
    verify_real_doppler_kaggle_dataloader()