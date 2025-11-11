import pandas as pd
import os

def programmatic_verify():
    # Define file paths
    original_file_path = r'C:\Users\yanzhang\Desktop\Research_Projects\TSL_customed\dataset_test\data.csv'
    extracted_file_path = r'C:\Users\yanzhang\Desktop\Research_Projects\TSL_customed\dataset_test\moteid_3_temp_volt.csv'

    try:
        # --- Load and process original data ---
        df_original = pd.read_csv(original_file_path)
        # Filter for moteid 3 and select the same attributes as the extracted file
        df_original_filtered = df_original[df_original['moteid'] == 3][['date', 'temperature', 'voltage']].copy()

        # --- Load extracted data ---
        df_extracted = pd.read_csv(extracted_file_path)

        # --- Verification ---
        # Sort both dataframes to ensure a consistent order for comparison
        # The original data might not be sorted by date, so we sort both to be sure.
        df_original_sorted = df_original_filtered.sort_values(by=['date', 'temperature', 'voltage']).reset_index(drop=True)
        df_extracted_sorted = df_extracted.sort_values(by=['date', 'temperature', 'voltage']).reset_index(drop=True)

        # Use pandas testing utility to compare the dataframes
        pd.testing.assert_frame_equal(df_original_sorted, df_extracted_sorted)
        
        print("Verification successful: The data in 'moteid_3_temp_volt.csv' is a correct subset of the original 'data.csv'.")

    except AssertionError as e:
        print("Verification failed. The data does not match.")
        print("Details:", e)
    except Exception as e:
        print(f"An error occurred during verification: {e}")

if __name__ == '__main__':
    programmatic_verify()
