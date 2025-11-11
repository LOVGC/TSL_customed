import pandas as pd
import matplotlib.pyplot as plt
import os

def verify_extraction():
    # Define file paths
    original_file_path = r'C:\Users\yanzhang\Desktop\Research_Projects\TSL_customed\dataset_test\data.csv'
    extracted_file_path = r'C:\Users\yanzhang\Desktop\Research_Projects\TSL_customed\dataset_test\moteid_3_temp_volt.csv'

    # --- Load and process original data ---
    try:
        df_original = pd.read_csv(original_file_path)
    except Exception as e:
        print(f"Error reading original file: {e}")
        return

    # Filter for moteid 3 and select attributes
    df_original_filtered = df_original[df_original['moteid'] == 3].copy()
    # No datetime creation, use default index for x-axis
    # Ensure consistent order for comparison, though index will be used for plotting
    df_original_filtered = df_original_filtered.sort_values(by=['date', 'temperature', 'voltage']).reset_index(drop=True)


    # --- Load extracted data ---
    try:
        df_extracted = pd.read_csv(extracted_file_path)
    except Exception as e:
        print(f"Error reading extracted file: {e}")
        return
    # No datetime creation, use default index for x-axis
    # Ensure consistent order for comparison, though index will be used for plotting
    df_extracted = df_extracted.sort_values(by=['date', 'temperature', 'voltage']).reset_index(drop=True)


    # --- Create comparison plots ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12)) # Removed sharex=True

    # Plot Temperature
    ax1.plot(df_original_filtered.index, df_original_filtered['temperature'], label='Original Data', marker='.', linestyle='None', alpha=0.6)
    ax1.plot(df_extracted.index, df_extracted['temperature'], label='Extracted Data', marker='x', linestyle='None', alpha=0.6)
    ax1.set_title('Temperature Comparison (Mote ID 3)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True)

    # Plot Voltage
    ax2.plot(df_original_filtered.index, df_original_filtered['voltage'], label='Original Data', marker='.', linestyle='None', alpha=0.6)
    ax2.plot(df_extracted.index, df_extracted['voltage'], label='Extracted Data', marker='x', linestyle='None', alpha=0.6)
    ax2.set_title('Voltage Comparison (Mote ID 3)')
    ax2.set_ylabel('Voltage (V)')
    ax2.set_xlabel('Data Point Index') # Updated x-axis label
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    
    # Display the plot
    try:
        plt.show()
        print("Displayed comparison plot.")
    except Exception as e:
        print(f"Error displaying plot: {e}")

if __name__ == '__main__':
    verify_extraction()