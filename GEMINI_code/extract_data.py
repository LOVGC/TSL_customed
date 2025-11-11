import pandas as pd
import os

# Define the path to the input .csv file
input_file_path = r'C:\Users\yanzhang\Desktop\Research_Projects\TSL_customed\dataset_test\data.csv'

# Define the moteid and attributes to extract
moteid_to_extract = 3
attributes_to_extract = ['date', 'temperature', 'voltage'] # Exclude 'time' as requested

# Read the .csv file into a pandas DataFrame
try:
    df = pd.read_csv(input_file_path)
except Exception as e:
    print(f"Error reading the file: {e}")
    exit()

# Filter the DataFrame for the specified moteid
df_filtered = df[df['moteid'] == moteid_to_extract].copy()

if df_filtered.empty:
    print(f"No data found for moteid {moteid_to_extract}")
    exit()

# Select the desired attributes
df_extracted = df_filtered[attributes_to_extract]

# Define the output .csv file path
output_directory = r'C:\Users\yanzhang\Desktop\Research_Projects\TSL_customed\dataset_test'
output_file_name = f'moteid_{moteid_to_extract}_temp_volt.csv'
output_file_path = os.path.join(output_directory, output_file_name)

# Save the extracted DataFrame to a new .csv file
try:
    df_extracted.to_csv(output_file_path, index=False)
    print(f"Successfully extracted data to '{output_file_path}'")
except Exception as e:
    print(f"Error saving the extracted data: {e}")
