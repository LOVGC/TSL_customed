import pandas as pd
import os

# Define the path to the input .txt file
input_file_path = r'C:\Users\yanzhang\Desktop\Research_Projects\TSL_customed\dataset_test\data.txt'

# Define the column names based on the description
column_names = [
    'date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage'
]

# Read the .txt file into a pandas DataFrame
# Assuming the delimiter is whitespace, and there's no header
try:
    df = pd.read_csv(input_file_path, sep=r'\s+', header=None, names=column_names, engine='python')
except Exception as e:
    print(f"Error reading the file: {e}")
    exit()

# Define the output .csv file path
output_directory = r'C:\Users\yanzhang\Desktop\Research_Projects\TSL_customed\dataset_test'
output_file_name = 'data.csv'
output_file_path = os.path.join(output_directory, output_file_name)

# Save the DataFrame to a .csv file
try:
    df.to_csv(output_file_path, index=False)
    print(f"Successfully converted '{input_file_path}' to '{output_file_path}'")
except Exception as e:
    print(f"Error saving the CSV file: {e}")
