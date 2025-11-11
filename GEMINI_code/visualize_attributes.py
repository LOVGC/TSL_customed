import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.preprocessing import StandardScaler

def visualize_attributes(moteid, attributes, scale):
    # Define the path to the input .csv file
    input_file_path = r'C:\Users\yanzhang\Desktop\Research_Projects\TSL_customed\dataset_test\data.csv'

    # Read the .csv file into a pandas DataFrame
    try:
        df = pd.read_csv(input_file_path)
    except Exception as e:
        print(f"Error reading the file: {e}")
        return

    # Filter the DataFrame for the specified moteid
    df_mote = df[df['moteid'] == moteid].copy()

    if df_mote.empty:
        print(f"No data found for moteid {moteid}")
        return

    # Convert date and time columns to a single datetime column
    try:
        df_mote['datetime'] = pd.to_datetime(df_mote['date'] + ' ' + df_mote['time'])
    except Exception as e:
        print(f"Error converting date and time columns: {e}")
        df_mote['datetime'] = df_mote.index

    # Sort by datetime
    df_mote = df_mote.sort_values(by='datetime')

    # Create the plot
    plt.figure(figsize=(15, 7))

    # Plot each specified attribute
    for attribute in attributes:
        if attribute in df_mote.columns:
            data_to_plot = df_mote[attribute]
            if scale:
                scaler = StandardScaler()
                # Reshape data for scaler and scale it
                scaled_data = scaler.fit_transform(data_to_plot.values.reshape(-1, 1))
                plt.plot(df_mote['datetime'], scaled_data, label=f'{attribute.capitalize()} (Scaled)')
            else:
                plt.plot(df_mote['datetime'], data_to_plot, label=attribute.capitalize())
        else:
            print(f"Warning: Attribute '{attribute}' not found in the data. Skipping.")

    plt.xlabel('Time')
    y_label = 'Scaled Value' if scale else 'Value'
    plt.ylabel(y_label)
    
    num_points = len(df_mote)
    attributes_str = ', '.join(a.capitalize() for a in attributes)
    title_suffix = ' (Scaled)' if scale else ''
    plt.title(f'{attributes_str} for Mote ID {moteid}{title_suffix} (Total Points: {num_points})')
    
    plt.legend()
    plt.grid(True)

    # Display the plot
    try:
        plt.show()
        print(f"Displayed the plot for Mote ID {moteid} and attributes {', '.join(attributes)}.")
    except Exception as e:
        print(f"Error displaying the plot: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize sensor data for a specific mote.')
    parser.add_argument('--moteid', type=int, required=True, help='The ID of the mote to visualize.')
    parser.add_argument('--attributes', nargs='+', required=True, 
                        choices=['temperature', 'humidity', 'light', 'voltage'],
                        help='A list of attributes to plot (e.g., temperature humidity).')
    parser.add_argument('--scale', action='store_true', help='If set, scale the attributes to mean 0 and std 1.')
    
    args = parser.parse_args()
    
    visualize_attributes(args.moteid, args.attributes, args.scale)