import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def visualize_ett_data(csv_file_path, columns_to_visualize):
    """
    Reads a CSV file, visualizes specified columns as time series plots.
    Each column is plotted on a separate subplot.

    Args:
        csv_file_path (str): The path to the CSV file.
        columns_to_visualize (list): A list of column names to visualize.
                                     If empty, all columns except 'date' will be visualized.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if 'date' not in df.columns:
        print("Error: 'date' column not found in the CSV file.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    if not columns_to_visualize:
        # Visualize all columns except 'date'
        columns_to_visualize = [col for col in df.columns if col != 'date']
    else:
        # Validate if all specified columns exist
        missing_columns = [col for col in columns_to_visualize if col not in df.columns]
        if missing_columns:
            print(f"Error: The following columns were not found in the CSV file: {', '.join(missing_columns)}")
            return

    num_columns = len(columns_to_visualize)
    if num_columns == 0:
        print("No columns to visualize.")
        return

    fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(12, 4 * num_columns), sharex=True)

    # Ensure axes is always an array, even for a single subplot
    if num_columns == 1:
        axes = [axes]

    for i, col in enumerate(columns_to_visualize):
        axes[i].plot(df.index, df[col])
        axes[i].set_title(f'Time Series of {col}')
        axes[i].set_ylabel(col)
        axes[i].grid(True)

    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize time series data from a CSV file.")
    parser.add_argument(
        '--file',
        type=str,
        default=os.path.join('dataset', 'ETT-small', 'ETTh1.csv'),
        help="Path to the CSV file (default: dataset/ETT-small/ETTh1.csv)"
    )
    parser.add_argument(
        '--columns',
        nargs='*',  # 0 or more arguments
        default=[],
        help="Columns to visualize. If not specified, all non-date columns will be plotted. "
             "Example: --columns HUFL HULL"
    )

    args = parser.parse_args()
    visualize_ett_data(args.file, args.columns)
