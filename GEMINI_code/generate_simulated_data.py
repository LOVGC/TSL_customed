import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt

def simple_sine_simulator(t, frequency=0.001, amplitude=1.0, noise_level=0.2):
    """
    Generates a single sine wave with some random noise.
    """
    noise = np.random.normal(0, noise_level)
    return amplitude * np.sin(2 * np.pi * frequency * t) + noise

def multi_sine_simulator(t, noise_level=0.2):
    """
    Generates a sum of 6 sine waves with different frequencies and amplitudes, plus noise.
    """
    frequencies = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    amplitudes = [1.0, 0.5, 0.25, 0.12, 0.06, 0.03]
    
    result = 0.0
    for freq, amp in zip(frequencies, amplitudes):
        result += amp * np.sin(2 * np.pi * freq * t)
        
    noise = np.random.normal(0, noise_level)
    return result + noise

def random_process_simulator(t, scale=1.0, noise_std=0.5):
    """
    Generates a random value at each time step.
    """
    return scale * np.random.normal(0, noise_std)

def non_stationary_simulator(previous_value, drift=0.001, step_std=0.1):
    """
    Generates a single step of a non-stationary random walk with drift.
    """
    return previous_value + drift + np.random.normal(0, step_std)

SIMULATORS = {
    'simple_sine': simple_sine_simulator,
    'multi_sine': multi_sine_simulator,
    'random_process': random_process_simulator,
    'non_stationary': non_stationary_simulator,
}

def generate_data(output_path, simulator_name, num_points):
    """
    Generates simulated time series data and saves it to a CSV file.
    """
    simulator = SIMULATORS[simulator_name]
    start_date = datetime(2016, 7, 1, 0, 0, 0)
    
    dates = [start_date + timedelta(seconds=i) for i in range(num_points)]
    
    data_points = []
    if simulator_name in ['non_stationary']: # Handle stateful simulators
        current_value = 0.0
        for i in range(num_points):
            current_value = simulator(current_value)
            data_points.append(current_value)
    else: # Handle stateless simulators
        data_points = [simulator(i) for i in range(num_points)]
        
    df = pd.DataFrame({
        'date': dates,
        'data': data_points
    })
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['data'])
    plt.title(f'Simulated Time Series Data ({simulator_name})')
    plt.xlabel('Date')
    plt.ylabel('Data')
    plt.grid(True)
    plt.show()
    
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    df.to_csv(output_path, index=False)
    print(f"Generated {num_points} data points and saved to {output_path}")

if __name__ == '__main__':
    output_dir = 'dataset/simulated_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    simulator_to_use = 'non_stationary'
    num_points_to_generate = 20000
    
    output_file = os.path.join(output_dir, f'simulated_data_{simulator_to_use}.csv')
    
    generate_data(output_file, simulator_to_use, num_points_to_generate)
