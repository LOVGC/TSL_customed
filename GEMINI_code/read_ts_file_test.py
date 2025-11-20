from sktime.datasets import load_from_tsfile_to_dataframe
import numpy as np



X, y = load_from_tsfile_to_dataframe("E:\Research_Projects\TSL_customed\GEMINI_code\example.ts")


print(X)
