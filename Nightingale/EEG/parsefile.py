import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import iirnotch, filtfilt, butter, hilbert
from scipy.stats import pearsonr

# Load the data
file_path = "/Users/jerrychen/PycharmProjects/music/EEG/OpenBCI-RAW-2025-01-31_15-23-55.txt"
with open(file_path, 'r') as file:
    lines = file.readlines()
data_start = next(i for i, line in enumerate(lines) if not line.startswith('%'))
df = pd.read_csv(file_path, skiprows=data_start, header=None, dtype=str)

column_names = ["Index"] + [f"Ch{i+1}" for i in range(4)] + ["Aux1", "Aux2"] + [f"Other{i+1}" for i in range(16)] + ["Timestamp", "Other", "Time"]
df.columns = column_names

channels = [f"Ch{i+1}" for i in range(4)]
eeg_channels = [f"Ch{i+1}" for i in range(4)]
df = df[channels]
df = df.iloc[1:, :]

# Save DataFrame as CSV
output_path = "/Users/jerrychen/PycharmProjects/music/EEG/processed_eeg.csv"
df.to_csv(output_path, index=False)

print(f"Data saved as CSV at: {output_path}")

