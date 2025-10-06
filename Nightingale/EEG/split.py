import pandas as pd
import numpy as np

# Load processed CSV
input_path = "/Users/jerrychen/PycharmProjects/music/EEG/processed_eeg.csv"
df = pd.read_csv(input_path)

# Convert to numeric (important for splitting)
df = df.apply(pd.to_numeric)

# Reduce length by half
half_len = len(df) // 2
df = df.iloc[:half_len, :]

# Split each channel into two (simulate by taking even/odd indices)
split_df = pd.DataFrame()
for ch in df.columns:
    split_df[f"{ch}_1"] = df[ch].iloc[::2].reset_index(drop=True)
    split_df[f"{ch}_2"] = df[ch].iloc[1::2].reset_index(drop=True)

# Make sure both halves have the same length
min_len = min(len(split_df[c]) for c in split_df.columns)
split_df = split_df.iloc[:min_len, :]

# Save the new split CSV
output_split_path = "/Users/jerrychen/PycharmProjects/music/EEG/processed_eeg_split.csv"
split_df.to_csv(output_split_path, index=False)

print(f"Split and halved data saved as CSV at: {output_split_path}")
