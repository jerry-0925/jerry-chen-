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

channels = [f"Ch{i+1}" for i in range(4)] + ['Time']
eeg_channels = [f"Ch{i+1}" for i in range(4)]
df = df[channels]
df = df.iloc[1:, :]

df["Time"] = df["Time"].str.strip()
df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S.%f")
df.dropna(subset=["Time"], inplace=True)

df["Elapsed_Time"] = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds()
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

fs = 1600  # Sampling rate

# --- Filtering functions ---
def notch_filter(data, freq=60.0, fs=1600, quality=30.0):
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, data)

def highpass_filter(data, cutoff=0.5, fs=1600, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, cutoff / nyq, btype='high')
    return signal.filtfilt(b, a, data)

def lowpass_filter(data, cutoff=100, fs=1600, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, cutoff / nyq, btype='low')
    return signal.filtfilt(b, a, data)

# --- Safe bandpass filter with length check ---
def safe_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    padlen = 3 * max(len(a), len(b))
    if len(data) <= padlen:
        # Skip filtering or pad (optional)
        return data  # or: return np.zeros_like(data)
    return filtfilt(b, a, data)

# --- Apply filtering to EEG channels ---
for ch in eeg_channels:
    df[ch] = notch_filter(df[ch].values, freq=60.0, fs=fs)
    df[ch] = lowpass_filter(highpass_filter(df[ch].values, cutoff=0.5, fs=fs, order=4), cutoff=100, fs=fs, order=4)

# --- Time filtering ---
start_time = 265
end_time = 325
filtered_df = df[(df["Elapsed_Time"] >= start_time) & (df["Elapsed_Time"] <= end_time)]


def remove_noise_segments(df, time_column="Elapsed_Time", segments_to_remove=[]):
    """
    Remove segments from the DataFrame where time falls within specified intervals.

    Parameters:
    - df: Input DataFrame
    - time_column: Name of the column containing elapsed time
    - segments_to_remove: List of (start_time, end_time) tuples

    Returns:
    - Filtered DataFrame with noise segments removed
    """
    for start, end in segments_to_remove:
        df = df[~df[time_column].between(start, end)]
    return df.reset_index(drop=True)

noise_intervals = [(292,295)]
filtered_df = remove_noise_segments(filtered_df, segments_to_remove=noise_intervals)


# --- Band definitions ---
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100)
}

#
# overall_corrs = []
# for ch in range(theta_env.shape[0]):  # channel-wise
#     corr, _ = pearsonr(theta_env[ch], gamma_env[ch])
#     overall_corrs.append(corr)
#
# # Print or log the per-channel and average correlation
# for i, ch in enumerate(eeg_channels):
#     print(f"Theta-Gamma correlation for {ch}: {overall_corrs[i]:.4f}")
#
# average_corr = np.mean(overall_corrs)
# print(f"\nAverage Theta-Gamma correlation: {average_corr:.4f}")
#
# plt.figure(figsize=(8, 4))
# plt.bar(eeg_channels, overall_corrs, color='purple')
# plt.title("Overall Theta-Gamma Correlation per Channel")
# plt.ylabel("Correlation")
# plt.grid(True)
# plt.ylim(-1, 1)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

# --- Bandpass and amplitude envelope ---
filtered = {}
envelopes = {}
for band, (low, high) in bands.items():
    order = 2 if low < 1.0 else 4  # Lower order for low-freq bands
    filtered[band] = np.array([
        safe_bandpass_filter(np.asarray(ch).flatten(), low, high, fs, order=order)
        for ch in filtered_df[eeg_channels].values.T
    ])
    envelopes[band] = np.abs(hilbert(filtered[band], axis=1))

# --- Theta-Gamma correlation ---
theta_env = envelopes['theta']
gamma_env = envelopes['gamma']

#
# --- Total power calculation per band ---
band_powers = {
    band: np.sum(env**2, axis=1)
    for band, env in envelopes.items()
}
# ==== 2. Modulation Index (PAC) Calculation ====
def compute_MI(theta_phase, gamma_amp, nbins=18):
    phase_bins = np.linspace(-np.pi, np.pi, nbins + 1)
    bin_means = np.zeros(nbins)

    for i in range(nbins):
        inds = np.where((theta_phase >= phase_bins[i]) & (theta_phase < phase_bins[i+1]))[0]
        bin_means[i] = np.mean(gamma_amp[inds]) if len(inds) > 0 else 0

    bin_means /= np.sum(bin_means)  # Normalize
    entropy = -np.sum(bin_means * np.log(bin_means + 1e-10))
    uniform_entropy = np.log(nbins)
    MI = (uniform_entropy - entropy) / uniform_entropy
    return MI, bin_means

# ==== 3. Main PAC Workflow ====
def pac_analysis(theta_env, gamma_env, nbins=18):
    mi_values = []
    phase_amp_distributions = []

    for ch in range(theta_env.shape[0]):
        # Get theta phase and gamma amplitude for this channel
        theta_phase = np.angle(hilbert(theta_env[ch]))
        gamma_amp = np.abs(hilbert(gamma_env[ch]))

        # Compute PAC
        mi, dist = compute_MI(theta_phase, gamma_amp, nbins)
        mi_values.append(mi)
        phase_amp_distributions.append(dist)

        print(f"Channel {ch+1}: MI = {mi:.4f}")

    return np.array(mi_values), np.array(phase_amp_distributions)

# ==== 4. Combined Visualization ====
def plot_pac_distribution(distributions, nbins=18):
    bin_edges = np.linspace(-np.pi, np.pi, nbins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    n_channels = len(distributions)

    fig, axs = plt.subplots(1, n_channels, subplot_kw={'projection': 'polar'}, figsize=(4 * n_channels, 4))
    if n_channels == 1:
        axs = [axs]

    for ch, dist in enumerate(distributions):
        if len(dist) != nbins:
            print(f"Skipping channel {ch+1} due to shape mismatch: {len(dist)} != {nbins}")
            continue
        axs[ch].plot(bin_centers, dist)
        axs[ch].set_title(f"Channel {ch+1}")
        axs[ch].tick_params(labelsize=6)
        axs[ch].set_yticks(axs[ch].get_yticks()[::2])

    plt.suptitle("Theta Phase vs Gamma Amplitude (PAC)")
    plt.tight_layout()
    plt.savefig("pac_all_channels.png", dpi=600, bbox_inches='tight')
    plt.show()


mi_vals, pac_dists = pac_analysis(theta_env, gamma_env, 18)


# Visualize
plot_pac_distribution(pac_dists)

# # Correlation plot
# axs[0].plot(filtered_df["Elapsed_Time"].values[:theta_gamma_corr.shape[0]], theta_gamma_corr, label="Theta-Gamma Corr")
# axs[0].set_title("Theta-Gamma Correlation Over Time")
# axs[0].set_xlabel("Time (s)")
# axs[0].set_ylabel("Correlation")
# axs[0].legend()
# axs[0].grid(True)

# Bar plot: total power per channel per band
channels = eeg_channels
x = np.arange(len(channels))
bar_width = 0.15

fig, axs = plt.subplots(figsize=(10, 6), constrained_layout=True)
for i, (band, power_array) in enumerate(band_powers.items()):
    axs.bar(x + i * bar_width, power_array, width=bar_width, label=band)

axs.set_xticks(x + bar_width * 2)
axs.set_xticklabels(channels, fontsize=14)
axs.set_title("Total Band Power per Channel",fontsize=18)
axs.set_ylabel("Power",fontsize=16)
axs.tick_params(axis='y', labelsize=14)  # Y-axis ticks font size
axs.legend(fontsize=14)
axs.grid(False)

plt.savefig("band_power_plot.png", dpi=1000)
plt.show()

# for i, ch in enumerate(eeg_channels):
#     plt.subplot(2, 2, i+1)  # Create a 2x2 grid of subplots
#     f, t, Sxx = signal.spectrogram(df[ch], fs)
#     plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
#     plt.ylabel('Frequency (Hz)')
#     plt.xlabel('Time (s)')
#     plt.title(f'Spectrogram of {ch}')
#     plt.colorbar(label='Power (dB)')
# plt.tight_layout()
# plt.show()
#
#
# Plot Power Spectral Density (PSD) for all EEG channels
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# Assume eeg_channels is a list of channel names, and df contains the data
channel = eeg_channels[0]  # First channel

# plt.figure(figsize=(10, 6))
#
# # Compute PSD using Welch’s method
# f, Pxx = signal.welch(df[channel], fs, nfft=1024, noverlap=0.8)
#
# # Plot
# plt.semilogy(f, Pxx)
# plt.xlim(0, 45)
# plt.ylim(1e-3, 1e2)
# plt.xlabel('Frequency (Hz)', fontsize=26)
# plt.ylabel('PSD (V²/Hz)', fontsize=26)
# plt.grid(True)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
#
# plt.tight_layout()
# plt.show()
#
# # Optionally convert and print PSD data
# Pxx_df = pd.DataFrame(Pxx)
# f_df = pd.DataFrame(f)
# print(Pxx_df)


# Filter data between 18s and 78s

# Plot the EEG signals over the selected time window
plt.figure(figsize=(12, 6))
for i, ch in enumerate(eeg_channels):
    plt.plot(filtered_df["Elapsed_Time"], filtered_df[ch] + i * 100, label=f'Channel {ch}')  # Offset for visibility

plt.xlabel("Elapsed Time (s)")
plt.ylabel("EEG Signal (µV)")
plt.title("Filtered EEG Time-Series (247s to 307s)")
plt.legend()
plt.grid(True)
plt.show()
#
# freq_bands = {
#     "Delta": (0.5, 4),
#     "Theta": (4, 8),
#     "Alpha": (8, 13),
#     "Beta": (13, 30),
#     "Gamma": (30, 45),
# }
#
# # Compute PSD and extract band power
# band_power = {ch: {} for ch in eeg_channels}
#
# plt.figure(figsize=(12, 8))
# for i, ch in enumerate(eeg_channels):
#     f, Pxx = signal.welch(df[ch], fs, nfft=512, noverlap=0.8)
#
#     # Compute band power
#     for band, (low, high) in freq_bands.items():
#         idx = np.logical_and(f >= low, f <= high)
#         band_power[ch][band] = np.trapz(Pxx[idx], f[idx])  # Integrate the PSD
#
# # Display band power results
# band_power_df = pd.DataFrame(band_power)
# print("Power Spectral Density over Frequency Bands:")
# print(band_power_df)

# import networkx as nx
# import seaborn as sns
#
# # Compute Pearson correlation between EEG channels
# corr_matrix = df[eeg_channels].corr()
#
# # Create a graph
# G = nx.Graph()
#
# # Add nodes (EEG channels)
# for ch in eeg_channels:
#     G.add_node(ch)
#
# # Add edges based on correlation strength (threshold > 0.6)
# threshold = 0.2  # Adjust this threshold to control edge density
# for i in range(len(eeg_channels)):
#     for j in range(i + 1, len(eeg_channels)):  # Avoid duplicate edges
#         if abs(corr_matrix.iloc[i, j]) > threshold:
#             G.add_edge(eeg_channels[i], eeg_channels[j], weight=corr_matrix.iloc[i, j])
#
# # Plot Connectivity Graph
# plt.figure(figsize=(6, 6))
# pos = nx.circular_layout(G)  # Circular layout for clarity
# edges = G.edges(data=True)
# weights = [d["weight"] for (u, v, d) in edges]
#
# nx.draw(G, pos, with_labels=True, node_size=2000, font_size=12,
#         edge_color=weights, edge_cmap=plt.cm.Blues, width=2)
#
# plt.title("Brain Connectivity Graph (Pearson Correlation)")
# plt.show()
#
# channel_names = ["Fp1", "P3", "Fp2", "P4"]  # Custom EEG channel names
# corr_matrix = np.array(corr_matrix)
#
# # Convert to DataFrame for labeling
# df_corr = pd.DataFrame(corr_matrix, index=channel_names, columns=channel_names)
#
# # Plot heatmap
# plt.figure(figsize=(6, 5))
# sns.heatmap(df_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, xticklabels=channel_names, yticklabels=channel_names)
# plt.title("EEG Connectivity Heatmap")
# plt.show()
