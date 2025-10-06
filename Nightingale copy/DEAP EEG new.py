import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

data1 = pd.read_csv("/Users/jerrychen/PycharmProjects/music/master csv file - recovered.csv")
final_set = data1.values
data2 = pd.read_csv("/Users/jerrychen/PycharmProjects/music/20002 on.csv")
final_set = np.concatenate((final_set, data2.values), axis=0)
data3 = pd.read_csv("/Users/jerrychen/PycharmProjects/music/38778 on.csv")
final_set = np.concatenate((final_set, data3.values), axis=0)
fs = 128

combined_data = []
for i in range(1280):
    EEG = np.concatenate((final_set[40*i:40*i+1, 6:], final_set[40*i+16:40*i+17, 6:]), axis=0)
    EEG = np.concatenate((EEG, final_set[40*i+10:40*i+11, 6:]), axis=0)
    EEG = np.concatenate((EEG, final_set[40 * i + 28:40 * i + 29, 6:]), axis=0)
    print(len(EEG))
    EEG=np.asarray(EEG)

    for j in range(4):
        psd_data = {}
        # plt.subplot(2, 2, j+1)
        f, Pxx = signal.welch(EEG[j], fs, nperseg=512, noverlap=0.8)
        # plt.semilogy(f, Pxx)
        # plt.xlim(0, 45)  # Set x-axis limit to 0-100 Hz
        # plt.ylim(1e-5, 1e2)
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('PSD (V²/Hz)')
        # plt.title(f'Power Spectral Density (PSD) - {i}')
        # plt.grid()
        temp_df = pd.DataFrame({
            'SubjectID': i,
            'Channel': j,
            'Frequency': f,
            'PSD': Pxx
        })
        combined_data.append(temp_df)

    # plt.tight_layout()
    # plt.show()

all_subjects_df = pd.concat(combined_data, ignore_index=True)
# Save to CSV
all_subjects_df.to_csv('all_subjects_psd.csv', index=False)
print("Saved combined PSD data for all subjects.")