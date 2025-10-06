import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import numpy as np
import yasa


data1 = pd.read_csv("/Users/jerrychen/PycharmProjects/music/master csv file - recovered copy.csv")
final_set = data1.values

physiological = final_set[32:40 , 6:]
EEG = final_set[0:32 , 6:]
for i in range(1280):
    EEG = np.concatenate([EEG , final_set[40*i:40*i+32,6:]],axis=0)
    physiological = np.concatenate([physiological, final_set[40*i+32:40*i+40,6:]],axis=0)
print(len(EEG))
EEG=np.asarray(EEG)

total_data = np.empty((0, 8))

for i in range(544):
    data = yasa.bandpower(EEG[32*i:32*i+32, :], sf=128, ch_names=['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5','CP1','P3',
                                                    'P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4','F8',"FC6",'FC2',
                                                    'Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2'],bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 30, 'Beta'), (30, 40, 'Gamma')],relative=False)
    total_data = np.vstack([total_data, data])

np.savetxt("bandpower.csv", total_data, delimiter=",", fmt="%.5f")

print("Array saved to bandpower.csv")