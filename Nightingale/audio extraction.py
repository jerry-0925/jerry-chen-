import opensmile
import pandas as pd

from pathlib import Path
import os
directory = Path('/Users/jerrychen/PycharmProjects/music/The other songs')

# Collect all file paths
files = [file_path for file_path in directory.iterdir() if file_path.is_file()]

# Use only the first 23 files
for file_path in files[:24]:
    if os.path.basename(file_path) != '.DS_Store':
        print(file_path)
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.emobase,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        y = smile.process_file(file_path)
        data_m = pd.DataFrame(y)
        data_m.to_csv(os.path.basename(file_path)+'.csv', sep=',', header=False, index=False)

# smile = opensmile.Smile(
#             feature_set=opensmile.FeatureSet.emobase,
#             feature_level=opensmile.FeatureLevel.Functionals,
#         )
# y = smile.process_file("/Users/jerrychen/PycharmProjects/music/music clips/love story MuVi.WAV")
# data_m = pd.DataFrame(y)
# data_m.to_csv("love story MuVi"+'.csv', sep=',', header=False, index=False)