import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import numpy as np

data = pd.read_csv("1.csv")
array = data.values
ids = np.array(array[0,:])
ids.shape = (1,995)
valence = array[:,0:1]
arousal = array[:,1:2]
features = array[:,2:996]

# test = SelectKBest(score_func=f_regression, k=100)
# fit = test.fit(features, valence)
# index = test.get_support()
# valence_features = features[:,index]
# np.save("valence_select1",index)
# index = np.insert(index,0,False)
# index = np.insert(index,0,False)
# valence_ids = data.columns[index]
#
# test2 = SelectKBest(score_func=f_regression, k=100)
# fit2= test2.fit(features, arousal)
# index2 = test2.get_support()
# arousal_features = features[:,index2]
# np.save("arousal_select1",index2)
# index2 = np.insert(index2,0,False)
# index2 = np.insert(index2,0,False)
# arousal_ids = data.columns[index2]
