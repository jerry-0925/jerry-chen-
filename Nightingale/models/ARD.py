from sklearn.model_selection import KFold
from sklearn.linear_model import ARDRegression
from feature_selection import valence_features
from feature_selection import valence
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
model = ARDRegression()
cv = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(model, valence_features, valence,scoring="r2", cv=cv, n_jobs=-1)
print(np.mean(np.abs(scores)))
