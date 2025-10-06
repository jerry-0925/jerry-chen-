from sklearn.linear_model import GammaRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from feature_selection import valence_features
from feature_selection import valence
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
model = GammaRegressor()
cv = KFold(n_splits=10)
scores = cross_val_score(model, valence_features, valence, scoring="r2", cv=cv, n_jobs=-1)
print(np.mean(np.abs(scores)))
