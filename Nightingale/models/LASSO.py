from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from feature_selection import arousal_features
from feature_selection import arousal
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
model = Lasso(alpha=1)
cv = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(model, arousal_features, arousal,scoring="r2", cv=cv, n_jobs=-1)
print(np.mean(np.abs(scores)))
