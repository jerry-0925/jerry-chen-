from sklearn.preprocessing import PolynomialFeatures
from feature_selection import valence_features
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
from feature_selection import valence
from sklearn.metrics import r2_score
model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                  ('linear', LinearRegression(fit_intercept=False))])
cv = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(model,valence_features,valence,scoring="r2",cv=cv,n_jobs=-1)
print(np.mean(np.abs(scores)))