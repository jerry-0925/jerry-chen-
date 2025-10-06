import sklearn.linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoLars
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge as Ba
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.ensemble import StackingRegressor as St
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
estimators = [
    ("Ada",AdaBoostRegressor()),
    ("Forest",RandomForestRegressor()),
    ("Bag",BaggingRegressor()),
    ("Ridge",Ridge(alpha=0.1)),
    ("Lasso",LassoLars()),
    ("ElasticNet",ElasticNet()),
    ("Bayesian",Ba())
]
model = St(estimators=estimators,final_estimator=KernelRidge())
cv = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(model, valence_features, valence,scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)
print(np.mean(np.abs(scores)))