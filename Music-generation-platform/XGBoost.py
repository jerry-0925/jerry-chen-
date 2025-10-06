
from feature_selection import valence_features, valence
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# Example: Load and preprocess your dataset
# Replace with actual data: X, y = your_data
# Example data creation for testing
# X, y = make_regression(n_samples=100, n_features=105, noise=0.1, random_state=42)
X, y = valence_features, valence
# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.98)  # Retain 95% variance
X_pca = pca.fit_transform(X_scaled)

# Custom wrapper for XGBoost to avoid the sklearn tagging error
# Custom wrapper for XGBoost
class CustomXGBRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.model = xgb.XGBRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    # Implementing get_params and set_params
    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self


# Load and preprocess the dataset
# Replace with actual data preparation
# X, y = your_data
# Example: X, y = make_regression(n_samples=100, n_features=105, noise=0.1, random_state=42)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Instantiate the custom XGBRegressor
xgb_model = CustomXGBRegressor(random_state=42)


# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2]  # Controls regularization
}
# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)

# Best hyperparameters from GridSearchCV
print("Best Hyperparameters from GridSearchCV:", grid_search.best_params_)

# Evaluate the best model using 10-fold cross-validation
best_model = grid_search.best_estimator_
cv_scores = cross_val_score(best_model, X_scaled, y, cv=10, scoring='r2', n_jobs=-1)

# Display the results
print("10-Fold Cross-Validation R^2 scores:", cv_scores)
print("Mean R^2 score from 10-fold cross-validation:", np.mean(cv_scores))

# Optionally, evaluate the model on the test set
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("R^2 on test set:", r2_score(y_test, y_pred))
print("Mean Squared Error on test set:", mean_squared_error(y_test, y_pred))