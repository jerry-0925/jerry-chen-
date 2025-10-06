from feature_selection import valence_features, valence
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(valence_features)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.98)  # Retain 98% variance
X_pca = pca.fit_transform(X_scaled)

# Custom wrapper for XGBoost
class CustomXGBRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.model = xgb.XGBRegressor(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_pca, valence, test_size=0.2, random_state=42)

# Instantiate the custom XGBRegressor
xgb_model = CustomXGBRegressor(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)

# Best hyperparameters from GridSearchCV
print("Best Hyperparameters from GridSearchCV:", grid_search.best_params_)

# Evaluate the best model using 10-fold cross-validation
best_model = grid_search.best_estimator_
kf = KFold(n_splits=10, shuffle=True, random_state=42)

fold_r2_scores = []
fold_percentage_errors = []

for train_idx, test_idx in kf.split(X_pca):
    X_train_fold, X_test_fold = X_pca[train_idx], X_pca[test_idx]
    y_train_fold, y_test_fold = valence[train_idx], valence[test_idx]

    best_model.fit(X_train_fold, y_train_fold)
    y_pred_fold = best_model.predict(X_test_fold)

    # R^2 score
    fold_r2 = r2_score(y_test_fold, y_pred_fold)
    fold_r2_scores.append(fold_r2)

    # Percentage error
    fold_percentage_error = mean_absolute_percentage_error(y_test_fold, y_pred_fold) * 100
    fold_percentage_errors.append(fold_percentage_error)

print("10-Fold Cross-Validation R^2 scores:", fold_r2_scores)
print("Mean R^2 score:", np.mean(fold_r2_scores))
print("10-Fold Percentage Errors:", fold_percentage_errors)
print("Mean Percentage Error:", np.mean(fold_percentage_errors))

# Test set evaluation
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

test_r2 = r2_score(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)
test_percentage_error = mean_absolute_percentage_error(y_test, y_pred) * 100

print("R^2 on test set:", test_r2)
print("Mean Squared Error on test set:", test_mse)
print("Mean Percentage Error on test set:", test_percentage_error)

# Visualizations
# 1. Actual vs Predicted
plt.figure(figsize=(8, 6))
absolute_errors = np.abs(y_test - y_pred)  # Calculate absolute errors
plt.scatter(y_test, y_pred, c=absolute_errors, cmap='viridis', alpha=0.7)
plt.colorbar(label='Absolute Error')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()

# 2. Residuals vs Predicted
residuals = y_test - y_pre89gd
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7, edgecolor='k')
plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

