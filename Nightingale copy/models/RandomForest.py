import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load Data
data = pd.read_csv('/content/emobase features for DEAP and MuVi shortened newest with fixed scale.csv')

# Separate features and target
X = data.drop(['Valence', 'Arousal'], axis=1).values
y = data['Arousal'].values

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.98)  # Retain 98% variance
X_pca = pca.fit_transform(X_scaled)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the RandomForestRegressor model
rf_model = RandomForestRegressor(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)

# Best hyperparameters from GridSearchCV
print("Best Hyperparameters from GridSearchCV:", grid_search.best_params_)

# Evaluate the best model using 10-fold cross-validation
best_model = grid_search.best_estimator_
kf = KFold(n_splits=10, shuffle=True, random_state=42)

fold_r2_scores = []
fold_percentage_errors = []

for train_idx, test_idx in kf.split(X_scaled):
    X_train_fold, X_test_fold = X_scaled[train_idx], X_scaled[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    # Train and predict on each fold
    best_model.fit(X_train_fold, y_train_fold)
    y_pred_fold = best_model.predict(X_test_fold)

    # R^2 and percentage error for each fold
    fold_r2 = r2_score(y_test_fold, y_pred_fold)
    fold_r2_scores.append(fold_r2)

    fold_percentage_error = np.mean(np.abs((y_pred_fold - y_test_fold) / y_test_fold)) * 100
    fold_percentage_errors.append(fold_percentage_error)

# Display cross-validation results
print("10-Fold Cross-Validation R^2 scores:", fold_r2_scores)
print("Average R^2 score:", np.mean(fold_r2_scores))

print("10-Fold Percentage Errors:", fold_percentage_errors)
print("Average Percentage Error:", np.mean(fold_percentage_errors))

# Evaluate the model on the test set
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Test set evaluation metrics
test_r2 = r2_score(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)
percentage_error_test = np.mean(np.abs((y_pred - y_test) / y_test)) * 100

print("R^2 on test set:", test_r2)
print("Mean Squared Error on test set:", test_mse)
print("Mean Percentage Error on Test Set:", percentage_error_test)

# Visualize Actual vs. Predicted Values on Test Set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c=np.abs(y_test - y_pred), cmap='viridis', alpha=0.7)
plt.colorbar(label='Absolute Error')
plt.title('Actual vs Predicted Values with Error Color Coding')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()

# Visualize Percentage Error Distribution for Test Set
plt.figure(figsize=(8, 6))
plt.hist(np.abs((y_pred - y_test) / y_test) * 100, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Percentage Errors on Test Set')
plt.xlabel('Percentage Error (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
