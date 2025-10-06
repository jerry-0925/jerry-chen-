import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression  # For creating a sample dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from feature_selection import arousal_features, arousal

# Step 2: Ravel the target variable `valence` to ensure it's a 1D array
arousal = arousal.ravel()

# Step 3: Define the Gradient Boosting Regressor model
gbr = GradientBoostingRegressor(random_state=42)

gbr = GradientBoostingRegressor(random_state=42)

# Step 4: Perform 10-fold cross-validation and evaluate using R^2 score
cv_scores = cross_val_score(gbr, arousal_features, arousal, cv=10, scoring='r2')
# Step 4: Perform 10-fold cross-validation and evaluate using R^2 score
cv_scores = cross_val_score(gbr, arousal_features, arousal, cv=10, scoring='r2')

# Print the individual fold R^2 scores and the mean R^2 score
print("Individual fold R^2 scores:", cv_scores)
print("Mean R^2 score:", cv_scores.mean())
print("Standard deviation of R^2 scores:", cv_scores.std())

# Step 5: Hyperparameter tuning using GridSearchCV
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of boosting stages
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Shrinks the contribution of each tree
    'max_depth': [3, 5, 7],  # Depth of individual trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'subsample': [0.8, 0.9, 1.0]  # Fraction of samples used for fitting each tree
}

# Set up GridSearchCV with 10-fold cross-validation
grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=10, scoring='r2')

# Fit GridSearchCV to the data
grid_search.fit(arousal_features, arousal)

# Print the best hyperparameters and the best R^2 score
print("Best hyperparameters:", grid_search.best_params_)
print("Best R^2 score:", grid_search.best_score_)

# Step 6: Train the best model with the entire dataset using the best hyperparameters
best_model = grid_search.best_estimator_

# Train the best model on the entire dataset
best_model.fit(arousal_features, arousal)

# Step 7: Evaluate the model
# Predict using the best model
y_pred = best_model.predict(arousal_features)

# Evaluate performance with R^2 and Mean Squared Error
r2_score = best_model.score(arousal_features, arousal)
mse = mean_squared_error(arousal, y_pred)
mae = mean_absolute_error(arousal, y_pred)

print(f"Best model R^2 score: {r2_score}")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Step 8: Visualization
# 1. Actual vs Predicted
plt.figure(figsize=(8, 6))
absolute_errors = np.abs(arousal - y_pred)  # Calculate absolute errors
plt.scatter(arousal, y_pred, c=absolute_errors, cmap='viridis', alpha=0.7)
plt.colorbar(label='Absolute Error')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()

# 2. Residuals vs Predicted
residuals = arousal - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7, edgecolor='k')
plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Step 9: Percentage Error Calculation
percentage_error = 100 * np.abs((arousal - y_pred) / arousal)
mean_percentage_error = np.mean(percentage_error)

print(f"Mean Percentage Error: {mean_percentage_error:.2f}%")

# 3. Histogram of Percentage Errors
plt.figure(figsize=(8, 6))
plt.hist(percentage_error, bins=30, color='blue', alpha=0.7, edgecolor='k')
plt.title('Histogram of Percentage Errors')
plt.xlabel('Percentage Error (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
