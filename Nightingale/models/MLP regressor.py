import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from feature_selection import valence_features,valence
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

# Step 1: Create or load the dataset
# Create a synthetic regression dataset (1000 samples, 105 features)
X, y = valence_features, valence
y = y.ravel()
# Step 2: Standardize the data (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Define the initial MLP Regressor
mlp = MLPRegressor(random_state=42, max_iter=500)

# Step 4: 10-Fold Cross-Validation with MLP Regressor
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation to get an estimate of the model's performance
cv_scores = cross_val_score(mlp, X_scaled, y, cv=cv, scoring='r2')

# Print the individual fold R^2 scores and the mean R^2 score
print("10-Fold Cross-Validation R^2 scores:", cv_scores)
print("Mean R^2 score from 10-fold CV:", cv_scores.mean())
print("Standard deviation of R^2 scores:", cv_scores.std())

# Step 5: Hyperparameter tuning using GridSearchCV
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # Single or double-layer networks
    'activation': ['relu', 'tanh'],  # Activation functions
    'solver': ['adam', 'sgd'],  # Solvers for optimization
    'learning_rate': ['constant', 'adaptive'],  # Learning rate schedule
    'alpha': [0.0001, 0.001, 0.01],  # L2 regularization
}

# Step 6: Hyperparameter tuning with GridSearchCV
grid_search = GridSearchCV(mlp, param_grid, cv=10, scoring='r2', n_jobs=-1)

# Fit GridSearchCV to the training data
grid_search.fit(X_scaled, y)

# Print the best hyperparameters and the best R^2 score from GridSearchCV
print("Best Hyperparameters from GridSearchCV:", grid_search.best_params_)
print("Best R^2 score from GridSearchCV:", grid_search.best_score_)

# Step 7: Evaluate the optimized model using the best hyperparameters
best_mlp = grid_search.best_estimator_

# Step 8: 10-Fold Cross-Validation with the optimized MLP model
cv_scores_best = cross_val_score(best_mlp, X_scaled, y, cv=cv, scoring='r2')

# Print the individual fold R^2 scores and the mean R^2 score for the optimized model
print("\nOptimized Model - 10-Fold Cross-Validation R^2 scores:", cv_scores_best)
print("Optimized Model - Mean R^2 score from 10-fold CV:", cv_scores_best.mean())
print("Optimized Model - Standard deviation of R^2 scores:", cv_scores_best.std())

# Step 9: Final Model Evaluation
# Train the best model on the full dataset
best_mlp.fit(X_scaled, y)

# Predict on the same dataset
y_pred = best_mlp.predict(X_scaled)

# Evaluate performance with R^2 and Mean Squared Error (MSE)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"\nFinal Optimized Model R^2 score: {r2}")
print(f"Final Optimized Model Mean Squared Error: {mse}")

# Step 10: Optional - Visualizing Predicted vs Actual Values
plt.scatter(y, y_pred, color='blue', label='Predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Full Dataset)')
plt.legend()
plt.show()
