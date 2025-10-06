import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from feature_selection import features, valence
# Example: Load and preprocess your dataset
# Replace with actual data: X, y = your_data
# Example data creation for testing
# X, y = make_regression(n_samples=100, n_features=105, noise=0.1, random_state=42)
X, y = features, valence
# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.995)  # Retain 95% variance
X_pca = pca.fit_transform(X_scaled)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Instantiate the MLPRegressor
mlp_model = MLPRegressor(random_state=42, early_stopping=True)

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50), (150, 100, 50), (200, 150, 100, 50)],   # Size of hidden layers
    'activation': ['relu', 'tanh', 'logistic'],                  # Activation functions
    'solver': ['adam', 'sgd'],                                   # Optimizer
    'learning_rate': ['constant', 'invscaling', 'adaptive'],     # Learning rate schedules
    'alpha': [0.0001, 0.001, 0.01],                              # Regularization parameter
    'max_iter': [500, 1000, 1500, 2000]                                # Maximum iterations
}

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(mlp_model, param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)

#    Best hyperparameters from GridSearchCV
print("Best Hyperparameters from GridSearchCV:", grid_search.best_params_)

# Evaluate the best model using 10-fold cross-validation
best_model = grid_search.best_estimator_
cv_scores = cross_val_score(best_model, X_pca, y, cv=10, scoring='r2', n_jobs=-1)

# Display the results
print("10-Fold Cross-Validation R^2 scores:", cv_scores)
print("Mean R^2 score from 10-fold cross-validation:", np.mean(cv_scores))

# Optionally, evaluate the model on the test set
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("R^2 on test set:", r2_score(y_test, y_pred))
print("Mean Squared Error on test set:", mean_squared_error(y_test, y_pred))
