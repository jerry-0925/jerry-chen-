import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from feature_selection import features, valence
from sklearn.decomposition import PCA
# Example: Load and preprocess your dataset
# Replace with actual data: X, y = your_data
# Example data creation for testing
# X, y = make_regression(n_samples=100, n_features=105, noise=0.1, random_state=42)
X, y = features, valence

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.96)
X_pca = pca.fit_transform(X_scaled)
# Split the dataset (For final test evaluation, after cross-validation)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Instantiate the RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],         # Number of trees in the forest
    'max_depth': [10, 20, None],              # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],          # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],            # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2'] # Number of features to consider when looking for the best split
}

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)

# Best hyperparameters from GridSearchCV
print("Best Hyperparameters from GridSearchCV:", grid_search.best_params_)

# Evaluate the best model using 10-fold cross-validation
best_rf_model = grid_search.best_estimator_
cv_scores = cross_val_score(best_rf_model, X_scaled, y, cv=10, scoring='r2', n_jobs=-1)

# Display the results
print("10-Fold Cross-Validation R^2 scores:", cv_scores)
print("Mean R^2 score from 10-fold cross-validation:", np.mean(cv_scores))

# Optionally, evaluate the model on the test set
best_rf_model.fit(X_train, y_train)
y_pred = best_rf_model.predict(X_test)

print("R^2 on test set:", r2_score(y_test, y_pred))
print("Mean Squared Error on test set:", mean_squared_error(y_test, y_pred))
