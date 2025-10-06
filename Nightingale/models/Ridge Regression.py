from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay
from feature_selection import features, arousal  # Replace with your actual module

# Initialize the Ridge Regression model
model = Ridge(alpha=0.1)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.99)  # Retain 99% variance
X_pca = pca.fit_transform(X_scaled)

# 10-Fold Cross-Validation
cv = KFold(n_splits=10, random_state=1, shuffle=True)

fold_r2_scores = []
fold_percentage_errors = []

for train_idx, test_idx in cv.split(X_pca):
    X_train, X_test = X_pca[train_idx], X_pca[test_idx]
    y_train, y_test = arousal[train_idx], arousal[test_idx]

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # R^2 score for each fold
    fold_r2 = r2_score(y_test, y_pred)
    fold_r2_scores.append(fold_r2)

    # Percentage Error for each fold
    fold_percentage_error = mean_absolute_percentage_error(y_test, y_pred) * 100
    fold_percentage_errors.append(fold_percentage_error)

# Display Results
print("10-Fold R^2 Scores:", fold_r2_scores)
print("Average R^2 Score:", np.mean(fold_r2_scores))
print("10-Fold Percentage Errors:", fold_percentage_errors)
print("Average Percentage Error:", np.mean(fold_percentage_errors))

# Cross-validated predictions
arousal_pred = cross_val_predict(model, X_pca, arousal, cv=cv)

# Visualization
# 1. Actual vs Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(arousal, arousal_pred, c=np.abs(arousal - arousal_pred), cmap='viridis', alpha=0.7)
plt.colorbar(label='Absolute Error')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()

# 2. Residuals vs Predicted Values
plt.figure(figsize=(8, 6))
residuals = arousal - arousal_pred
plt.scatter(arousal_pred, residuals, alpha=0.7, edgecolors='k')
plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()
