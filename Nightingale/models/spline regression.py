from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn import linear_model
from feature_selection import arousal_features, arousal
from sklearn.preprocessing import SplineTransformer, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# Define the model pipeline
model = Pipeline([
    ('spline', SplineTransformer(n_knots=20, degree=3)),
    ('linear', linear_model.RidgeCV())
])

# Standardize the features
scaler = RobustScaler()
arousal_features_scaled = scaler.fit_transform(arousal_features)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.99)  # Retain 99% variance
arousal_features_pca = pca.fit_transform(arousal_features_scaled)

# Perform K-Fold Cross Validation
cv = KFold(n_splits=10, random_state=1, shuffle=True)
r2_scores = []
percentage_errors = []

# Cross-validation with percentage error calculation
for train_idx, test_idx in cv.split(arousal_features_pca):
    X_train, X_test = arousal_features_pca[train_idx], arousal_features_pca[test_idx]
    y_train, y_test = arousal[train_idx], arousal[test_idx]

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate R^2 score
    r2_scores.append(r2_score(y_test, y_pred))

    # Calculate percentage error for this fold
    fold_percentage_error = mean_absolute_percentage_error(y_test, y_pred) * 100
    percentage_errors.append(fold_percentage_error)

# Print cross-validation results
print("10-Fold Cross-Validation R^2 scores:", r2_scores)
print("Average R^2 score:", np.mean(r2_scores))
print("10-Fold Percentage Errors (%):", percentage_errors)
print("Average Percentage Error (%):", np.mean(percentage_errors))

# Generate cross-validated predictions
arousal_pred = cross_val_predict(model, arousal_features_pca, arousal, cv=cv)

# Visualization: Actual vs. Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(arousal, arousal_pred, c=np.abs(arousal - arousal_pred), cmap='viridis', alpha=0.7)
plt.colorbar(label='Absolute Error')
plt.title('Actual vs Predicted Values with Error Color Coding')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)
plt.show()

# Visualization: Residuals vs. Predicted Values
residuals = arousal - arousal_pred
plt.figure(figsize=(8, 6))
plt.scatter(arousal_pred, residuals, alpha=0.7, color='skyblue', edgecolor='black')
plt.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()
