# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load and Normalize Data
data = pd.read_csv('/content/emobase features for DEAP and MuVi shortened newest with fixed scale.csv')

# Separate features and target
X = data.drop(['Valence', 'Arousal'], axis=1).values
y = data['Valence'].values

# Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Reshape target for compatibility
y = y.reshape(-1, 1)


# Function to Create Sequences
def create_sequences(X, y, timesteps):
    X_seq, y_seq = [], []
    for i in range(len(X) - timesteps):
        X_seq.append(X[i:i + timesteps])
        y_seq.append(y[i + timesteps])
    return np.array(X_seq), np.array(y_seq)


# Parameters
timesteps = 10  # Number of timesteps for the sequence

# Generate Sequences
X_seq, y_seq = create_sequences(X, y, timesteps)

# Split into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)


# Build the RNN Model
def build_model():
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=False, input_shape=(timesteps, X_train.shape[2])),
        Dropout(0.2),  # Regularization
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


# K-Fold Cross-Validation Setup
kf = KFold(n_splits=10, shuffle=True, random_state=42)


# Function to Calculate R² Score for Each Fold
def r2_cross_val(model, X_train, y_train, kf):
    r2_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Train the model
        model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, verbose=0)

        # Predict and calculate R² score
        y_pred_fold = model.predict(X_val_fold)
        r2 = r2_score(y_val_fold, y_pred_fold)
        r2_scores.append(r2)

    return np.mean(r2_scores)


# Evaluate with K-Fold Cross-Validation
model = build_model()
r2_mean = r2_cross_val(model, X_train, y_train, kf)
print(f"Mean R² from 10-Fold Cross-Validation: {r2_mean}")

# Train the Model on the full training set
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    verbose=1
)

# Evaluate the Model on the Test Set
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Make Predictions
y_pred = model.predict(X_test)

# Save the Model
model.save('rnn_regression_model.h5')

# Visualize Training and Validation Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Compare Predictions vs Actual Values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()
