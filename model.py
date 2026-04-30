"""
Simple Linear Regression Model with TensorFlow
A minimal example demonstrating regression with TensorFlow/Keras
"""

# Suppress TensorFlow warnings and info messages
import os
import warnings

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Show only errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN

# Suppress Python warnings
warnings.filterwarnings('ignore')

# Import required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    """
    Plot training data, test data, and model predictions.
    
    Args:
        train_data: Training features
        train_labels: Training labels
        test_data: Test features
        test_labels: True test labels
        predictions: Model predictions on test data
    """
    plt.figure(figsize=(8, 6))
    
    # Plot data
    plt.scatter(train_data, train_labels, c='b', label='Training data', alpha=0.7)
    plt.scatter(test_data, test_labels, c='g', label='Testing data', alpha=0.7)
    plt.scatter(test_data, predictions, c='r', label='Predictions', alpha=0.7)
    
    # Formatting
    plt.title('Model Predictions vs Actual Values', fontsize=14, fontweight='bold')
    plt.xlabel('X values', fontsize=12)
    plt.ylabel('Y values', fontsize=12)
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save and show
    plt.tight_layout()
    plt.savefig('model_results.png', dpi=120, bbox_inches='tight')
    plt.show()


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics: MAE and MSE using NumPy.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        tuple: (mae, mse) rounded to 2 decimal places
    """
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    return float(mae), float(mse)


def save_metrics_to_file(mae, mse, filename='metrics.txt'):
    """
    Save metrics to a text file.
    
    Args:
        mae: Mean Absolute Error value
        mse: Mean Squared Error value
        filename: Output file name
    """
    with open(filename, 'w') as f:
        f.write("="*50 + "\n")
        f.write("MODEL PERFORMANCE METRICS\n")
        f.write("="*50 + "\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.2f}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write("="*50 + "\n")
    print(f"Metrics saved to {filename}")

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Create synthetic data (linear relationship)
# X values from -100 to 96 with step 4
X = np.arange(-100, 100, 4)  # 50 samples
# y = X + 10 (linear relationship with offset)
y = np.arange(-90, 110, 4)   # 50 samples

print(f"Dataset size: {len(X)} samples")
print(f"X shape before reshape: {X.shape}")
print(f"y shape: {y.shape}")

# IMPORTANT: Reshape data to 2D (samples, features)
# For a single feature, we need shape (n_samples, 1)
X = X.reshape(-1, 1)  # Reshape from (50,) to (50, 1)
y = y.reshape(-1, 1)  # Reshape from (50,) to (50, 1)

print(f"X shape after reshape: {X.shape}")
print(f"y shape after reshape: {y.shape}")

# Split data into train and test sets (80% train, 20% test)
split_idx = 40  # First 40 samples for training
X_train = X[:split_idx]  # Shape: (40, 1)
y_train = y[:split_idx]  # Shape: (40, 1)
X_test = X[split_idx:]   # Shape: (10, 1)
y_test = y[split_idx:]   # Shape: (10, 1)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# ============================================================================
# MODEL BUILDING
# ============================================================================

# Set random seed for reproducibility
tf.random.set_seed(42)

# Create a simple sequential model (single layer for linear regression)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,), name='linear_layer')
])

# Compile the model
model.compile(
    loss='mae',                        # Mean Absolute Error loss function
    optimizer='sgd',                   # Stochastic Gradient Descent
    metrics=['mae']                    # Track MAE during training
)

# Display model architecture
print("\nModel Architecture:")
model.summary()

# ============================================================================
# MODEL TRAINING
# ============================================================================

print("\n" + "="*50)
print("TRAINING MODEL")
print("="*50)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    verbose=1,                    # Show progress bar
    validation_split=0.2          # Use 20% of training data for validation
)

# ============================================================================
# MODEL EVALUATION
# ============================================================================

# Make predictions
y_preds = model.predict(X_test, verbose=0)
y_preds = y_preds.flatten()  # Flatten to 1D for plotting
y_test_flat = y_test.flatten()

# Plot results
plot_predictions(
    X_train.flatten(), y_train.flatten(),  # Flatten for plotting
    X_test.flatten(), y_test_flat, 
    y_preds
)

# Calculate metrics using NumPy
mae_value, mse_value = calculate_metrics(y_test_flat, y_preds)

# Display metrics
print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Mean Absolute Error (MAE): {mae_value:.2f}")
print(f"Mean Squared Error (MSE): {mse_value:.2f}")
print("="*50)

# Save metrics to file
save_metrics_to_file(mae_value, mse_value)

# ============================================================================
# Visualize training history
# ============================================================================

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss over epochs
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_title('Model Loss (MAE)', fontweight='bold')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Zoom into later epochs (excluding first 10 for better view)
axes[1].plot(history.history['loss'][10:], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'][10:], label='Validation Loss', linewidth=2)
axes[1].set_title('Loss (Epochs 10+)', fontweight='bold')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=120, bbox_inches='tight')
plt.show()

# ============================================================================
# SAVE MODEL
# ============================================================================

# Save the trained model
model.save('linear_regression_model.h5')
print("\nModel saved as 'linear_regression_model.h5'")

# Simple prediction example
sample_input = np.array([[50.0]])  # Note the double brackets for 2D shape
sample_prediction = model.predict(sample_input, verbose=0)
print(f"\nExample prediction: X={sample_input[0][0]:.0f} -> y={sample_prediction[0][0]:.2f}")

# Test the learned relationship
print("\n" + "="*50)
print("LEARNED RELATIONSHIP")
print("="*50)
# Get model weights
weights = model.get_weights()
if len(weights) >= 2:
    print(f"Weight (slope): {weights[0][0][0]:.4f}")
    print(f"Bias (intercept): {weights[1][0]:.4f}")
    print(f"Expected relationship: y = x + 10")
    print(f"Learned relationship: y = {weights[0][0][0]:.4f}*x + {weights[1][0]:.4f}")

print("\n" + "="*50)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*50)
