"""
Generates synthetic data and trains a simple neural network for linear
regression. When this runs through the pipeline it saves output files
that get uploaded as artefacts in GitHub Actions.
"""

import os
import warnings

# these two lines stop tensorflow from printing a wall of info messages
# every time the script runs - keeps the pipeline logs clean and readable.
# without these, tensorflow logs things like cuda warnings and device info
# which just clutters the output and makes it harder to see what's happening
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # needed for headless environments like GitHub Actions
                       # (there's no display available on the runner, so the
                       # default matplotlib backend would crash without this)
import matplotlib.pyplot as plt


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    """
    Scatter plot showing training data, actual test values, and predictions
    side by side. Makes it easy to see at a glance whether the model is
    predicting sensibly - basically a quick visual check without having to
    look at the raw numbers.

    Three colours are used so each set of points is easy to distinguish:
    blue for training data, green for actual test values, red for predictions.
    If the red and green points are close together the model is doing well.
    """
    plt.figure(figsize=(9, 6))

    plt.scatter(train_data, train_labels, c='steelblue',
                label='Training data', alpha=0.7, s=50)
    plt.scatter(test_data, test_labels, c='seagreen',
                label='Test data (actual)', alpha=0.85, s=65)
    plt.scatter(test_data, predictions, c='tomato',
                label='Predictions', alpha=0.85, s=65, marker='x', linewidths=2)

    plt.title('Model Predictions vs Actual Values', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('X values', fontsize=12)
    plt.ylabel('Y values', fontsize=12)
    plt.legend(frameon=True, shadow=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # save to file rather than showing - plt.show() would fail in CI
    # since there's no display, so savefig is the right approach here
    plt.savefig('model_results.png', dpi=150, bbox_inches='tight')
    plt.close()  # close after saving so memory doesn't build up across plots
    print("Saved: model_results.png")


def plot_training_history(history):
    """
    Two panel loss curve - left shows all epochs, right zooms in from
    epoch 10 onwards. The reason for the two panels is that the first
    few epochs tend to have really high loss which squashes the rest of
    the chart and makes it hard to see the convergence behaviour, so the
    zoomed panel is more useful for actually checking whether the model
    settled properly.

    If the training and validation loss both decrease and stay close
    together the model is learning well without overfitting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left panel - full picture of training from start to finish
    axes[0].plot(history.history['loss'], label='Training Loss',
                 linewidth=2, color='steelblue')
    axes[0].plot(history.history['val_loss'], label='Validation Loss',
                 linewidth=2, color='tomato')
    axes[0].set_title('Model Loss (All Epochs)', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MAE Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # right panel - zoomed in after the initial high-loss phase settles
    axes[1].plot(history.history['loss'][10:], label='Training Loss',
                 linewidth=2, color='steelblue')
    axes[1].plot(history.history['val_loss'][10:], label='Validation Loss',
                 linewidth=2, color='tomato')
    axes[1].set_title('Loss (Epochs 10 onwards)', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: training_history.png")


def calculate_metrics(y_true, y_pred):
    """
    Calculate MAE, MSE and RMSE on the test set.

    Using multiple metrics rather than just one is generally a good idea -
    a single number doesn't always give the full picture of how a model
    is actually performing in practice. for example a model could have a
    low average error but still make a few really bad predictions, which
    MSE would catch but MAE might understate.

    MAE  - average size of errors, easy to interpret in real units
    MSE  - penalises larger errors more heavily than MAE does
    RMSE - square root of MSE, same units as the target so a bit more
           intuitive to read than MSE on its own
    """
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    return mae, mse, rmse


def save_metrics(mae, mse, rmse, weights, filename='metrics.txt'):
    """
    Saves metrics and some notes to a text file. This gets uploaded as a
    pipeline artefact so it can be downloaded from the GitHub Actions UI
    without needing to re-run anything - useful for checking results after
    the fact without having to trigger a whole new pipeline run.

    One thing worth noting here - logging the learned weight and bias means
    anyone reading this file can understand what the model learned without
    opening the code. for a single dense layer it's just y = W*x + b, so
    you can basically read off the full model from this file. this is what
    interpretability looks like in practice for a simple model like this.
    """
    with open(filename, 'w') as f:
        f.write("CL07_G03 - ML Pipeline Model Metrics\n")
        f.write("COS40007 Applied Project Task 2\n\n")

        f.write("Evaluation Metrics (test set)\n")
        f.write(f"  MAE  : {mae:.4f}\n")
        f.write(f"  MSE  : {mse:.4f}\n")
        f.write(f"  RMSE : {rmse:.4f}\n\n")

        # log the learned parameters alongside the metrics so the file
        # tells the full story of what the model learned, not just how
        # well it performed numerically
        if len(weights) >= 2:
            slope     = weights[0][0][0]
            intercept = weights[1][0]
            f.write("Learned Model Parameters\n")
            f.write(f"  Weight (slope)   : {slope:.4f}\n")
            f.write(f"  Bias (intercept) : {intercept:.4f}\n")
            f.write(f"  Learned formula  : y = {slope:.4f}*x + {intercept:.4f}\n")
            f.write(f"  Expected formula : y = 1.0000*x + 10.0000\n\n")

        f.write("Responsible AI Considerations\n\n")
        f.write("  Interpretability - the model uses a single Dense layer so it\n")
        f.write("  stays fully interpretable. the weight and bias above explain\n")
        f.write("  every prediction without needing to dig into the code.\n\n")
        f.write("  Transparency - data is synthetic (y = x + 10, evenly spaced)\n")
        f.write("  with no sampling bias. all design decisions are documented in\n")
        f.write("  model.py so results are fully reproducible.\n\n")
        f.write("  Accountability - GitHub Actions keeps a timestamped log of\n")
        f.write("  every training run tied to a specific commit, so there is\n")
        f.write("  always a clear record of what was trained and when.\n\n")
        f.write("  Bias Awareness - synthetic data is evenly distributed with no\n")
        f.write("  underrepresented groups. for a real deployment on the Tetouan\n")
        f.write("  dataset a proper bias audit would be needed first.\n\n")
        f.write("Training Notes\n\n")
        f.write("  Gradient clipping (clipnorm=1.0) was applied to prevent\n")
        f.write("  exploding gradients during backpropagation. if any gradient\n")
        f.write("  exceeds L2 norm of 1.0 it gets scaled down, which keeps\n")
        f.write("  training stable and stops the loss from going to NaN.\n\n")
        f.write("Pipeline completed successfully.\n")

    print(f"Saved: {filename}")


# data preparation

print(f"TensorFlow version: {tf.__version__}")
print("Setting up data...")

# generate synthetic data with a simple linear relationship: y = x + 10
# using arange so the values are evenly spaced and completely deterministic.
# this is important because it means every pipeline run produces the exact
# same data, which makes results reproducible and easy to verify
X = np.arange(-100, 100, 4)   # 50 values from -100 to 96 with step 4
y = np.arange(-90, 110, 4)    # corresponding y values where y = x + 10

print(f"Dataset: synthetic (y = x + 10), {len(X)} samples, no sampling bias")

# reshape from 1D (50,) to 2D (50, 1) - keras dense layers expect the input
# to have shape (samples, features) not just a flat array, so this step is
# necessary even though there's only one feature
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# fixed 80/20 train/test split using a hard-coded index rather than random
# shuffling - this ensures the exact same split happens every single run,
# which is important for reproducibility and fair comparison across runs
split_idx = 40
X_train = X[:split_idx]   # first 40 samples for training
y_train = y[:split_idx]
X_test  = X[split_idx:]   # last 10 samples held out for testing
y_test  = y[split_idx:]

print(f"Training: {len(X_train)} samples, Test: {len(X_test)} samples")


# model

print("Building model...")

# set seeds before building and training so results are reproducible.
# without this the weight initialisation is random and the model could
# converge to slightly different values each run
tf.random.set_seed(42)
np.random.seed(42)

# single dense layer with one output unit - this is the right choice for
# a simple linear regression task where the relationship is y = W*x + b.
# the learned weight and bias then fully explain every prediction the model
# makes, which is good from an interpretability standpoint.
#
# one thing worth mentioning - a more complex architecture with hidden relu
# layers would actually perform worse here. relu kills negative activations
# by outputting zero for any negative input, and our x values range from
# -100 to 100, so a relu hidden layer would zero out half the signal and
# make it much harder for the model to learn the full linear relationship.
# keeping it simple here is both more interpretable and more accurate.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        units=1,              # single output - predicting one value per input
        input_shape=(1,),     # one input feature (the x value)
        name='linear_layer'   # named so model.summary() is easy to read
    )
], name='linear_regression_model')

# adam optimiser with gradient clipping applied.
# gradient clipping (clipnorm=1.0) means if any gradient's l2 norm exceeds
# 1.0 during backpropagation it gets scaled down to that maximum. this
# prevents exploding gradients which can cause training to become unstable
# or the loss to go to NaN, especially in the early epochs when the weights
# are far from their optimal values. adam is used instead of plain sgd
# because it converges much faster for this kind of task
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.1,   # relatively high lr is fine here since adam adapts it
    clipnorm=1.0         # scale down any gradients that exceed this norm
)

# mae loss is a natural fit for regression - it's the average absolute
# difference between predicted and actual values, easy to interpret
model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
model.summary()


# training

print("Training...")

# training for 500 epochs with 20% of the training data held back for
# validation each epoch. adam converges a lot faster than sgd so 500 epochs
# is more than enough to get the model really close to the true relationship.
# verbose=0 keeps the actions log clean - no point printing 500 lines of
# per-epoch output when a summary at the end is more useful
history = model.fit(
    X_train, y_train,
    epochs=500,
    verbose=0,             # suppress per-epoch output
    validation_split=0.2   # hold back 20% of training data for validation
)

# print a quick summary of how training went
print(f"Final training loss   : {history.history['loss'][-1]:.4f}")
print(f"Final validation loss : {history.history['val_loss'][-1]:.4f}")


# evaluation

print("Evaluating...")

# run predictions on the held-out test set - these are samples the model
# has never seen during training so this gives a fair measure of performance
y_preds     = model.predict(X_test, verbose=0).flatten()
y_test_flat = y_test.flatten()

mae, mse, rmse = calculate_metrics(y_test_flat, y_preds)
weights        = model.get_weights()

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")

# print the learned parameters so it's easy to see what the model picked up
if len(weights) >= 2:
    print(f"Learned: y = {weights[0][0][0]:.4f}*x + {weights[1][0]:.4f}")
    print(f"Expected: y = 1.0000*x + 10.0000")


# save outputs

# generate and save the plots and metrics file - these all get picked up
# by the upload step in train.yml and made available as downloadable artefacts
plot_predictions(
    X_train.flatten(), y_train.flatten(),
    X_test.flatten(), y_test_flat,
    y_preds
)
plot_training_history(history)
save_metrics(mae, mse, rmse, weights)

# save the model weights so they can be loaded later without retraining
model.save('linear_regression_model.h5')
print("Saved: linear_regression_model.h5")

# quick sanity check on an input the model hasn't seen - x=75 should
# give something close to 85 (since y = x + 10). if it's way off then
# something has gone wrong with training
sample_y_pred = model.predict(np.array([[75.0]]), verbose=0)[0][0]
print(f"\nSanity check - x=75 -> predicted={sample_y_pred:.2f} (expected ~85.00)")
print("Pipeline completed successfully.")
