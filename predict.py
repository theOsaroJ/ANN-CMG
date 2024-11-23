import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('co2_data.csv')

# Separate input features (first 7 columns) and target variables (last 3 columns)
X = data.iloc[:, :7].values  # First 7 columns as input features
y = data.iloc[:, -3:].values  # Last 3 columns as target variables

# Modify Target 3: Start from row 738 and exclude rows with 0 values
y[:, 2] = np.where(data.index >= 738, y[:, 2], np.nan)  # Replace values before row 738 with NaN
non_zero_indices = ~np.isnan(y[:, 2]) & (y[:, 2] != 0)  # Find rows where Target 3 is non-zero
X = X[non_zero_indices]
y = y[non_zero_indices]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Create directories to save the plots
if not os.path.exists("plots"):
    os.makedirs("plots")

# Lists to store the models, scalers, results, and metrics
models = []
scalers_y = []
results = []
metrics_data = []

# Define target names for display purposes
target_names = [
    "Cumulative Oil Production (bbl)",
    "Oil Recovery Factor (%)",
    "Cumulative CO2 Stored (SCF)"
]

# Set larger font sizes for plots
plt.rcParams.update({'font.size': 14})

# Loop to create and train a separate model for each target variable
for i in range(y.shape[1]):
    # Standardize each target variable separately
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train[:, i].reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test[:, i].reshape(-1, 1))
    scalers_y.append(scaler_y)

    # Define the model-building function for Keras Tuner
    def build_model(hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units_input', min_value=32, max_value=128, step=32),
                        activation='relu', input_dim=X_train.shape[1]))
        for j in range(hp.Int('num_layers', 1, 4)):
            model.add(Dense(units=hp.Int(f'units_{j}', min_value=32, max_value=128, step=32),
                            activation='relu'))
        model.add(Dense(1))  # Single output for each target variable
        model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                      loss='mse', metrics=['mae'])
        return model

    # Initialize Keras Tuner for each target variable
    tuner = RandomSearch(build_model,
                         objective='val_loss',
                         max_trials=5,
                         executions_per_trial=2,
                         directory='hyperparam_tuning',
                         project_name=f'co2_prediction_target_{i}')

    # Run the tuner
    tuner.search(X_train, y_train_scaled, epochs=50, validation_split=0.2, batch_size=32)

    # Retrieve the best hyperparameters and rebuild the model
    best_hyperparameters = tuner.get_best_hyperparameters()[0]
    best_model = build_model(best_hyperparameters)

    # Retrain the best model to capture the training history
    history = best_model.fit(X_train, y_train_scaled, epochs=50, validation_split=0.2, batch_size=32, verbose=1)

    # Store the best model for later use
    models.append(best_model)

    # Evaluate the model
    y_pred_scaled = best_model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)  # Inverse transform predictions
    y_true = scaler_y.inverse_transform(y_test_scaled)  # Inverse transform true values

    # Calculate evaluation metrics
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    results.append((mse, r2, mae))
    metrics_data.append({"Target": target_names[i], "Mean Squared Error": mse, "R-Squared": r2 , "Mean Absolute Error": mae})

    print(f"{target_names[i]} - Best Hyperparameters:", best_hyperparameters.values)
    print(f"{target_names[i]} - Mean Squared Error:", mse)
    print(f"{target_names[i]} - R-squared:", r2)

    # Save training and validation loss plots
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss for {target_names[i]}', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss (Mean Squared Error)', fontsize=14)
    plt.legend(fontsize=12)
    # plt.grid(True)
    loss_plot_path = f"plots/loss_plot_{target_names[i].replace(' ', '_')}.png"
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save predicted vs actual scatter plots
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='k')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', linewidth=2)
    plt.title(f'Predicted vs Actual for {target_names[i]}', fontsize=16)
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    # plt.grid(True)
    scatter_plot_path = f"plots/scatter_plot_{target_names[i].replace(' ', '_')}.png"
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

# Save the metrics to a CSV file
metrics_df = pd.DataFrame(metrics_data)
metrics_csv_path = "plots/metrics_summary.csv"
metrics_df.to_csv(metrics_csv_path, index=False)
