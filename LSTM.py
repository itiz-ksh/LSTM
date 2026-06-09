# LSTM Stock Price Predictor
# We use an LSTM (Long Short-Term Memory) neural network to
# predict stock closing prices based on historical data.
# LSTMs are great for time-series data because they can
# "remember" patterns over long sequences.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "Dataset" / "AAPL.csv"
MODEL_PATH   = PROJECT_DIR / "Model"   / "lstm_stock_model.keras"

# Cleaned data and loaded it.
def load_close_prices(csv_path=DATASET_PATH):
    # Reads the CSV, parses dates, strips dollar signs from the
    # Close/Last column, and returns rows sorted oldest -> newest.
    df = pd.read_csv(csv_path, skipinitialspace=True)

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

    # The raw CSV has prices like "$182.34" — strip the $ sign becomes 182.34
    df["Close/Last"] = (
        df["Close/Last"]
        .replace(r"[\$,]", "", regex=True)
        .astype(float)
    )
    return (
        df.sort_values("Date")[["Date", "Close/Last"]]
        .reset_index(drop=True)
    )


# Built sequences 
def create_sequences(data, time_steps=60):
    
    # Converts a 1-D price array into (X, y) pairs where:
    #   X[i] = the 60 days BEFORE day i  (the "look-back window")
    #   y[i] = the price ON day i        (what we want to predict)

    # Example with time_steps=3 and data=[1,2,3,4,5]:
    #   X = [[1,2,3], [2,3,4]]
    #   y = [4, 5]
    
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# Main LSTM Model
def build_model(time_steps):
    
    # Two stacked LSTM layers followed by two Dense layers.
    # - return_sequences=True on the first LSTM passes the full
    #   sequence output to the next LSTM layer.
    # - Dropout(0.2) randomly zeros 20 % of neurons each step
    #   to reduce overfitting.
    # - The final Dense(1) outputs a single predicted price.
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),

        LSTM(units=50, return_sequences=False),
        Dropout(0.2),

        Dense(units=25, activation="relu"),
        Dense(units=1),   # output: one predicted price
    ])

    # Adam is a Adaptive learning-rate optimizer — a solid
    # default choice. MSE penalises large errors heavily,
    # which suits price prediction.
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# Evaluate and print metrics 
def print_metrics(y_true, y_pred):
    
    # RMSE(Root Mean Square Error) tells us same unit as price; lower is better.
    # MAE(Mean Absolute Error) tells average absolute error in dollars; easier to read.
    # MAPE(Mean Absolute Predicted Error ) tells percentage error; scale-independent.
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n{'─'*35}")
    print(f"  RMSE : ${rmse:.2f}")
    print(f"  MAE  : ${mae:.2f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"{'─'*35}\n")


# Plot 
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(14, 6))
    plt.plot(y_true,  color="royalblue", label="Actual Price",    linewidth=1.5)
    plt.plot(y_pred,  color="tomato",    label="Predicted Price",
             linestyle="dashed", linewidth=1.5)
    plt.title("AAPL Stock Price — LSTM Prediction", fontsize=14)
    plt.xlabel("Trading Days (test set)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    prices = load_close_prices()
    print(f"Loaded {len(prices)} rows  |  date range: "
          f"{prices['Date'].min().date()} → {prices['Date'].max().date()}")
    print(prices.head(), "\n")

    # Scale
    # LSTMs train much better when inputs are in [0, 1].
    # We fit the scaler ONLY on training data later, but here
    # we scale everything first for simplicity — acceptable for
    # a learning project; in production fit only on train.
    scaler      = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(prices[["Close/Last"]])

    # Sequence
    TIME_STEPS = 60
    X, y = create_sequences(data_scaled, TIME_STEPS)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # → (samples, 60, 1)
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}\n")

    # Split: 70 % train | 10 % val | 20 % test
    # Crucially we do NOT shuffle — order matters in time series!
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test  = X[val_end:], y[val_end:]

    print(f"Train : {X_train.shape[0]} samples")
    print(f"Val : {X_val.shape[0]} samples")
    print(f"Test : {X_test.shape[0]} samples\n")

    # Train
    model = build_model(TIME_STEPS)
    model.summary()
    # EarlyStopping watches val_loss; if it stops improving for
    # 10 epochs it halts training and restores the best weights.
    # This prevents overfitting and saves time.
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        epochs=100,           # EarlyStopping will cut this short
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1,
    )

    print(f"\nStopped after {len(history.history['loss'])} epochs.")

    # Evaluation
    predictions_scaled = model.predict(X_test)

    # Inverse-transform back to real dollar values
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    print_metrics(y_test_actual, predictions)
    plot_predictions(y_test_actual, predictions)

    # Save model 
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")

if __name__ == "__main__":
    main()