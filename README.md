# LSTM Stock Price Predictor

A beginner-friendly deep learning project that uses an **LSTM (Long Short-Term Memory)** neural network to predict stock closing prices based on historical data. Built while learning time-series forecasting and deep learning fundamentals.

---

## What I Learned

- How to preprocess financial time-series data
- Why scaling inputs to `[0, 1]` helps neural networks train better
- How LSTMs "remember" patterns across long sequences — unlike regular RNNs
- How to build, train, evaluate, and save a Keras model
- The importance of keeping train/val/test sets in chronological order (no shuffling!)
- How `EarlyStopping` prevents overfitting and saves training time

---

## Project Structure

```
project/
│
├── Dataset/
│   └── AAPL.csv               # Historical Apple stock data
│
├── Model/
│   └── lstm_stock_model.keras # Saved model (generated after training)
│
├── LSTM.py                    # Main training & prediction script
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Prerequisites

- **Python 3.13.4** — [Download here](https://www.python.org/downloads/)
- `pip` (comes with Python)

> **Note:** TensorFlow does not yet officially support Python 3.13.
> If you run into installation issues, consider using **Python 3.11** or **3.12** instead.

### 2. Clone the repository

```bash
git clone https://github.com/itiz-ksh/LSTM
cd LSTM
```

### 3. Create a virtual environment

```bash
python -m venv venv

# Activate it:
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Make sure your dataset is placed at `Dataset/AAPL.csv`, then run:

```bash
python LSTM.py
```

The script will:
1. Load and clean the CSV data
2. Scale prices to `[0, 1]` using MinMaxScaler
3. Create 60-day look-back sequences
4. Split data into train (70%) / validation (10%) / test (20%)
5. Train the LSTM with early stopping
6. Print RMSE, MAE, and MAPE metrics
7. Plot actual vs predicted prices
8. Save the trained model to `Model/lstm_stock_model.keras`

---

## Dataset Format

The CSV file should have at least these columns:

| Column       | Example     |
|--------------|-------------|
| `Date`       | 01/15/2023  |
| `Close/Last` | $182.34     |

Dollar signs and commas in the price column are handled automatically.

---

## Model Architecture

```
Input → LSTM(50) → Dropout(0.2)
      → LSTM(50) → Dropout(0.2)
      → Dense(25, relu)
      → Dense(1)   ← predicted price
```

- **Optimizer:** Adam
- **Loss:** Mean Squared Error
- **Early Stopping:** patience=10, monitors `val_loss`

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE   | Root Mean Squared Error — in dollars |
| MAE    | Mean Absolute Error — average dollar error |
| MAPE   | Mean Absolute Percentage Error — scale-independent |

---

## Dependencies

See `requirements.txt` for the full list. Key libraries:

- `tensorflow` — LSTM model
- `scikit-learn` — scaling + metrics
- `pandas` — data loading and cleaning
- `numpy` — array operations
- `matplotlib` — plotting results

---

## Notes

- This project is for **learning purposes** and should not be used for real financial decisions.
- Stock prices are influenced by many factors a simple LSTM cannot capture.
- The model is retrained from scratch every run — a `load_model` inference mode can be added as a next step.

---

## Acknowledgements

Built while learning deep learning and time-series forecasting with Keras and TensorFlow.
