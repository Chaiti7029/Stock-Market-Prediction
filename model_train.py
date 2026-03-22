import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import joblib
import os

# Set your training parameters
EPOCHS = 5
BATCH_SIZE = 32

tickers = [
    'AAPL',         # Apple
    'TSLA',         # Tesla
    'JPM',          # JPMorgan
    'JNJ',          # Johnson & Johnson
    'RELIANCE.NS',  # Reliance
    'TCS.NS',       # Tata Consultancy
    'INFY.NS',      # Infosys
    'HDFCBANK.NS'   # HDFC Bank
]

# Create output directory
output_dir = 'models'
os.makedirs(output_dir, exist_ok=True)

for ticker in tickers:
    print(f"\n====== Training model for {ticker} ======")
    try:
        df = yf.download(ticker, start='2012-01-01',auto_adjust=True)
        if df.empty:
            print(f"No data found for {ticker}, skipping.")
            continue

        # Use 'Close' or fallback to 'Adj Close'
        if 'Close' in df.columns:
            data = df[['Close']]
        elif 'Adj Close' in df.columns:
            data = df[['Adj Close']].rename(columns={'Adj Close': 'Close'})
        else:
            print(f"No Close or Adj Close in {ticker} data. Skipping.")
            continue

        dataset = data.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        training_data_len = int(np.ceil(len(dataset) * 0.95))
        train_data = scaled_data[:training_data_len]

        x_train, y_train = [], []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build model
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(60, 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train model
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

        # Save model and scaler
        model_path = os.path.join(output_dir, f'{ticker}_lstm_model.h5')
        scaler_path = os.path.join(output_dir, f'{ticker}_scaler.save')
        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        print(f"✅ Model saved: {model_path}")
        print(f"✅ Scaler saved: {scaler_path}")

    except Exception as e:
        print(f"❌ Error with {ticker}: {e}")
