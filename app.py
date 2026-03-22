import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import joblib
from flask import Flask, render_template, request, jsonify
import io
import base64
import os

app = Flask(__name__)

# Supported stock list
STOCKS = {
    'AAPL': {'name': 'Apple Inc.'},
    'TSLA': {'name': 'Tesla Inc.'},
    'JNJ': {'name': 'Johnson & Johnson'},
    'JPM': {'name': 'JPMorgan Chase & Co.'},
    'HDFCBANK.NS': {'name': 'HDFC Bank'},
    'INFY.NS': {'name': 'Infosys Limited'},
    'RELIANCE.NS': {'name': 'Reliance Industries'},
    'TCS.NS': {'name': 'Tata Consultancy Services'}
}

# Load model and scaler
def load_stock_model(stock_symbol):
    try:
        model_path = f'models/{stock_symbol}_lstm_model.h5'
        scaler_path = f'models/{stock_symbol}_scaler.save'
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, None
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        print(f"Error loading model for {stock_symbol}: {e}")
        return None, None

# Predict next 10 days
def predict_next_10_days(stock_symbol):
    model, scaler = load_stock_model(stock_symbol)
    if model is None or scaler is None:
        return None, None, None, "Model not available for this stock."

    try:
        df = yf.download(stock_symbol, start='2012-01-01', auto_adjust=True)
        if df.empty:
            return None, None, None, "No data available for this stock."

        data = df[['Close']]
        scaled_data = scaler.transform(data.values)
        last_60_days = scaled_data[-60:]
        input_seq = last_60_days.copy()

        future_predictions = []
        for _ in range(10):
            pred_input = np.reshape(input_seq, (1, 60, 1))
            pred_scaled = model.predict(pred_input, verbose=0)
            future_predictions.append(pred_scaled[0, 0])
            input_seq = np.append(input_seq, pred_scaled)[-60:]

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
        pred_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=10, freq='B')

        return data, pred_dates, future_predictions, None

    except Exception as e:
        return None, None, None, f"Prediction error: {str(e)}"

# Recommendation logic
def get_recommendation(preds, current_price):
    if preds is None or len(preds) == 0:
        return "Hold", "neutral"
    change = preds[-1] - current_price
    pct_change = (change / current_price) * 100
    if pct_change > 5:
        return "Strong Buy", "buy"
    elif pct_change > 2:
        return "Buy", "buy"
    elif pct_change < -5:
        return "Strong Sell", "sell"
    elif pct_change < -2:
        return "Sell", "sell"
    else:
        return "Hold", "neutral"

@app.route('/')
def index():
    return render_template('index.html', stocks=STOCKS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock_symbol = request.json.get('stock_symbol')
        print(f"Received prediction request for: {stock_symbol}")
    except Exception as e:
        return jsonify({'error': f'Invalid JSON: {str(e)}'}), 400

    if stock_symbol not in STOCKS:
        return jsonify({'error': 'Invalid stock symbol'}), 400

    data, pred_dates, future_predictions, error = predict_next_10_days(stock_symbol)
    if error:
        return jsonify({'error': error}), 500

    current_price = float(data['Close'].iloc[-1].item())
    recommendation, rec_type = get_recommendation(future_predictions, current_price)

    # Create prediction chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Close'][-150:], label='Last 150 Days', color='#2E86AB')
    ax.plot(pred_dates, future_predictions, marker='o', label='Predicted Next 10 Days', color='#F24236')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title(f"{STOCKS[stock_symbol]['name']} Stock Forecast")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Build prediction table
    pred_table = []
    for i, (date, price) in enumerate(zip(pred_dates, future_predictions)):
        pred_table.append({
            'date': str(date.strftime('%Y-%m-%d')),
            'price': float(round(price, 2)),
            'day': f'Day {i+1}'
        })

    price_change = float(future_predictions[-1] - current_price)
    percentage_change = (price_change / current_price) * 100

    # Convert all values to native Python types before jsonify
    return jsonify({
        'success': True,
        'stock_name': str(STOCKS[stock_symbol]['name']),
        'current_price': float(round(current_price, 2)),
        'predictions': pred_table,
        'recommendation': str(recommendation),
        'recommendation_type': str(rec_type),
        'price_change': float(round(price_change, 2)),
        'percentage_change': float(round(percentage_change, 2)),
        'chart': str(img_base64)
    })

if __name__ == '__main__':
    app.run(debug=True)
