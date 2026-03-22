# AI Stock Predictor 📈

A sophisticated web application that uses Long Short-Term Memory (LSTM) neural networks to predict stock prices and provide intelligent investment recommendations. Built with Flask, TensorFlow/Keras, and modern web technologies.

## 🚀 Features

- **Advanced LSTM Predictions**: Uses deep learning to forecast stock prices for the next 10 days
- **Multi-Stock Support**: Predictions for major US and Indian stocks including Apple, Tesla, JPMorgan, and more
- **Intelligent Recommendations**: AI-powered buy/sell/hold recommendations based on predicted price movements
- **Interactive Charts**: Beautiful visualizations showing historical data and future predictions
- **Real-time Data**: Fetches live stock data using Yahoo Finance API
- **Responsive Design**: Modern, mobile-friendly interface with smooth animations
- **Detailed Analytics**: Price change analysis, percentage calculations, and trend insights

## 📊 Supported Stocks

### US Stocks
- **AAPL** - Apple Inc.
- **TSLA** - Tesla Inc.
- **JNJ** - Johnson & Johnson
- **JPM** - JPMorgan Chase & Co.

### Indian Stocks (NSE)
- **HDFCBANK.NS** - HDFC Bank
- **INFY.NS** - Infosys Limited
- **RELIANCE.NS** - Reliance Industries
- **TCS.NS** - Tata Consultancy Services

## 🛠️ Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: TensorFlow/Keras with LSTM networks
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Stock Data**: Yahoo Finance API (yfinance)
- **Visualization**: Matplotlib for chart generation
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Styling**: Custom CSS with modern gradients and animations

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Internet connection for stock data fetching

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock-market-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models** (First time setup)
   ```bash
   python model_train.py
   ```
   This will download historical data and train LSTM models for all supported stocks.

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and navigate to `http://127.0.0.1:5000`

## 📖 Usage

1. **Select a Stock**: Choose from the dropdown menu of supported stocks
2. **Generate Prediction**: Click "Predict Stock" to analyze the selected stock
3. **Review Results**: 
   - View current price and predicted price changes
   - Check AI recommendation (Buy/Sell/Hold)
   - Examine the interactive chart showing historical and predicted data
   - Review the 10-day forecast table

## 🧠 How It Works

### Machine Learning Model
- **Architecture**: LSTM (Long Short-Term Memory) neural network
- **Input**: 60 days of historical closing prices
- **Output**: 10-day price predictions
- **Training**: Uses 95% of historical data for training, 5% for validation
- **Data**: Fetches data from 2012 onwards for comprehensive training

### Prediction Process
1. Downloads historical stock data using Yahoo Finance API
2. Preprocesses data using MinMaxScaler (0-1 normalization)
3. Feeds the last 60 days of data into the trained LSTM model
4. Generates predictions for the next 10 business days
5. Calculates price changes and percentage movements
6. Provides investment recommendations based on predicted changes

### Recommendation Logic
- **Strong Buy**: >5% predicted increase
- **Buy**: 2-5% predicted increase
- **Hold**: -2% to +2% predicted change
- **Sell**: -2% to -5% predicted decrease
- **Strong Sell**: <-5% predicted decrease

## 📁 Project Structure

```
stock-market-predictor/
├── app.py                 # Main Flask application
├── model_train.py         # Model training script
├── model.py              # Additional model utilities
├── requirements.txt       # Python dependencies
├── builder.config.json    # Build configuration
├── templates/
│   └── index.html        # Web interface
└── models/               # Trained model files (generated)
    ├── AAPL_lstm_model.h5
    ├── AAPL_scaler.save
    └── ... (other stock models)
```

## 🔧 Configuration

### Model Training Parameters
- **Epochs**: 5 (configurable in `model_train.py`)
- **Batch Size**: 32
- **Training Data**: 95% of historical data
- **Sequence Length**: 60 days
- **Features**: Closing price only

### Web Application Settings
- **Port**: 5000 (default Flask port)
- **Debug Mode**: Enabled for development
- **Chart Quality**: 150 DPI for high-resolution images

## 🚨 Important Notes

- **Model Training**: Models must be trained before using the application
- **Internet Required**: Application needs internet connection for stock data
- **Prediction Accuracy**: Past performance doesn't guarantee future results
- **Investment Decisions**: Use predictions as one of many tools, not sole basis for investment decisions

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**
   - Solution: Run `python model_train.py` to train models

2. **No data available error**
   - Solution: Check internet connection and stock symbol validity

3. **Import errors**
   - Solution: Ensure all dependencies are installed: `pip install -r requirements.txt`

4. **Port already in use**
   - Solution: Change port in `app.py` or kill existing process

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This application is for educational and research purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always consult with financial advisors and conduct thorough research before making investment decisions.

## 📞 Support

For questions, issues, or contributions, please open an issue on the GitHub repository.

---

**Built with ❤️ using Python, Flask, and TensorFlow**

---

**Made with 💖 by Amarnath** 