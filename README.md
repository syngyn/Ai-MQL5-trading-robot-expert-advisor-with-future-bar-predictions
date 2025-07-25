# LSTM Trading System v1.0

🚀 **Complete Machine Learning Trading System for MetaTrader 5**

## 🎯 Features

- **Dual-head LSTM Neural Network**: Price predictions + Buy/Sell/Hold classification
- **Technical Indicators**: RSI, Bollinger Bands, Stochastic, CCI, MACD
- **5-step Ahead Forecasting**: Predicts H+1 through H+5 price levels
- **Risk Management**: Confidence-based trade gating system
- **Real-time Integration**: Seamless MT5 Expert Advisor connection
- **Accuracy Tracking**: Live performance monitoring

## 📦 Quick Installation

1. **Extract** this package to your desired folder
2. **Run** `install.bat` (Windows) or install requirements manually
3. **Place** your `EURUSD60.csv` data file in the main directory
4. **Train** the model: Double-click `train_model.bat`
5. **Start** the daemon: Double-click `start_daemon.bat`
6. **Connect** your MT5 EA (`GGTH4_hardened.mq5`)

## 📊 Data Format

Your CSV file should have these columns:
```
"DATE","TIME","OPEN","HIGH","LOW","CLOSE","TICKVOL","VOL","SPREAD"
```

## 🖥️ System Requirements

- **Python 3.8+**
- **8GB+ RAM** (for training large datasets)
- **MetaTrader 5**
- **Windows 10/11** (tested)

## 📈 Expected Performance

- **Classification Accuracy**: ~67% (conservative, risk-averse)
- **Regression MSE**: <0.000003 (excellent price prediction)
- **Training Time**: 5-15 minutes (depending on data size)
- **Prediction Speed**: <100ms per request

## 🔧 Configuration

Edit `config.ini` to customize:
- Model parameters (hidden_size, num_layers)
- Training settings (epochs, batch_size)
- Risk thresholds (confidence levels)

## 📁 Directory Structure

```
LSTM_Trading_System/
├── install.bat              # Main installer
├── model.py                 # PyTorch LSTM model
├── daemon_final.py          # Prediction server
├── train_model.py           # Training script
├── requirements.txt         # Python dependencies
├── start_daemon.bat         # Start prediction server
├── train_model.bat          # Start training
├── config.ini               # Configuration
├── README.md                # This file
├── models/                  # Trained models saved here
├── data/                    # Daemon communication
├── logs/                    # System logs
└── backup/                  # Backups
```

## 🎮 Usage Workflow

1. **Training Phase**:
   - Place historical data (EURUSD60.csv)
   - Run training (`train_model.bat`)
   - Wait for completion (~10 minutes)
   - Model saved to `models/` folder

2. **Prediction Phase**:
   - Start daemon (`start_daemon.bat`)
   - Verify "✅ Model loaded successfully"
   - Run MT5 EA
   - See predictions on chart GUI

3. **Trading Phase**:
   - EA receives H+1 to H+5 price predictions
   - Confidence-based trade filtering
   - Automatic risk management
   - Live accuracy tracking

## ⚠️ Important Notes

- **Conservative by Design**: Model favors "Hold" decisions (safer)
- **Requires Quality Data**: 1000+ rows minimum, 10,000+ preferred
- **CPU Training**: Uses CPU by default (GPU optional)
- **Paper Trade First**: Test thoroughly before live trading

## 🐛 Troubleshooting

**"Python not found"**: Install Python 3.8+ from python.org
**"Module not found"**: Run `pip install -r requirements.txt`
**"Data file not found"**: Ensure EURUSD60.csv is in main directory
**"Model not trained"**: Run train_model.bat before starting daemon
**"Daemon timeout"**: Check Windows Firewall, restart daemon

## 📞 Support

- **Created by**: Jason Rusk (Jason.W.Rusk@gmail.com)
- **Version**: 1.0 (2025)
- **License**: Personal trading use only
- **Updates**: Check for new versions periodically

## 🚨 Disclaimer

This is experimental trading software. Use at your own risk. 
Always test thoroughly on demo accounts before live trading.
Past performance does not guarantee future results.

---

**🎉 Happy Trading! May your predictions be accurate and your profits consistent! 📈**
