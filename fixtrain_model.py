#!/usr/bin/env python3
"""
LSTM Trading System - Complete System Fixer
Diagnoses and fixes all common issues
"""

import os
import shutil
from datetime import datetime

def backup_file(filename):
    """Backup a file with timestamp"""
    if os.path.exists(filename):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{filename}_backup_{timestamp}"
        shutil.copy2(filename, backup_name)
        print(f"   📁 Backed up {filename} to {backup_name}")
        return True
    return False

def check_file_contents(filename, expected_content_check):
    """Check if file contains expected content"""
    if not os.path.exists(filename):
        return False, "File does not exist"
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if expected_content_check(content):
            return True, "File content is correct"
        else:
            return False, "File content needs updating"
    except Exception as e:
        return False, f"Error reading file: {e}"

def create_correct_model_py():
    """Create the correct PyTorch model.py"""
    content = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_regression_outputs):
        super(CombinedLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        
        # Regression head (for price predictions)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_regression_outputs)
        )
        
        # Classification head (for buy/sell/hold)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output for predictions
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Regression output (price predictions)
        regression_output = self.regression_head(last_output)
        
        # Classification output (buy/sell/hold probabilities)
        classification_logits = self.classification_head(last_output)
        
        return regression_output, classification_logits

# Alternative simpler model if you prefer
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1):
        super(SimpleLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

def test_model():
    """Test the model to make sure it works"""
    print("Testing CombinedLSTM model...")
    
    # Create model
    model = CombinedLSTM(
        input_size=15,
        hidden_size=128,
        num_layers=2,
        num_classes=3,
        num_regression_outputs=5
    )
    
    # Test input
    batch_size = 4
    seq_len = 20
    input_size = 15
    test_input = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    reg_output, class_logits = model(test_input)
    
    print(f"✅ Model test successful!")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Regression output shape: {reg_output.shape}")
    print(f"   Classification output shape: {class_logits.shape}")
    
    return True

if __name__ == "__main__":
    test_model()
'''
    
    with open('model.py', 'w', encoding='utf-8') as f:
        f.write(content)

def create_correct_data_processing_py():
    """Create the correct data_processing_with_indicators.py"""
    content = '''import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe"""
    df = df.copy()  # Work on a copy to avoid modifying original
    
    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = compute_bollinger_bands(df["close"])
    df["stoch_k"], df["stoch_d"] = compute_stochastic(df["high"], df["low"], df["close"])
    df["cci_14"] = compute_cci(df["high"], df["low"], df["close"], 14)
    
    # Fill NaN values (use newer pandas methods)
    df = df.bfill().ffill()
    
    return df

def compute_rsi(series, period=14):
    """Compute Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # Add small value to avoid division by zero
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(series, period=20, num_std=2):
    """Compute Bollinger Bands"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower

def compute_stochastic(high, low, close, k_period=14, d_period=3):
    """Compute Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
    d = k.rolling(window=d_period).mean()
    return k, d

def compute_cci(high, low, close, period=14):
    """Compute Commodity Channel Index"""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))) if len(x) > 0 else 0
    )
    return (typical_price - sma) / (0.015 * mean_deviation + 1e-10)

def load_and_align_data(symbol_files, base_path):
    """Load and align data from multiple symbol files"""
    combined_df = pd.DataFrame()
    for symbol, file in symbol_files.items():
        filepath = os.path.join(base_path, file)  # Fixed: added os.path.join
        df = pd.read_csv(filepath)
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]]
        df.columns = [f"{col}_{symbol}" for col in df.columns]
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = combined_df.join(df, how="outer")
    combined_df = combined_df.dropna()
    return combined_df

def create_features_for_live(df, sequence_length=20):
    """
    Create features for live prediction - CORRECTED for proper LSTM input shape
    Returns: (num_sequences, sequence_length, num_features)
    """
    print(f"Creating features for {len(df)} data points with sequence length {sequence_length}")
    
    # Add technical indicators
    df_with_indicators = add_indicators(df.copy())
    
    # Initialize feature array: (num_sequences, sequence_length, num_features)
    num_sequences = len(df_with_indicators) - sequence_length + 1
    num_features = 15  # Fixed to 15 features as expected by model
    features = np.zeros((num_sequences, sequence_length, num_features), dtype=np.float32)
    
    for seq_idx in range(num_sequences):
        for time_step in range(sequence_length):
            data_idx = seq_idx + time_step
            row_features = []
            
            # 1. Price change (normalized) - matches MQL5 EA
            if data_idx > 0:
                price_change = (df_with_indicators.iloc[data_idx]['close'] / 
                              df_with_indicators.iloc[data_idx-1]['close']) - 1.0
            else:
                price_change = 0.0
            row_features.append(price_change)
            
            # 2. Volume (normalized)
            volume = df_with_indicators.iloc[data_idx].get('volume', 1000)
            row_features.append(float(volume))
            
            # 3. ATR (using high-low range as proxy)
            atr_proxy = ((df_with_indicators.iloc[data_idx]['high'] - 
                         df_with_indicators.iloc[data_idx]['low']) / 
                        (df_with_indicators.iloc[data_idx]['close'] + 1e-10))
            row_features.append(atr_proxy)
            
            # 4. MACD proxy (simple price momentum)
            if data_idx >= 12:
                ema12 = df_with_indicators.iloc[data_idx-11:data_idx+1]['close'].ewm(span=12).mean().iloc[-1]
                ema26 = df_with_indicators.iloc[max(0,data_idx-25):data_idx+1]['close'].ewm(span=26).mean().iloc[-1]
                macd_proxy = ema12 - ema26
            else:
                macd_proxy = 0.0
            row_features.append(macd_proxy)
            
            # 5. RSI
            rsi = df_with_indicators.iloc[data_idx].get('rsi_14', 50.0)
            row_features.append(rsi)
            
            # 6. Stochastic K
            stoch_k = df_with_indicators.iloc[data_idx].get('stoch_k', 50.0)
            row_features.append(stoch_k)
            
            # 7. CCI
            cci = df_with_indicators.iloc[data_idx].get('cci_14', 0.0)
            row_features.append(cci)
            
            # 8. Hour of day
            timestamp = df_with_indicators.index[data_idx]
            row_features.append(float(timestamp.hour))
            
            # 9. Day of week
            row_features.append(float(timestamp.dayofweek))
            
            # 10. USD strength proxy (placeholder - will be 0 for single pair)
            row_features.append(0.0)
            
            # 11. EUR strength proxy (placeholder)
            row_features.append(0.0)
            
            # 12. JPY strength proxy (placeholder)
            row_features.append(0.0)
            
            # 13. Bollinger Bands ratio
            bb_upper = df_with_indicators.iloc[data_idx].get('bb_upper', df_with_indicators.iloc[data_idx]['close'])
            bb_lower = df_with_indicators.iloc[data_idx].get('bb_lower', df_with_indicators.iloc[data_idx]['close'])
            bb_ratio = (bb_upper - bb_lower) / (df_with_indicators.iloc[data_idx]['close'] + 1e-10)
            row_features.append(bb_ratio)
            
            # 14. Volume change
            if data_idx > 4:
                vol_change = volume - df_with_indicators.iloc[data_idx-5].get('volume', volume)
            else:
                vol_change = 0.0
            row_features.append(vol_change)
            
            # 15. Candlestick pattern indicator
            body = abs(df_with_indicators.iloc[data_idx]['close'] - df_with_indicators.iloc[data_idx]['open'])
            range_val = df_with_indicators.iloc[data_idx]['high'] - df_with_indicators.iloc[data_idx]['low']
            if range_val > 0:
                body_ratio = body / range_val
                if body_ratio < 0.1:
                    pattern_indicator = 1.0  # Doji-like
                elif df_with_indicators.iloc[data_idx]['close'] > df_with_indicators.iloc[data_idx]['open']:
                    pattern_indicator = 2.0  # Bullish
                else:
                    pattern_indicator = -2.0  # Bearish
            else:
                pattern_indicator = 0.0
            
            # Add ATR normalization
            if data_idx > 0:
                pattern_indicator += ((df_with_indicators.iloc[data_idx]['open'] - 
                                     df_with_indicators.iloc[data_idx-1]['close']) / 
                                    (atr_proxy + 1e-10))
            
            row_features.append(pattern_indicator)
            
            # Ensure exactly 15 features
            row_features = row_features[:15]  # Take first 15
            while len(row_features) < 15:  # Pad if less than 15
                row_features.append(0.0)
            
            # Store in the feature array
            features[seq_idx, time_step, :] = row_features
    
    print(f"Created feature matrix with shape: {features.shape}")
    print(f"Expected shape for LSTM: (num_sequences, {sequence_length}, {num_features})")
    return features

def prepare_training_data(df, sequence_length=20, prediction_steps=5):
    """
    Prepare data for training the LSTM model - CORRECTED for proper shapes
    """
    print(f"Preparing training data from {len(df)} rows...")
    
    # Create features with correct shape: (num_sequences, sequence_length, num_features)
    X = create_features_for_live(df, sequence_length)
    
    # Create targets (next N price values)
    y_regression = []
    y_classification = []
    
    for i in range(len(df) - sequence_length - prediction_steps + 1):
        # Regression targets (future prices)
        future_prices = []
        current_price = df.iloc[i + sequence_length - 1]['close']
        
        for step in range(1, prediction_steps + 1):
            if i + sequence_length + step - 1 < len(df):
                future_price = df.iloc[i + sequence_length + step - 1]['close']
                # Store as price change ratio
                price_change = (future_price / current_price) - 1.0
                future_prices.append(price_change)
            else:
                future_prices.append(0.0)
        
        y_regression.append(future_prices)
        
        # Classification targets (buy/sell/hold)
        # Simple classification based on future price movement
        avg_future_change = np.mean(future_prices)
        if avg_future_change > 0.001:  # 0.1% threshold
            classification = [0, 0, 1]  # Buy (index 2)
        elif avg_future_change < -0.001:
            classification = [0, 1, 0]  # Sell (index 1)  
        else:
            classification = [1, 0, 0]  # Hold (index 0)
        
        y_classification.append(classification)
    
    # Ensure X and y have same length
    min_length = min(len(X), len(y_regression))
    X = X[:min_length]
    y_regression = np.array(y_regression[:min_length], dtype=np.float32)
    y_classification = np.array(y_classification[:min_length], dtype=np.float32)
    
    print(f"Prepared training data:")
    print(f"  X shape: {X.shape} (sequences, timesteps, features)")
    print(f"  y_regression shape: {y_regression.shape}")
    print(f"  y_classification shape: {y_classification.shape}")
    
    return X, y_regression, y_classification

# Test function
def test_data_processing():
    """Test the data processing functions"""
    print("Testing data processing functions...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='h')
    df = pd.DataFrame({
        'open': np.random.uniform(1.0500, 1.0600, 100),
        'high': np.random.uniform(1.0550, 1.0650, 100),
        'low': np.random.uniform(1.0450, 1.0550, 100),
        'close': np.random.uniform(1.0500, 1.0600, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    }, index=dates)
    
    # Make prices more realistic
    for i in range(1, len(df)):
        df.iloc[i] = df.iloc[i-1] * (1 + np.random.normal(0, 0.001))
    
    # Test functions
    try:
        df_with_indicators = add_indicators(df)
        print("✅ add_indicators works")
        
        features = create_features_for_live(df, 20)
        print(f"✅ create_features_for_live works - shape: {features.shape}")
        
        X, y_reg, y_class = prepare_training_data(df, 20, 5)
        print(f"✅ prepare_training_data works - X: {X.shape}, y_reg: {y_reg.shape}, y_class: {y_class.shape}")
        
        # Verify the shape is correct for LSTM
        expected_shape = (X.shape[0], 20, 15)  # (sequences, timesteps, features)
        if X.shape == expected_shape:
            print(f"✅ Shape is correct for LSTM: {X.shape}")
        else:
            print(f"❌ Shape is wrong! Expected {expected_shape}, got {X.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_processing()
'''
    
    with open('data_processing_with_indicators.py', 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    """Main diagnostic and fix function"""
    print("🔧 LSTM Trading System - Complete System Fixer")
    print("=" * 60)
    
    issues_found = []
    issues_fixed = []
    
    # 1. Check model.py
    print("\n1. 🔍 Checking model.py...")
    is_correct, msg = check_file_contents('model.py', lambda content: 'CombinedLSTM' in content and 'torch.nn' in content)
    print(f"   {msg}")
    
    if not is_correct:
        issues_found.append("model.py needs PyTorch version")
        backup_file('model.py')
        create_correct_model_py()
        print("   ✅ Created correct PyTorch model.py")
        issues_fixed.append("Fixed model.py")
    
    # 2. Check data_processing_with_indicators.py
    print("\n2. 🔍 Checking data_processing_with_indicators.py...")
    is_correct, msg = check_file_contents('data_processing_with_indicators.py', 
                                        lambda content: 'import os' in content and 'create_features_for_live' in content)
    print(f"   {msg}")
    
    if not is_correct:
        issues_found.append("data_processing_with_indicators.py needs fixing")
        backup_file('data_processing_with_indicators.py')
        create_correct_data_processing_py()
        print("   ✅ Created correct data_processing_with_indicators.py")
        issues_fixed.append("Fixed data_processing_with_indicators.py")
    
    # 3. Check for conflicting files
    print("\n3. 🔍 Checking for conflicting files...")
    conflicting_files = []
    for file in os.listdir('.'):
        if file.endswith('.py') and ('tensorflow' in file.lower() or 'tf_' in file.lower()):
            conflicting_files.append(file)
    
    if conflicting_files:
        print(f"   ⚠️  Found conflicting TensorFlow files: {conflicting_files}")
        for file in conflicting_files:
            backup_name = f"{file}_old_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(file, backup_name)
            print(f"   📁 Moved {file} to {backup_name}")
        issues_fixed.append(f"Moved conflicting files: {conflicting_files}")
    
    # 4. Test imports
    print("\n4. 🧪 Testing imports...")
    try:
        # Clear any cached imports
        import sys
        modules_to_clear = ['model', 'data_processing_with_indicators']
        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]
        
        from model import CombinedLSTM
        print("   ✅ Successfully imported CombinedLSTM")
        
        from data_processing_with_indicators import create_features_for_live, prepare_training_data
        print("   ✅ Successfully imported data processing functions")
        
        # Test model creation
        model = CombinedLSTM(input_size=15, hidden_size=128, num_layers=2, num_classes=3, num_regression_outputs=5)
        print("   ✅ Successfully created model instance")
        
        issues_fixed.append("All imports working correctly")
        
    except Exception as e:
        print(f"   ❌ Import test failed: {e}")
        issues_found.append(f"Import error: {e}")
    
    # 5. Check requirements
    print("\n5. 📦 Checking Python packages...")
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("   ❌ PyTorch not installed")
        issues_found.append("PyTorch not installed")
    
    try:
        import numpy as np
        print(f"   ✅ NumPy: {np.__version__}")
    except ImportError:
        print("   ❌ NumPy not installed") 
        issues_found.append("NumPy not installed")
    
    try:
        import pandas as pd
        print(f"   ✅ Pandas: {pd.__version__}")
    except ImportError:
        print("   ❌ Pandas not installed")
        issues_found.append("Pandas not installed")
    
    # 6. Check data file
    print("\n6. 📊 Checking data file...")
    if os.path.exists('EURUSD60.csv'):
        size = os.path.getsize('EURUSD60.csv')
        print(f"   ✅ EURUSD60.csv found ({size:,} bytes)")
    else:
        print("   ⚠️  EURUSD60.csv not found")
        issues_found.append("Data file missing")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if issues_found:
        print("❌ Issues Found:")
        for issue in issues_found:
            print(f"   • {issue}")
    
    if issues_fixed:
        print("✅ Issues Fixed:")
        for fix in issues_fixed:
            print(f"   • {fix}")
    
    if not issues_found:
        print("🎉 All systems operational!")
        print("\n🚀 Ready to run:")
        print("   python train_model.py")
    else:
        remaining_issues = [issue for issue in issues_found if not any(fix in issue for fix in issues_fixed)]
        if remaining_issues:
            print(f"\n⚠️  Remaining issues to fix manually:")
            for issue in remaining_issues:
                print(f"   • {issue}")
        else:
            print("\n🎉 All issues have been fixed!")
            print("\n🚀 Ready to run:")
            print("   python train_model.py")
    
    print("\n📝 Note: TensorFlow warnings are normal - the system uses PyTorch")

if __name__ == "__main__":
    main()