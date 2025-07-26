#!/usr/bin/env python3
"""
Complete LSTM Training Script for Forex Trading Predictions
Final Version - Compatible with daemon_final.py and MQL5 EA
Trains both regression (price prediction) and classification (buy/sell/hold) heads
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib not available - plots will be skipped")

# Import our custom modules
try:
    from model import CombinedLSTM
    from data_processing_with_indicators import prepare_training_data
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure model.py and data_processing_with_indicators.py exist in the same directory")
    print("Run 'python fix_system.py' to fix all import issues")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ================================
# CONFIGURATION PARAMETERS
# ================================
SEQUENCE_LENGTH = 20          # Number of time steps to look back
PREDICTION_STEPS = 5          # Number of future steps to predict (H+1 to H+5)
FEATURE_COUNT = 15            # Number of features per time step
BATCH_SIZE = 32               # Training batch size
EPOCHS = 100                  # Number of training epochs
LEARNING_RATE = 0.001         # Initial learning rate
TRAIN_SPLIT = 0.8             # Train/test split ratio
VALIDATION_SPLIT = 0.1        # Validation split from training data
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model architecture parameters
HIDDEN_SIZE = 128             # LSTM hidden size
NUM_LAYERS = 2               # Number of LSTM layers
NUM_CLASSES = 3              # Buy(2), Sell(1), Hold(0)

# Training parameters
EARLY_STOPPING_PATIENCE = 20  # Epochs to wait before early stopping
GRAD_CLIP_MAX_NORM = 1.0     # Gradient clipping threshold
L2_REGULARIZATION = 1e-5     # L2 regularization strength

# Loss weights
REGRESSION_WEIGHT = 1.0      # Weight for regression loss
CLASSIFICATION_WEIGHT = 0.5   # Weight for classification loss

print(f"🚀 LSTM Trading Model Training")
print(f"Using device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"🎲 Random seeds set to {seed}")

def normalize_column_names(df):
    """Normalize column names by removing quotes and converting to lowercase"""
    # Remove quotes and convert to lowercase
    df.columns = [col.strip('"').strip("'").lower() for col in df.columns]
    
    # Map common variations to standard names
    column_mapping = {
        'date': 'date',
        'time': 'time', 
        'datetime': 'timestamp',
        'timestamp': 'timestamp',
        'dt': 'timestamp',
        'open': 'open',
        'o': 'open',
        'high': 'high',
        'h': 'high',
        'low': 'low',
        'l': 'low',
        'close': 'close',
        'c': 'close',
        'tickvol': 'tickvol',
        'tick_volume': 'tickvol',
        'volume': 'volume',
        'vol': 'vol',
        'v': 'vol',
        'spread': 'spread'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    return df

def load_and_validate_data(csv_file):
    """Load and validate the CSV data with flexible column handling"""
    print(f"📊 Loading data from {csv_file}...")
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found!")
        print("Please place your historical data file as EURUSD60.csv in this directory")
        print("Expected format: DATE, TIME, OPEN, HIGH, LOW, CLOSE, TICKVOL, VOL, SPREAD")
        return None
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} rows from {csv_file}")
        print(f"   Original columns: {list(df.columns)}")
        
        # Normalize column names
        df = normalize_column_names(df)
        print(f"   Normalized columns: {list(df.columns)}")
        
        # Handle different timestamp formats
        if 'timestamp' in df.columns:
            # Single timestamp column
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns and 'time' in df.columns:
            # Separate date and time columns
            print("   Combining DATE and TIME columns...")
            if df['time'].dtype == 'object':
                # Handle time as string (e.g., "14:30:00")
                df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
            else:
                # Handle time as integer (e.g., 1430 for 14:30)
                time_str = df['time'].astype(str).str.zfill(4)  # Pad with zeros
                time_formatted = time_str.str[:2] + ':' + time_str.str[2:] + ':00'
                df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + time_formatted)
        elif 'date' in df.columns:
            # Only date column
            df['timestamp'] = pd.to_datetime(df['date'])
        else:
            print("❌ Error: No timestamp, date, or date+time columns found!")
            return None
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Ensure we have OHLC data
        required_price_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_price_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Error: Missing required price columns: {missing_columns}")
            return None
        
        # Handle volume column
        if 'tickvol' in df.columns:
            df['volume'] = df['tickvol']
        elif 'volume' in df.columns:
            df['volume'] = df['volume']
        elif 'vol' in df.columns:
            df['volume'] = df['vol']
        else:
            df['volume'] = 1000  # Default volume if not available
            print("⚠️  No volume data found, using default values")
        
        # Keep only OHLCV data
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Convert to numeric (handle any string numbers)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Validate data - use newer pandas methods
        if df.isnull().sum().sum() > 0:
            print("⚠️  Found NaN values, filling...")
            df = df.bfill().ffill()
        
        # Remove any remaining NaN rows
        df = df.dropna()
        
        # Check for reasonable price ranges
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if df[col].min() <= 0:
                print(f"❌ Error: Found non-positive prices in {col}")
                return None
        
        # Ensure high >= low
        invalid_bars = df['high'] < df['low']
        if invalid_bars.sum() > 0:
            print(f"⚠️  Found {invalid_bars.sum()} bars with high < low, fixing...")
            df.loc[invalid_bars, 'high'] = df.loc[invalid_bars, 'low']
        
        # Sort by timestamp to ensure chronological order
        df = df.sort_index()
        
        print(f"   Final data shape: {df.shape}")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")
        print(f"   Price range: {df['close'].min():.5f} to {df['close'].max():.5f}")
        print(f"   Data quality: ✅ Good")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_train_val_test_splits(X, y_reg, y_class):
    """Create train, validation, and test splits"""
    print(f"📊 Creating data splits...")
    
    total_samples = len(X)
    train_end = int(total_samples * TRAIN_SPLIT)
    val_end = int(total_samples * (TRAIN_SPLIT + VALIDATION_SPLIT))
    
    # Split the data
    X_train = X[:train_end]
    y_reg_train = y_reg[:train_end]
    y_class_train = y_class[:train_end]
    
    X_val = X[train_end:val_end]
    y_reg_val = y_reg[train_end:val_end]
    y_class_val = y_class[train_end:val_end]
    
    X_test = X[val_end:]
    y_reg_test = y_reg[val_end:]
    y_class_test = y_class[val_end:]
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Testing samples: {len(X_test)}")
    
    return (X_train, y_reg_train, y_class_train, 
            X_val, y_reg_val, y_class_val,
            X_test, y_reg_test, y_class_test)

def create_data_loaders(X_train, y_reg_train, y_class_train, 
                       X_val, y_reg_val, y_class_val,
                       X_test, y_reg_test, y_class_test):
    """Create PyTorch data loaders with feature scaling"""
    print(f"⚙️  Creating data loaders and scaling features...")
    
    # Scale features
    scaler = StandardScaler()
    
    # Reshape for scaling: (samples * sequence_length, features)
    X_train_flat = X_train.reshape(-1, FEATURE_COUNT)
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    
    X_val_flat = X_val.reshape(-1, FEATURE_COUNT)
    X_val_scaled = scaler.transform(X_val_flat)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)
    
    X_test_flat = X_test.reshape(-1, FEATURE_COUNT)
    X_test_scaled = scaler.transform(X_test_flat)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    # Convert to PyTorch tensors
    datasets = {}
    for name, (X, y_reg, y_class) in [
        ('train', (X_train_scaled, y_reg_train, y_class_train)),
        ('val', (X_val_scaled, y_reg_val, y_class_val)),
        ('test', (X_test_scaled, y_reg_test, y_class_test))
    ]:
        X_tensor = torch.FloatTensor(X)
        y_reg_tensor = torch.FloatTensor(y_reg)
        y_class_tensor = torch.FloatTensor(y_class)
        
        dataset = TensorDataset(X_tensor, y_reg_tensor, y_class_tensor)
        shuffle = (name == 'train')
        datasets[name] = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    
    print(f"✅ Data loaders created successfully")
    return datasets['train'], datasets['val'], datasets['test'], scaler

def initialize_model():
    """Initialize the LSTM model"""
    print(f"🧠 Initializing model...")
    
    model = CombinedLSTM(
        input_size=FEATURE_COUNT,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        num_regression_outputs=PREDICTION_STEPS
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model

def train_epoch(model, train_loader, optimizer, regression_criterion, classification_criterion):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_class_loss = 0.0
    batch_count = 0
    
    for batch_X, batch_y_reg, batch_y_class in train_loader:
        batch_X = batch_X.to(DEVICE)
        batch_y_reg = batch_y_reg.to(DEVICE)
        batch_y_class = batch_y_class.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass
        reg_output, class_logits = model(batch_X)
        
        # Calculate losses
        reg_loss = regression_criterion(reg_output, batch_y_reg)
        
        # Convert one-hot to class indices for classification loss
        class_targets = torch.argmax(batch_y_class, dim=1)
        class_loss = classification_criterion(class_logits, class_targets)
        
        # L2 regularization
        l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
        
        # Combined loss
        total_batch_loss = (REGRESSION_WEIGHT * reg_loss + 
                           CLASSIFICATION_WEIGHT * class_loss + 
                           L2_REGULARIZATION * l2_reg)
        
        # Backward pass
        total_batch_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        
        optimizer.step()
        
        total_loss += total_batch_loss.item()
        total_reg_loss += reg_loss.item()
        total_class_loss += class_loss.item()
        batch_count += 1
    
    return total_loss / batch_count, total_reg_loss / batch_count, total_class_loss / batch_count

def validate_epoch(model, val_loader, regression_criterion, classification_criterion):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_reg_loss = 0.0
    total_class_loss = 0.0
    batch_count = 0
    
    all_class_preds = []
    all_class_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y_reg, batch_y_class in val_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y_reg = batch_y_reg.to(DEVICE)
            batch_y_class = batch_y_class.to(DEVICE)
            
            # Forward pass
            reg_output, class_logits = model(batch_X)
            
            # Calculate losses
            reg_loss = regression_criterion(reg_output, batch_y_reg)
            
            class_targets = torch.argmax(batch_y_class, dim=1)
            class_loss = classification_criterion(class_logits, class_targets)
            
            # L2 regularization
            l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
            
            total_batch_loss = (REGRESSION_WEIGHT * reg_loss + 
                               CLASSIFICATION_WEIGHT * class_loss + 
                               L2_REGULARIZATION * l2_reg)
            
            total_loss += total_batch_loss.item()
            total_reg_loss += reg_loss.item()
            total_class_loss += class_loss.item()
            batch_count += 1
            
            # Store predictions for accuracy calculation
            class_preds = torch.argmax(class_logits, dim=1)
            all_class_preds.extend(class_preds.cpu().numpy())
            all_class_targets.extend(class_targets.cpu().numpy())
    
    # Calculate classification accuracy
    accuracy = accuracy_score(all_class_targets, all_class_preds)
    
    return total_loss / batch_count, total_reg_loss / batch_count, total_class_loss / batch_count, accuracy

def train_model(model, train_loader, val_loader):
    """Main training loop"""
    print(f"🎯 Starting training for {EPOCHS} epochs...")
    
    # Loss functions
    regression_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_reg_loss': [], 'val_reg_loss': [],
        'train_class_loss': [], 'val_class_loss': [],
        'val_accuracy': [], 'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        
        # Training
        train_loss, train_reg_loss, train_class_loss = train_epoch(
            model, train_loader, optimizer, regression_criterion, classification_criterion
        )
        
        # Validation
        val_loss, val_reg_loss, val_class_loss, val_accuracy = validate_epoch(
            model, val_loader, regression_criterion, classification_criterion
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_reg_loss'].append(train_reg_loss)
        history['val_reg_loss'].append(val_reg_loss)
        history['train_class_loss'].append(train_class_loss)
        history['val_class_loss'].append(val_class_loss)
        history['val_accuracy'].append(val_accuracy)
        history['learning_rate'].append(current_lr)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            os.makedirs("models", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'model_config': {
                    'input_size': FEATURE_COUNT,
                    'hidden_size': HIDDEN_SIZE,
                    'num_layers': NUM_LAYERS,
                    'num_classes': NUM_CLASSES,
                    'num_regression_outputs': PREDICTION_STEPS
                }
            }, 'models/best_model.pth')
        else:
            patience_counter += 1
        
        epoch_time = time.time() - epoch_start_time
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_loss:.6f}, Reg: {train_reg_loss:.6f}, Class: {train_class_loss:.6f}")
            print(f"  Val   - Loss: {val_loss:.6f}, Reg: {val_reg_loss:.6f}, Class: {val_class_loss:.6f}, Acc: {val_accuracy:.3f}")
            print(f"  LR: {current_lr:.2e}, Best Val Loss: {best_val_loss:.6f}")
            
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time:.1f} seconds")
    print(f"   Best validation loss: {best_val_loss:.6f}")
    
    return history

def evaluate_model(model, test_loader):
    """Evaluate the model on test data"""
    print(f"📊 Evaluating model on test data...")
    
    model.eval()
    all_reg_preds = []
    all_reg_targets = []
    all_class_preds = []
    all_class_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y_reg, batch_y_class in test_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y_reg = batch_y_reg.to(DEVICE)
            batch_y_class = batch_y_class.to(DEVICE)
            
            reg_output, class_logits = model(batch_X)
            
            # Store predictions
            all_reg_preds.append(reg_output.cpu().numpy())
            all_reg_targets.append(batch_y_reg.cpu().numpy())
            
            class_preds = torch.argmax(class_logits, dim=1)
            class_targets = torch.argmax(batch_y_class, dim=1)
            all_class_preds.extend(class_preds.cpu().numpy())
            all_class_targets.extend(class_targets.cpu().numpy())
    
    # Concatenate all predictions
    all_reg_preds = np.concatenate(all_reg_preds, axis=0)
    all_reg_targets = np.concatenate(all_reg_targets, axis=0)
    
    # Calculate metrics
    mse = np.mean((all_reg_preds - all_reg_targets) ** 2)
    mae = np.mean(np.abs(all_reg_preds - all_reg_targets))
    test_accuracy = accuracy_score(all_class_targets, all_class_preds)
    
    print(f"📈 Test Results:")
    print(f"   Regression MSE: {mse:.8f}")
    print(f"   Regression MAE: {mae:.8f}")
    print(f"   Classification Accuracy: {test_accuracy:.3f}")
    
    # Detailed classification report
    print(f"\n📊 Classification Report:")
    class_names = ['Hold', 'Sell', 'Buy']
    print(classification_report(all_class_targets, all_class_preds, 
                              target_names=class_names, digits=3))
    
    return {
        'mse': mse,
        'mae': mae,
        'accuracy': test_accuracy,
        'reg_predictions': all_reg_preds,
        'reg_targets': all_reg_targets,
        'class_predictions': all_class_preds,
        'class_targets': all_class_targets
    }

def save_model_and_scaler(model, scaler, history, test_results):
    """Save the trained model, scaler, and training information"""
    print(f"💾 Saving model and training artifacts...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Save the final model (required by daemon.py)
    model_path = "models/lstm_model_regression.pth"
    torch.save({
        'model_state': model.state_dict(),
        'model_config': {
            'input_size': FEATURE_COUNT,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'num_classes': NUM_CLASSES,
            'num_regression_outputs': PREDICTION_STEPS
        },
        'training_config': {
            'sequence_length': SEQUENCE_LENGTH,
            'prediction_steps': PREDICTION_STEPS,
            'feature_count': FEATURE_COUNT,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE
        },
        'test_results': test_results
    }, model_path)
    
    # Save scaler (required by daemon.py)
    scaler_path = "models/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    # Save training history
    history_path = "models/training_history.pkl"
    joblib.dump(history, history_path)
    
    print(f"✅ Saved files:")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    print(f"   History: {history_path}")

def plot_training_history(history):
    """Plot training history if matplotlib is available"""
    if not MATPLOTLIB_AVAILABLE:
        print(f"📊 Skipping plots (matplotlib not available)")
        return
    
    print(f"📊 Creating training plots...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Training Loss', alpha=0.8)
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', alpha=0.8)
        axes[0, 0].set_title('Combined Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Regression loss
        axes[0, 1].plot(history['train_reg_loss'], label='Training Reg Loss', alpha=0.8)
        axes[0, 1].plot(history['val_reg_loss'], label='Validation Reg Loss', alpha=0.8)
        axes[0, 1].set_title('Regression Loss (MSE)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Classification loss and accuracy
        axes[1, 0].plot(history['train_class_loss'], label='Training Class Loss', alpha=0.8)
        axes[1, 0].plot(history['val_class_loss'], label='Validation Class Loss', alpha=0.8)
        axes[1, 0].set_title('Classification Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('CrossEntropy Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Validation accuracy and learning rate
        ax1 = axes[1, 1]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(history['val_accuracy'], 'b-', label='Validation Accuracy', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        
        line2 = ax2.plot(history['learning_rate'], 'r-', label='Learning Rate', alpha=0.8)
        ax2.set_ylabel('Learning Rate', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_yscale('log')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
        
        ax1.set_title('Validation Accuracy & Learning Rate')
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        print(f"   Saved training plots: models/training_history.png")
        plt.close()
            
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")

def main():
    """Main training function"""
    print("🚀 LSTM Trading Model Training Pipeline")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Load and validate data
    data_file = "EURUSD60.csv"
    df = load_and_validate_data(data_file)
    if df is None:
        print("❌ Failed to load data. Exiting.")
        print("\n💡 Make sure EURUSD60.csv exists with the correct format:")
        print("   Expected columns: DATE, TIME, OPEN, HIGH, LOW, CLOSE, TICKVOL, VOL, SPREAD")
        return False
    
    # Check minimum data requirements
    min_required_rows = SEQUENCE_LENGTH + PREDICTION_STEPS + 100
    if len(df) < min_required_rows:
        print(f"❌ Insufficient data: {len(df)} rows, need at least {min_required_rows}")
        return False
    
    try:
        # Prepare training data
        print(f"⚙️  Preparing training data...")
        X, y_regression, y_classification = prepare_training_data(
            df, 
            sequence_length=SEQUENCE_LENGTH, 
            prediction_steps=PREDICTION_STEPS
        )
        
        print(f"✅ Data preparation complete:")
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Regression targets shape: {y_regression.shape}")
        print(f"   Classification targets shape: {y_classification.shape}")
        
        # Create train/validation/test splits
        splits = create_train_val_test_splits(X, y_regression, y_classification)
        X_train, y_reg_train, y_class_train, X_val, y_reg_val, y_class_val, X_test, y_reg_test, y_class_test = splits
        
        # Create data loaders
        train_loader, val_loader, test_loader, scaler = create_data_loaders(
            X_train, y_reg_train, y_class_train,
            X_val, y_reg_val, y_class_val,
            X_test, y_reg_test, y_class_test
        )
        
        # Initialize model
        model = initialize_model()
        
        # Train model
        history = train_model(model, train_loader, val_loader)
        
        # Load best model for evaluation
        best_model_path = 'models/best_model.pth'
        if os.path.exists(best_model_path):
            print("📥 Loading best model for final evaluation...")
            checkpoint = torch.load(best_model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state'])
        
        # Evaluate model
        test_results = evaluate_model(model, test_loader)
        
        # Save everything
        save_model_and_scaler(model, scaler, history, test_results)
        
        # Plot training history
        plot_training_history(history)
        
        print("\n🎉 Training completed successfully!")
        print("=" * 60)
        print("📋 Summary:")
        print(f"   Final test accuracy: {test_results['accuracy']:.3f}")
        print(f"   Final test MSE: {test_results['mse']:.8f}")
        print(f"   Model saved: models/lstm_model_regression.pth")
        print(f"   Scaler saved: models/scaler.pkl")
        print("\n🚀 Next steps:")
        print("   1. Run: python daemon_final.py")
        print("   2. Test with your MT5 EA")
        print("   3. Monitor performance and retrain as needed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("1. Run 'python fix_system.py' to fix import issues")
        print("2. Ensure EURUSD60.csv is in the correct format")
        print("3. Check that you have enough RAM (8GB+ recommended)")
        print("4. Install dependencies: pip install torch numpy pandas scikit-learn")
        sys.exit(1)