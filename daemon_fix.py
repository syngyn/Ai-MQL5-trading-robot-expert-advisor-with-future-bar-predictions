#!/usr/bin/env python3
"""
Quick Fix Script for LSTM Trading System
Fixes the "data directory not found" error and sets up the environment
"""

import os
import sys

def main():
    print("🔧 LSTM Trading System Quick Fix")
    print("=" * 40)
    
    # Get current directory
    current_dir = os.getcwd()
    print(f"📂 Working in: {current_dir}")
    print()
    
    # Create required directories
    directories = ["data", "models"]
    
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✅ Created directory: {directory}")
            except Exception as e:
                print(f"❌ Failed to create {directory}: {e}")
                return False
        else:
            print(f"✅ Directory exists: {directory}")
    
    print()
    print("🎯 Quick Fix Complete!")
    print()
    
    # Check if model files exist
    model_file = "models/lstm_model_regression.pth"
    scaler_file = "models/scaler.pkl"
    
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        print("✅ Model files found - you can run the daemon now!")
        print()
        print("💻 To start the daemon:")
        print("   python daemon_final.py")
        print()
    else:
        print("⚠️  Model files not found. You need to train the model first.")
        print()
        
        # Check for training data
        if os.path.exists("EURUSD60.csv"):
            print("✅ Training data found (EURUSD60.csv)")
            print()
            print("💻 To train the model:")
            print("   python train_model.py")
            print()
            print("💻 Then start the daemon:")
            print("   python daemon_final.py")
        else:
            print("❌ Training data not found (EURUSD60.csv)")
            print()
            print("📋 Steps to complete setup:")
            print("1. Get EURUSD H1 historical data and save as 'EURUSD60.csv'")
            print("   Format: DATE, TIME, OPEN, HIGH, LOW, CLOSE, TICKVOL, VOL, SPREAD")
            print("2. Train the model: python train_model.py")
            print("3. Start the daemon: python daemon_final.py")
    
    print()
    print("📁 Directory structure is now ready:")
    print(".")
    print("├── data/          (✅ Ready for EA communication)")
    print("├── models/        (✅ Ready for model files)")
    print("├── daemon_final.py")
    print("├── model.py")
    print("└── other files...")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Environment fixed! The daemon should run without the directory error.")
    else:
        print("\n❌ Fix failed. Please check permissions and try again.")
        sys.exit(1)