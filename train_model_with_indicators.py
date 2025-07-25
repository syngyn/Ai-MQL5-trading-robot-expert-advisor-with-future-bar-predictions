import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import dump
from model import create_lstm_model
from data_processing_with_indicators import load_and_align_data

# Paths
BASE_PATH = "data"
MODEL_PATH = "lstm_model.h5"
SCALER_PATH = "scaler.save"

# Load data
symbol_files = {"EURUSD": "EURUSD60.csv"}
df = load_and_align_data(symbol_files, BASE_PATH)

# Drop rows with NaNs
df.dropna(inplace=True)

# Features and target
feature_columns = [col for col in df.columns if col not in ["timestamp", "close"]]
target_column = "close"

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[feature_columns])
target = df[target_column].values

# Sequence creation
SEQ_LEN = 24
X, y = [], []
for i in range(SEQ_LEN, len(scaled_features)):
    X.append(scaled_features[i-SEQ_LEN:i])
    y.append(target[i])
X, y = np.array(X), np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create and train model
model = create_lstm_model((X.shape[1], X.shape[2]))
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save model and scaler
model.save(MODEL_PATH)
dump(scaler, SCALER_PATH)
print("Model and scaler saved successfully.")