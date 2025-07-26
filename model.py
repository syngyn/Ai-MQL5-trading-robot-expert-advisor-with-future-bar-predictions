import torch
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
