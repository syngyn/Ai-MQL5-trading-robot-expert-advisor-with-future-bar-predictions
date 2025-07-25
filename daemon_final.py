import os
import json
import time
import torch
import joblib
import numpy as np
from model import CombinedLSTM
import torch.nn.functional as F

class LSTMServer:
    def __init__(self, model_path="models/lstm_model_regression.pth", scaler_path="models/scaler.pkl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CombinedLSTM(
            input_size=15, hidden_size=128, num_layers=2,
            num_classes=3, num_regression_outputs=5
        )
        
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        self.model.to(self.device)
        self.model.eval()
        self.scaler_features = joblib.load(scaler_path)
        
        print("✅ Model and scaler loaded successfully")
        print(f"   Device: {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _get_combined_prediction(self, feature_window, current_price, atr):
        # Convert to numpy array and reshape properly
        features = np.array(feature_window, dtype=np.float32)
        
        # Check if features are flattened (1D array of 300 elements)
        if features.ndim == 1 and len(features) == 300:
            features = features.reshape(20, 15)
        elif features.ndim == 2 and features.shape == (20, 15):
            pass
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}. Expected (300,) or (20, 15)")
        
        # Scale features (expects 2D input)
        scaled = self.scaler_features.transform(features)
        
        # Reshape for LSTM: (1, sequence_length, features) = (1, 20, 15)
        tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            reg_output, class_logits = self.model(tensor)

        raw_predictions = reg_output.cpu().numpy()[0]
        mean_adjusted = raw_predictions - raw_predictions.mean()
        predicted_prices = current_price + (mean_adjusted * atr)

        # Calculate buy/sell probabilities from classification output
        probs = F.softmax(class_logits, dim=1).cpu().numpy()[0]
        buy_prob = float(probs[2])    # Buy class (index 2)
        sell_prob = float(probs[1])   # Sell class (index 1) 
        hold_prob = float(probs[0])   # Hold class (index 0)
        
        predicted_class = int(np.argmax(probs))
        confidence = float(np.max(probs))

        print("🔍 Raw model output:", raw_predictions)
        print("⚙️ Mean-adjusted deltas:", mean_adjusted)
        print("📈 Projected prices:", predicted_prices)
        print(f"🧠 Class = {predicted_class}, Confidence = {confidence:.2f}")
        print(f"📊 Probabilities - Buy: {buy_prob:.3f}, Sell: {sell_prob:.3f}, Hold: {hold_prob:.3f}")

        # Gating logic
        allowed = predicted_class == 2 and confidence >= 0.65
        print(f"🔐 Gating decision: {'✅ ALLOWED' if allowed else '❌ BLOCKED'}")

        if allowed:
            return predicted_prices.tolist(), predicted_class, confidence, buy_prob, sell_prob
        else:
            return [current_price] * len(raw_predictions), predicted_class, confidence, buy_prob, sell_prob

    def run(self):
        print("\n🚀 LSTM Daemon (Final Version w/ Indicators & Gating) Running...")
        while True:
            files = [f for f in os.listdir("data") if f.startswith("request_") and f.endswith(".json")]
            for file in files:
                try:
                    with open(f"data/{file}", "r") as f:
                        req = json.load(f)

                    features = req["features"]
                    current_price = req.get("current_price", 1.0)
                    atr = req.get("atr", 0.001)
                    
                    print(f"📥 Processing request {req['request_id']}")
                    print(f"   Features length: {len(features)}")
                    print(f"   Current price: {current_price}")
                    print(f"   ATR: {atr}")

                    pred_prices, pred_class, confidence, buy_prob, sell_prob = self._get_combined_prediction(features, current_price, atr)

                    resp = {
                        "request_id": req["request_id"],
                        "predicted_prices": pred_prices,
                        "predicted_class": pred_class,
                        "confidence_score": confidence,
                        "buy_probability": buy_prob,
                        "sell_probability": sell_prob
                    }

                    with open(f"data/response_{req['request_id']}.json", "w") as out:
                        json.dump(resp, out)

                    os.remove(f"data/{file}")
                    print(f"[{time.strftime('%H:%M:%S')}] ✅ Responded to {req['request_id']}")

                except Exception as e:
                    print(f"❌ Error handling {file}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            time.sleep(0.25)

if __name__ == "__main__":
    server = LSTMServer()
    server.run()
