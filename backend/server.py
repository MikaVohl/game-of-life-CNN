import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS

# Define the CNN architecture
class Life_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        m = 24
        self.net = nn.Sequential(
            nn.Conv2d(1, 2*m, kernel_size=3, padding=1), nn.BatchNorm2d(2*m), nn.ReLU(),
            nn.Conv2d(2*m,    m, kernel_size=1),         nn.BatchNorm2d(m),    nn.ReLU(),

            nn.Conv2d(m,   2*m, kernel_size=3, padding=1), nn.BatchNorm2d(2*m), nn.ReLU(),
            nn.Conv2d(2*m,    m, kernel_size=1),          nn.BatchNorm2d(m),    nn.ReLU(),

            nn.Conv2d(m,   2*m, kernel_size=3, padding=1), nn.BatchNorm2d(2*m), nn.ReLU(),
            nn.Conv2d(2*m,    m, kernel_size=1),          nn.BatchNorm2d(m),    nn.ReLU(),

            nn.Conv2d(m,   2*m, kernel_size=3, padding=1), nn.BatchNorm2d(2*m), nn.ReLU(),
            nn.Conv2d(2*m,    m, kernel_size=1),          nn.BatchNorm2d(m),    nn.ReLU(),

            nn.Conv2d(m,   2*m, kernel_size=3, padding=1), nn.BatchNorm2d(2*m), nn.ReLU(),
            nn.Conv2d(2*m,    m, kernel_size=1),          nn.BatchNorm2d(m),    nn.ReLU(),

            nn.Conv2d(m, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)

# Initialize model and load weights
model = Life_CNN()
model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
model.eval()

# Create Flask app
app = Flask(__name__)
CORS(app)
@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON payload:
    {
        "grid": [[0,1,0,...], ...]  # 2D list of 0s and 1s, size e.g. 64x64
    }
    Returns:
    {
        "prediction": [[0,1,0,...], ...]  # Predicted next state
    }
    """
    data = request.get_json()
    grid = torch.tensor(data['grid'], dtype=torch.float32)
    # Prepare tensor: [batch, channel, H, W]

    input_tensor = grid.unsqueeze(0).unsqueeze(0)
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
        # Apply sigmoid if needed and threshold
        pred_binary = (output > 0.5).to(torch.uint8).squeeze(0).squeeze(0)
        prediction = pred_binary.cpu().tolist()
    print(pred_binary)
    print(prediction)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)