import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from simulator import Grid

# Define the CNN architecture
class Life_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        m = 24
        self.net = nn.Sequential(
            nn.Conv2d(1, 2*m, kernel_size=3, padding=1), nn.BatchNorm2d(2*m), nn.ReLU(),
            nn.Conv2d(2*m, m, kernel_size=1), nn.BatchNorm2d(m), nn.ReLU(),

            nn.Conv2d(m, 2*m, kernel_size=3, padding=1), nn.BatchNorm2d(2*m), nn.ReLU(),
            nn.Conv2d(2*m, m, kernel_size=1), nn.BatchNorm2d(m), nn.ReLU(),

            nn.Conv2d(m, 2*m, kernel_size=3, padding=1), nn.BatchNorm2d(2*m), nn.ReLU(),
            nn.Conv2d(2*m, m, kernel_size=1), nn.BatchNorm2d(m), nn.ReLU(),

            nn.Conv2d(m, 2*m, kernel_size=3, padding=1), nn.BatchNorm2d(2*m), nn.ReLU(),
            nn.Conv2d(2*m, m, kernel_size=1), nn.BatchNorm2d(m), nn.ReLU(),

            nn.Conv2d(m, 2*m, kernel_size=3, padding=1), nn.BatchNorm2d(2*m), nn.ReLU(),
            nn.Conv2d(2*m, m, kernel_size=1), nn.BatchNorm2d(m), nn.ReLU(),

            nn.Conv2d(m, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)

# Initialize model and load weights
torch_device = torch.device('cpu')
model = Life_CNN()
model.load_state_dict(torch.load('model_weights_2.pth', map_location=torch_device))
model.eval()

# Create Flask app
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON payload:
    { "grid": [[0,1,0,...], ...] }
    Returns: { "prediction": [[0,1,0,...], ...] }
    """
    data = request.get_json()
    grid = torch.tensor(data['grid'], dtype=torch.float32)
    input_tensor = grid.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred_binary = (output > 0.5).to(torch.uint8).squeeze().cpu().numpy().tolist()
    return jsonify({'prediction': pred_binary})

@app.route('/simulate', methods=['POST'])
def simulate():
    """
    Expects JSON payload:
    { "grid": [[0,1,0,...], ...], "steps": 5 }
    Returns:
    { "simulations": [ [[...],...], [[...],...], ... ] }
    """
    data = request.get_json()
    raw = np.array(data.get('grid', []), dtype=bool)
    steps = int(data.get('steps', 5))
    size = raw.shape[0]

    g = Grid(size, grid=raw)
    sims = []
    for _ in range(steps):
        g.step()
        sims.append(g.grid.astype(np.uint8).tolist())

    return jsonify({'simulations': sims})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
