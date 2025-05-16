import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import numpy as np

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
    grid = np.array(data['grid'], dtype=np.float32)

    # Prepare tensor: [batch, channel, H, W]
    input_tensor = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
        # Apply sigmoid if needed and threshold
        pred = output.squeeze().numpy()
        pred_binary = (pred > 0.5).astype(np.uint8)

    return jsonify({'prediction': pred_binary.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)