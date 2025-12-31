import json
from pathlib import Path

import torch
import torch.nn as nn


class Life_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        m = 24
        self.net = nn.Sequential(
            nn.Conv2d(1, 2 * m, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * m),
            nn.ReLU(),
            nn.Conv2d(2 * m, m, kernel_size=1),
            nn.BatchNorm2d(m),
            nn.ReLU(),
            nn.Conv2d(m, 2 * m, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * m),
            nn.ReLU(),
            nn.Conv2d(2 * m, m, kernel_size=1),
            nn.BatchNorm2d(m),
            nn.ReLU(),
            nn.Conv2d(m, 2 * m, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * m),
            nn.ReLU(),
            nn.Conv2d(2 * m, m, kernel_size=1),
            nn.BatchNorm2d(m),
            nn.ReLU(),
            nn.Conv2d(m, 2 * m, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * m),
            nn.ReLU(),
            nn.Conv2d(2 * m, m, kernel_size=1),
            nn.BatchNorm2d(m),
            nn.ReLU(),
            nn.Conv2d(m, 2 * m, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * m),
            nn.ReLU(),
            nn.Conv2d(2 * m, m, kernel_size=1),
            nn.BatchNorm2d(m),
            nn.ReLU(),
            nn.Conv2d(m, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


def tensor_to_list(t):
    return t.detach().cpu().numpy().reshape(-1).tolist()


def conv_payload(state, prefix, padding):
    weight = state[f"{prefix}.weight"]
    bias = state[f"{prefix}.bias"]
    return {
        "weight": tensor_to_list(weight),
        "bias": tensor_to_list(bias),
        "shape": list(weight.shape),
        "padding": padding,
    }


def bn_payload(state, prefix, eps=1e-5):
    return {
        "weight": tensor_to_list(state[f"{prefix}.weight"]),
        "bias": tensor_to_list(state[f"{prefix}.bias"]),
        "mean": tensor_to_list(state[f"{prefix}.running_mean"]),
        "var": tensor_to_list(state[f"{prefix}.running_var"]),
        "eps": eps,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    model_path = root / "api" / "model_weights_2.pt"
    out_path = root / "frontend" / "src" / "model_weights.json"

    model = Life_CNN()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)

    conv_keys = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
    bn_keys = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]

    blocks = []
    for i in range(10):
        conv_prefix = f"net.{conv_keys[i]}"
        bn_prefix = f"net.{bn_keys[i]}"
        weight_shape = state[f"{conv_prefix}.weight"].shape
        padding = 1 if weight_shape[-1] == 3 else 0
        blocks.append({
            "conv": conv_payload(state, conv_prefix, padding),
            "bn": bn_payload(state, bn_prefix),
        })

    final_conv_prefix = f"net.{conv_keys[-1]}"
    final = {
        "conv": conv_payload(state, final_conv_prefix, 0),
    }

    payload = {
        "blocks": blocks,
        "final": final,
        "size": 32,
    }

    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
