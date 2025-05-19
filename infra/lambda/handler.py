import json
import numpy as np
import onnxruntime as ort
from life_sim import Grid

# Cold-start: load the ONNX model once
sess = ort.InferenceSession("/var/task/life.onnx", providers=["CPUExecutionProvider"])

def handler(event, context):
    """
    HTTP API v2 (rawPath) or REST API (path) -> /predict or /simulate
    """
    path = event.get("rawPath") or event.get("path", "")
    data = json.loads(event.get("body") or "{}")

    if path.endswith("/predict"):
        # prepare a [1,1,H,W] float32 tensor
        grid = np.array(data.get("grid", []), dtype=np.float32)[None, None, ...]
        # run the ONNX graph
        out  = sess.run(None, {"x": grid})[0]       # shape [1,1,H,W]
        pred = (out[0,0] > 0.5).astype(np.uint8).tolist()
        body = {"prediction": pred}

    else:  # assume /simulate
        raw   = np.array(data.get("grid", []), dtype=bool)
        steps = int(data.get("steps", 5))
        g     = Grid(raw.shape[0], grid=raw)

        sims = []
        for _ in range(steps):
            g.step()
            sims.append(g.grid.astype(np.uint8).tolist())
        body = {"simulations": sims}

    return {
        "statusCode": 200,
        "headers":    {"Content-Type": "application/json"},
        "body":       json.dumps(body),
    }