"""Bounded model structural/smoke validation; this does not establish fish accuracy."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf


def validate(path: Path, manifest: dict) -> dict:
    labels = manifest.get("labels", [])
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("Invalid label mapping.")
    expected = {"inputSize": 224, "inputType": "float32", "outputType": "float32", "normalization": "raw_0_255", "resize": "stretch_bilinear", "colorOrder": "RGB", "minAppBuild": 1}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported manifest contract.")
    if not .5 <= manifest.get("threshold", 0) <= .99:
        raise ValueError("Invalid confidence threshold.")
    if not 0 < path.stat().st_size <= 30 * 1024 * 1024:
        raise ValueError("Model size outside limits.")
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("One input/output tensor required.")
    if inputs[0]["shape"].tolist() != [1, 224, 224, 3] or inputs[0]["dtype"] != np.float32:
        raise ValueError("Invalid input tensor.")
    if outputs[0]["shape"].tolist() != [1, len(labels)] or outputs[0]["dtype"] != np.float32:
        raise ValueError("Invalid output tensor or label count.")
    interpreter.set_tensor(inputs[0]["index"], np.zeros((1, 224, 224, 3), np.float32))
    interpreter.invoke()
    scores = interpreter.get_tensor(outputs[0]["index"])
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any() or not np.allclose(scores.sum(axis=1), 1, atol=1e-3):
        raise ValueError("Expected finite normalized class probabilities.")
    return {"valid": True, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size, "labels": labels, "inputShape": inputs[0]["shape"].tolist(), "outputShape": outputs[0]["shape"].tolist(), "accuracyVerified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.model, json.loads(args.manifest.read_text(encoding="utf-8")))))
