"""Package a measured model; compare Keras/TFLite predictions before release."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from train import image_array, verify_splits
import tensorflow as tf
from validate import validate


def export(run, splits_path, version, float16=False):
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Semantic version required.")
    metadata = json.loads((run / "training.json").read_text())
    evaluation = json.loads((run / "evaluation.json").read_text())
    splits = json.loads(splits_path.read_text())
    verify_splits(splits)
    if any(value != metadata["datasetVersion"] for value in [evaluation["datasetVersion"], splits["datasetVersion"]]):
        raise ValueError("Evaluation/split provenance mismatch.")
    if evaluation["negativeCount"] < 1 or evaluation["negativeFalseAcceptRate"] is None:
        raise ValueError("Held-out unsupported/non-fish evaluation required before export.")
    model = tf.keras.models.load_model(run / "model.keras")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    artifact = converter.convert()
    path = run / "model.tflite"
    path.write_bytes(artifact)
    manifest = {"version": version, "labels": metadata["labels"], "inputSize": 224, "inputType": "float32", "outputType": "float32", "normalization": "raw_0_255", "resize": "stretch_bilinear", "colorOrder": "RGB", "threshold": evaluation["threshold"], "minAppBuild": 1, "datasetVersion": metadata["datasetVersion"], "evaluation": evaluation}
    structural = validate(path, manifest)
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    differences, agreements = [], []
    for row in splits["splits"]["test"]:
        image = image_array(row["path"])[None]
        original = model(image, training=False).numpy()
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], image)
        interpreter.invoke()
        converted = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
        differences.append(float(np.max(np.abs(original - converted))))
        agreements.append(bool(original.argmax() == converted.argmax()))
    if not agreements or not all(agreements) or max(differences) > (.03 if float16 else .001):
        raise ValueError("Keras/TFLite golden-image comparison failed; do not publish.")
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run / "export-report.json").write_text(json.dumps({**structural, "goldenCount": len(agreements), "maxOutputDifference": max(differences)}, indent=2))
    (run / "MODEL_CARD.md").write_text(f"# Fish2Dish model {version}\n\nLabels: {', '.join(metadata['labels'])}\n\nDataset: {metadata['datasetVersion']}\n\nArtifact SHA-256: {hashlib.sha256(artifact).hexdigest()}\n\nReview evaluation.json, class-level weaknesses, licensing, negative-image behavior and real-phone performance before publication. This model does not assess freshness or edibility.\n")
    print("Export complete. Human evaluation review and real-device checks still required.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/run"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--float16", action="store_true")
    args = parser.parse_args()
    export(args.run, args.splits, args.version, args.float16)
