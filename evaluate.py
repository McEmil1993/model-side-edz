"""Measure held-out quality. Threshold must be chosen with validation/negative data first."""
import argparse
import json
import hashlib
import math
from pathlib import Path

import numpy as np
from train import image_array, verify_splits
import tensorflow as tf


def classification_metrics(truth, predictions, labels):
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for expected, actual in zip(truth, predictions):
        matrix[expected, actual] += 1
    classes = []
    for i, label in enumerate(labels):
        tp = int(matrix[i, i])
        precision = tp / max(1, int(matrix[:, i].sum()))
        recall = tp / max(1, int(matrix[i, :].sum()))
        classes.append({"id": label, "support": int(matrix[i, :].sum()), "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-12, precision + recall)})
    return {"testAccuracy": float(np.trace(matrix) / max(1, matrix.sum())), "macroF1": float(np.mean([c["f1"] for c in classes])), "classes": classes, "confusionMatrix": matrix.tolist()}


def evaluate(run, splits, threshold, negatives=None):
    if not .5 <= threshold <= .99:
        raise ValueError("Threshold must be calibrated on validation data and lie in [.5,.99].")
    metadata = json.loads((run / "training.json").read_text())
    split = json.loads(splits.read_text())
    verify_splits(split)
    if metadata["datasetVersion"] != split["datasetVersion"] or metadata["labels"] != split["labels"]:
        raise ValueError("Split/label version differs from training.")
    calibration = json.loads((run / 'calibration.json').read_text())
    if calibration['datasetVersion'] != split['datasetVersion'] or calibration['threshold'] != threshold or calibration['modelSha256'] != hashlib.sha256((run / 'model.keras').read_bytes()).hexdigest():
        raise ValueError('Freeze validation-only calibration for this exact model before evaluating test.')
    model = tf.keras.models.load_model(run / "model.keras")
    truth, predictions, accepted = [], [], []
    for row in split["splits"]["test"]:
        scores = model(image_array(row["path"])[None], training=False).numpy()[0]
        truth.append(split["labels"].index(row["label"]))
        predictions.append(int(np.argmax(scores)))
        accepted.append(bool(scores.max() >= threshold))
    report = classification_metrics(truth, predictions, split["labels"])
    report.update(datasetVersion=split["datasetVersion"], threshold=threshold, testCount=len(truth), acceptedFraction=float(np.mean(accepted)), negativeCount=0, negativeFalseAcceptRate=None)
    correct=np.array(truth)==np.array(predictions)
    report['acceptedCount']=sum(accepted)
    report['acceptedAccuracy']=float(correct[np.array(accepted)].mean()) if any(accepted) else None
    # Image-level Wilson interval: capture clustering makes effective uncertainty
    # larger; do not interpret this as field certification.
    n=len(truth);p=float(correct.mean());z=1.96
    center=(p+z*z/(2*n))/(1+z*z/n)
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    report['imageAccuracy95PercentInterval']=[center-radius,center+radius]
    report['intervalCaveat']='Image-level interval only; grouped captures reduce effective sample size. Not Philippine field accuracy.'
    if negatives:
        paths = json.loads(negatives.read_text())
        if paths:
            negative_hashes=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]
            positive_hashes={r['sha256'] for rows in split['splits'].values() for r in rows}
            if set(negative_hashes).intersection(calibration['negativeHashes']) or set(negative_hashes).intersection(positive_hashes) or len(set(negative_hashes)) != len(paths):
                raise ValueError('Test negatives must be distinct from tuning negatives and every positive.')
            false_accepts = [bool(model(image_array(p)[None], training=False).numpy()[0].max() >= threshold) for p in paths]
            report.update(negativeCount=len(paths), negativeFalseAcceptRate=float(np.mean(false_accepts)))
            report['negativeFalseAcceptCount']=sum(false_accepts)
            report['negativeBreakdown']={category:{'count':sum(category in Path(p).parts for p in paths),'falseAcceptCount':sum(flag for p,flag in zip(paths,false_accepts) if category in Path(p).parts)} for category in sorted({Path(p).parent.name for p in paths})}
    gates=json.loads((Path(__file__).parent/'config/quality-gates.json').read_text())
    checks={key:report[key] is not None and report[key]>=gates[key] for key in ['testAccuracy','macroF1','acceptedAccuracy','acceptedFraction']}
    checks['minimumClassRecall']=min(c['recall'] for c in report['classes'])>=gates['minimumClassRecall']
    checks['negativeFalseAcceptRate']=report['negativeFalseAcceptRate'] is not None and report['negativeFalseAcceptRate']<=gates['negativeFalseAcceptRate']
    report['qualityGateResults']=checks
    report['numericGatesPassed']=all(checks.values())
    report['releaseStatus']='experimental; Philippine market and Flutter device validation pending'
    (run / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/run"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--negatives", type=Path, help="JSON array of held-out unsupported/non-fish image paths")
    args = parser.parse_args()
    evaluate(args.run, args.splits, args.threshold, args.negatives)
