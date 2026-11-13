"""Choose rejection threshold using validation only; never read test image pixels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from train import image_array, verify_splits
import tensorflow as tf
from evaluate import classification_metrics

def choose_threshold(positive_scores, truth, negative_scores):
    predictions=positive_scores.argmax(axis=1)
    candidates=[]
    for threshold in np.linspace(.5,.99,491):
        accept=positive_scores.max(axis=1)>=threshold
        count=int(accept.sum())
        accuracy=float(np.mean(predictions[accept]==truth[accept])) if count else None
        far=float(np.mean(negative_scores.max(axis=1)>=threshold))
        candidates.append(dict(threshold=round(float(threshold),3),acceptedCount=count,acceptedFraction=float(accept.mean()),acceptedAccuracy=accuracy,negativeFalseAcceptRate=far))
    feasible=[c for c in candidates if c['acceptedCount'] and c['acceptedAccuracy']>=.9 and c['negativeFalseAcceptRate']<=.1]
    if feasible:
        selected=max(feasible,key=lambda c:(c['acceptedCount'],-c['negativeFalseAcceptRate'],c['threshold']))
    else:
        selected=min(candidates,key=lambda c:(c['negativeFalseAcceptRate'], -(c['acceptedAccuracy'] or 0),-c['acceptedCount'],c['threshold']))
    return {**selected,'validationRejectionTargetsMet':bool(feasible)},candidates

def calibrate(run, splits, negatives):
    split=json.loads(splits.read_text());verify_splits(split)
    metadata=json.loads((run/'training.json').read_text())
    if split['datasetVersion']!=metadata['datasetVersion']:raise ValueError('Training split mismatch')
    paths=json.loads(negatives.read_text())
    negative_hashes=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]
    positives={r['sha256'] for rows in split['splits'].values() for r in rows}
    if not paths or positives.intersection(negative_hashes):raise ValueError('Independent validation negatives required')
    model=tf.keras.models.load_model(run/'model.keras')
    rows=split['splits']['validation']
    scores=np.concatenate([model(np.stack([image_array(r['path']) for r in rows[i:i+8]]),training=False).numpy() for i in range(0,len(rows),8)])
    unknown=np.concatenate([model(np.stack([image_array(p) for p in paths[i:i+8]]),training=False).numpy() for i in range(0,len(paths),8)])
    truth=np.array([split['labels'].index(r['label']) for r in rows])
    selected,candidates=choose_threshold(scores,truth,unknown)
    report={**selected,'datasetVersion':split['datasetVersion'],'modelSha256':hashlib.sha256((run/'model.keras').read_bytes()).hexdigest(),'partition':'validation only','positiveCount':len(rows),'negativeCount':len(paths),'negativeHashes':negative_hashes,'classification':classification_metrics(truth,scores.argmax(axis=1),split['labels']),'search':candidates}
    (run/'calibration.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['search','negativeHashes']},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('artifacts/market-v0.1.0'))
    parser.add_argument('--splits',type=Path,default=Path('data/splits.json'))
    parser.add_argument('--negatives',type=Path,default=Path('data/validation-negatives.json'))
    args=parser.parse_args();calibrate(args.run,args.splits,args.negatives)
