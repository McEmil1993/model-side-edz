"""Check the actual bundled TFLite file, catalog and measured predictions on CPU."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from train import image_array, verify_splits
from validate import validate
from evaluate import classification_metrics
import numpy as np
import tensorflow as tf

def main():
    bundle=ROOT.parent/'app-side/assets/models'
    manifest=json.loads((bundle/'manifest.json').read_text())
    structural=validate(bundle/'model.tflite',manifest)
    reference=json.loads((ROOT/'reports'/manifest['version']/'export-report.json').read_text())
    assert structural['sha256']==reference['sha256'],'Bundled bytes differ from export'
    catalog=json.loads((ROOT.parent/'app-side/assets/data/catalog.json').read_text())
    assert set(manifest['labels']) <= {f['id'] for f in catalog['fish']}
    split=json.loads((ROOT/'data/splits.json').read_text());verify_splits(split)
    interpreter=tf.lite.Interpreter(model_path=str(bundle/'model.tflite'),num_threads=2);interpreter.allocate_tensors()
    def infer(array):
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'],array)
        interpreter.invoke()
        return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
    scores=np.array([infer(image_array(r['path'])[None]) for r in split['splits']['test']])
    truth=np.array([manifest['labels'].index(r['label']) for r in split['splits']['test']])
    metrics=classification_metrics(truth,scores.argmax(axis=1),manifest['labels'])
    accept=scores.max(axis=1)>=manifest['threshold']
    unknown=json.loads((ROOT/'data/test-negatives.json').read_text())
    false_accepts=[bool(infer(image_array(p)[None]).max()>=manifest['threshold']) for p in unknown]
    metrics.update(testCount=len(truth),acceptedCount=int(accept.sum()),acceptedFraction=float(accept.mean()),acceptedAccuracy=float((truth[accept]==scores.argmax(axis=1)[accept]).mean()) if accept.any() else None,negativeCount=len(unknown),negativeFalseAcceptRate=float(np.mean(false_accepts)))
    for key in ['testAccuracy','acceptedCount','negativeFalseAcceptRate']:
        assert metrics[key]==manifest['evaluation'][key],f'Actual TFLite metric differs: {key}'
    sample=image_array(split['splits']['test'][0]['path'])[None]
    for _ in range(5):infer(sample)
    durations=[]
    for _ in range(50):
        started=time.perf_counter();infer(sample);durations.append((time.perf_counter()-started)*1000)
    report={'artifact':structural,'classification':metrics,'benchmark':{'environment':'Windows laptop CPU, Python TensorFlow Lite/XNNPACK; not a Flutter phone measurement','threads':2,'warmups':5,'iterations':50,'medianInferenceMilliseconds':float(np.median(durations)),'p95InferenceMilliseconds':float(np.percentile(durations,95)),'excludes':'image decode, preprocessing, file I/O and Flutter overhead'},'catalogClassIdsVerified':True,'phonePreprocessingVerified':False,'productionReady':False}
    (ROOT/'reports'/manifest['version']/'bundle-verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
