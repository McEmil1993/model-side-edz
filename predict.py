"""Run the real bundled offline classifier on one local photo."""
import argparse
import json
from pathlib import Path

from train import image_array
import tensorflow as tf
from validate import validate

def predict(image, model, manifest_path):
    manifest=json.loads(manifest_path.read_text())
    validate(model,manifest)
    runtime=tf.lite.Interpreter(model_path=str(model),num_threads=2)
    runtime.allocate_tensors()
    runtime.set_tensor(runtime.get_input_details()[0]['index'],image_array(image)[None])
    runtime.invoke()
    scores=runtime.get_tensor(runtime.get_output_details()[0]['index'])[0]
    best=int(scores.argmax())
    return {'fishId':manifest['labels'][best],'confidence':float(scores[best]),'uncertain':bool(scores[best]<manifest['threshold']),'modelVersion':manifest['version'],'experimental':True,'scores':{label:float(score) for label,score in zip(manifest['labels'],scores)}}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    bundle=Path(__file__).resolve().parent.parent/'app-side/assets/models'
    parser.add_argument('--model',type=Path,default=bundle/'model.tflite')
    parser.add_argument('--manifest',type=Path,default=bundle/'manifest.json')
    args=parser.parse_args();print(json.dumps(predict(args.image,args.model,args.manifest),indent=2))
