import json
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tensorflow as tf
from PIL import Image
from dataset import prepare
from evaluate import classification_metrics
from validate import validate
from train import verify_splits
from calibrate import choose_threshold


class PipelineTests(unittest.TestCase):
    def test_rejection_calibration_uses_accuracy_and_unknown_false_accepts(self):
        positives=np.array([[.95,.05],[.1,.9],[.4,.6],[.2,.8]])
        truth=np.array([0,1,0,1])
        negatives=np.array([[.7,.3],[.45,.55]])
        result,_=choose_threshold(positives,truth,negatives)
        self.assertTrue(result['validationRejectionTargetsMet'])
        self.assertGreater(result['threshold'],.7)
        self.assertEqual(result['acceptedCount'],3)
        self.assertEqual(result['acceptedAccuracy'],1)
        self.assertEqual(result['negativeFalseAcceptRate'],0)
        failed,_=choose_threshold(np.array([[.6,.4]]),np.array([0]),np.array([[1.,0.]]))
        self.assertFalse(failed['validationRejectionTargetsMet'])

    def test_grouped_split_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            registry=root/'registry.json'
            registry.write_text(json.dumps({'reviewed':True,'classes':[{'id':'a'},{'id':'b'}]}))
            rows=[]
            rng=np.random.default_rng(42)
            for label in ['a','b']:
                for group in range(3):
                    name=f'{label}-{group}.png'
                    Image.fromarray(rng.integers(0,256,(80,80,3),dtype=np.uint8)).save(root/name)
                    rows.append({'path':name,'label':label,'group':f'{label}-{group}','source':'Synthetic test fixture','license':'Test only'})
            csv_path=root/'images.csv'
            with csv_path.open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            prepared=prepare(csv_path,registry,root/'splits.json')
            verify_splits(prepared)
            self.assertEqual([len(prepared['splits'][key]) for key in ['train','validation','test']],[2,2,2])
            groups=[{row['group'] for row in split} for split in prepared['splits'].values()]
            self.assertTrue(groups[0].isdisjoint(groups[1]) and groups[1].isdisjoint(groups[2]) and groups[0].isdisjoint(groups[2]))
            with csv_path.open('a',newline='') as handle:
                csv.DictWriter(handle,fieldnames=list(rows[0])).writerow(rows[0])
            with self.assertRaisesRegex(ValueError,'Exact duplicate'):
                prepare(csv_path,registry,root/'other.json')
            (root/rows[0]['path']).write_bytes(b'changed image bytes')
            with self.assertRaisesRegex(ValueError,'image changed'):
                verify_splits(prepared)
    def test_unreviewed_taxonomy_cannot_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'registry.json').write_text(json.dumps({'reviewed': False}))
            with self.assertRaisesRegex(ValueError, 'Review taxonomy'):
                prepare(root / 'missing.csv', root / 'registry.json', root / 'output.json')

    def test_per_class_metrics_expose_weak_class(self):
        report = classification_metrics([0, 0, 1, 1], [0, 0, 0, 1], ['a', 'b'])
        self.assertAlmostEqual(report['testAccuracy'], .75)
        self.assertEqual(report['classes'][1]['recall'], .5)
        self.assertEqual(report['confusionMatrix'], [[2, 0], [1, 1]])

    def test_real_tflite_tensor_validation_and_label_mismatch(self):
        # Synthetic structural fixture only, never exported as a fish classifier.
        inputs = tf.keras.Input((224, 224, 3))
        features = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = tf.keras.layers.Dense(2, activation='softmax')(features)
        model = tf.keras.Model(inputs, outputs)
        artifact = tf.lite.TFLiteConverter.from_keras_model(model).convert()
        manifest = {'labels': ['a', 'b'], 'inputSize': 224, 'inputType': 'float32', 'outputType': 'float32', 'normalization': 'raw_0_255', 'resize': 'stretch_bilinear', 'colorOrder': 'RGB', 'minAppBuild': 1, 'threshold': .7}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.tflite'
            path.write_bytes(artifact)
            report = validate(path, manifest)
            self.assertTrue(report['valid'])
            self.assertFalse(report['accuracyVerified'])
            with self.assertRaisesRegex(ValueError, 'output tensor'):
                validate(path, {**manifest, 'labels': ['a', 'b', 'c']})
            with self.assertRaisesRegex(ValueError, 'manifest contract'):
                validate(path, {**manifest, 'normalization': 'minus_1_to_1'})


if __name__ == '__main__':
    unittest.main()
