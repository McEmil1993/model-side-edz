"""Second candidate: stronger augmentation and more feature adaptation, validation selection only."""
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train import dataset, verify_splits
import tensorflow as tf

def main():
    tf.keras.utils.set_random_seed(42)
    source=ROOT/'artifacts/market-v0.1.0';target=ROOT/'artifacts/tuned-v0.1.0'
    target.mkdir(parents=True,exist_ok=True)
    splits=json.loads((ROOT/'data/splits.json').read_text());verify_splits(splits)
    base=tf.keras.models.load_model(source/'model.keras')
    backbone=base.layers[1];classifier=base.layers[-1]
    backbone.trainable=True
    for layer in backbone.layers[:-100]:layer.trainable=False
    for layer in backbone.layers:
        if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
    inputs=tf.keras.Input((224,224,3))
    x=tf.keras.layers.RandomFlip('horizontal',seed=42)(inputs)
    x=tf.keras.layers.RandomRotation(.07,seed=43)(x)
    x=tf.keras.layers.RandomZoom(.15,.15,seed=44)(x)
    x=tf.keras.layers.RandomBrightness(.12,value_range=(0,255),seed=45)(x)
    scores=classifier(tf.keras.layers.Dropout(.3)(backbone(x,training=False)))
    model=tf.keras.Model(inputs,scores)
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-5),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    counts=[sum(r['label']==label for r in splits['splits']['train']) for label in splits['labels']]
    weights={i:len(splits['splits']['train'])/(len(counts)*count) for i,count in enumerate(counts)}
    started=time.monotonic()
    history=model.fit(dataset(splits['splits']['train'],splits['labels'],8,True),validation_data=dataset(splits['splits']['validation'],splits['labels'],8),class_weight=weights,epochs=50,verbose=2,callbacks=[tf.keras.callbacks.EarlyStopping(patience=8,restore_best_weights=True)])
    inference=tf.keras.Model(inputs,classifier(backbone(inputs,training=False)))
    inference.save(target/'model.keras')
    metadata=json.loads((source/'training.json').read_text())
    metadata.update(parentRun='market-v0.1.0',selectedStage='second_candidate_last_100_layers',selectionMetric='minimum validation loss; held-out test unused',candidateHistory=history.history,candidateElapsedSeconds=time.monotonic()-started,classWeights=weights)
    (target/'training.json').write_text(json.dumps(metadata,indent=2))
    print('Second candidate saved; compare validation before final held-out evaluation.',flush=True)

if __name__=='__main__':main()
