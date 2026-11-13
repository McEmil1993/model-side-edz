"""CPU-friendly transfer-learning baseline; no model accuracy is assumed."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


def verify_splits(split):
    """Fail closed if split metadata or source bytes changed after preparation."""
    body = {key: value for key, value in split.items() if key != "datasetVersion"}
    actual = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if actual != split.get("datasetVersion"):
        raise ValueError("Split manifest was modified after preparation.")
    seen_groups = {}
    seen_images = set()
    for name, rows in split["splits"].items():
        for row in rows:
            if row["label"] not in split["labels"]:
                raise ValueError("Unknown class in split.")
            checksum = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
            if checksum != row["sha256"]:
                raise ValueError("Dataset image changed after preparation.")
            if checksum in seen_images:
                raise ValueError("Duplicate image in split manifest.")
            seen_images.add(checksum)
            if row["group"] in seen_groups and seen_groups[row["group"]] != name:
                raise ValueError("Capture group crosses split boundary.")
            seen_groups[row["group"]] = name


def image_array(path):
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)


def dataset(rows, labels, batch, training=False):
    # Small baseline fits in RAM as uint8 (~24 MB training images), avoiding
    # repeated JPEG decoding and unknown generator cardinality each epoch.
    images = np.stack([image_array(row['path']).astype(np.uint8) for row in rows])
    targets = np.array([labels.index(row['label']) for row in rows], dtype=np.int32)
    data = tf.data.Dataset.from_tensor_slices((images, targets))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    data = data.with_options(options)
    if training:
        data = data.shuffle(min(len(rows), 512), seed=42)
    return data.batch(batch).prefetch(1)


def train(splits_path, output, epochs=30, batch=8, fine_tune_epochs=12):
    tf.keras.utils.set_random_seed(42)
    split = json.loads(splits_path.read_text(encoding="utf-8"))
    verify_splits(split)
    labels = split["labels"]
    if any(not split["splits"][name] for name in ["train", "validation", "test"]):
        raise ValueError("Nonempty train, validation and held-out test required.")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    backbone = tf.keras.applications.MobileNetV3Small(input_shape=(224, 224, 3), include_top=False, weights="imagenet", include_preprocessing=True, pooling="avg")
    backbone.trainable = False
    inputs = tf.keras.Input((224, 224, 3), dtype=tf.float32)
    # Augmentation is training-only; exported inference has raw RGB 0–255 input.
    augmented = tf.keras.layers.RandomFlip("horizontal", seed=42)(inputs)
    augmented = tf.keras.layers.RandomRotation(.04, seed=42)(augmented)
    features = backbone(augmented, training=False)
    outputs = tf.keras.layers.Dense(len(labels), activation="softmax")(tf.keras.layers.Dropout(.2)(features))
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    training_data = dataset(split['splits']['train'], labels, batch, True)
    validation_data = dataset(split['splits']['validation'], labels, batch)
    history = model.fit(training_data, validation_data=validation_data, epochs=epochs, verbose=2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True), tf.keras.callbacks.ModelCheckpoint(str(output / "best.keras"), save_best_only=True)])
    head_loss = float(model.evaluate(validation_data, verbose=0)[0])
    head_weights = model.get_weights()
    fine_history = None
    selected_stage = 'frozen_backbone'
    if fine_tune_epochs:
        backbone.trainable = True
        # Only adapt the final feature blocks; freeze BatchNorm statistics on
        # this very small collection (backbone still called training=False).
        for layer in backbone.layers[:-40]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        fine_history = model.fit(training_data, validation_data=validation_data, epochs=fine_tune_epochs, verbose=2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
        if float(model.evaluate(validation_data, verbose=0)[0]) < head_loss:
            selected_stage = 'fine_tuned_final_blocks'
        else:
            model.set_weights(head_weights)
    # Export graph excludes augmentation/dropout entirely.
    inference_inputs = tf.keras.Input((224, 224, 3), dtype=tf.float32)
    inference = tf.keras.Model(inference_inputs, model.layers[-1](backbone(inference_inputs, training=False)))
    inference.save(output / "model.keras")
    config = {"labels": labels, "datasetVersion": split["datasetVersion"], "epochs": epochs, "fineTuneEpochs": fine_tune_epochs, "selectedStage": selected_stage, "selectionMetric": "minimum validation loss; held-out test unused", "batch": batch, "seed": 42, "elapsedSeconds": time.monotonic() - started, "tensorflow": tf.__version__, "history": history.history, "fineTuneHistory": fine_history.history if fine_history else None}
    (output / "training.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Training saved. Evaluate held-out data and calibrate rejection before export.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/run"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=12)
    args = parser.parse_args()
    train(args.splits, args.output, args.epochs, args.batch, args.fine_tune_epochs)
