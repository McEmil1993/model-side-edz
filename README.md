# Fish2Dish Python model pipeline

An actual **experimental MobileNetV3-Small model 0.1.0 is trained and bundled** in `../app-side/assets/models/`. Ordered classes: bangus, dalag, hito, karpa, tilapia. Source data: 250 photos, 50 per class. Real held-out accuracy: **31/40 = 77.5%**, macro F1 **0.7602**. Numeric production gates failed. Dalag/hito confusion and very low accepted coverage remain serious limitations.

Read [the model card](reports/0.1.0/MODEL_CARD.md), [evaluation](reports/0.1.0/evaluation.json) and [actual bundled TFLite verification](reports/0.1.0/bundle-verification.json). Do not claim 100% accuracy from the seven accepted test predictions. At threshold 0.977, only 7/40 known fish photos were accepted, while 3/40 negative photos were wrongly accepted.

## Start here after cloning

`model-side` is a Python training/export pipeline. It has no web server, APK build or required Docker image of its own. Choose the workflow you need:

| Goal | Required work |
| --- | --- |
| Build admin Docker | Keep this folder beside admin-side; Docker copies `model-side/validate.py` and installs the validator dependencies itself. No training/dataset restore is required. App-side shared catalog/assets are also required by the current Dockerfile. |
| Use the existing Flutter model | Keep `app-side/assets/models/model.tflite` and `manifest.json` together. No retraining required. |
| Train a new model | Set up Python, restore/review photos, prepare splits, train, calibrate, evaluate and export below. |

Clone/place the folders as siblings when using the current Docker build or default Flutter model paths:

```text
maam-edz/
  admin-side/
  app-side/
  model-side/
```

If they are separate Git repositories, clone each into its matching sibling directory. The training pipeline itself can run in a standalone model-side checkout; default commands referencing `../app-side/` require that sibling or explicit `--model`/`--manifest` paths.

### 1. Python setup

The recorded setup used **64-bit Python 3.12** and CPU TensorFlow. Dependency installation and the first pretrained ImageNet weight download require internet. Commands below start from the model-side folder.

PowerShell (no activation or execution-policy change needed):

```powershell
cd model-side
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -c "import tensorflow as tf; print(tf.__version__)"
```

If `py` is unavailable, use your installed Python 3.12 executable instead. On Windows Git Bash:

```bash
cd model-side
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux/macOS, create with `python3.12 -m venv .venv`, activate with `source .venv/bin/activate`, then run the same pip commands. The previously verified environment is Windows/Python 3.12; other platforms require their own dependency/device validation.

### 2. Restore the ignored dataset

Git contains Python source, `requirements.txt`, `config/`, `dataset.csv`, source/license records and evaluation reports. **`data/`, `artifacts/` and trained weight files are ignored** and will not appear after cloning. CSV manifests describe images; they are not the image data.

Install **UnRAR or bsdtar with RAR support** before restoring Fish-gres photos. On Windows the script also detects `C:/Program Files/WinRAR/UnRAR.exe`. A GNU tar executable without RAR support is insufficient.

PowerShell, from model-side:

```powershell
./.venv/Scripts/python.exe tools/restore_selected_dataset.py
./.venv/Scripts/python.exe tools/restore_selected_dataset.py --verify-only
./.venv/Scripts/python.exe tools/prepare_training_review.py
```

In activated Git Bash/Linux/macOS, replace `./.venv/Scripts/python.exe` with `python` in the remaining commands.

The first command downloads the frozen 250 selected photos using tracked source URLs and checksums. The second verifies them; **`--verify-only` does not download missing images**. The third regenerates local grouped splits for this machine and updates the reviewed species registry. Inspect any tracked config changes before committing.

Source archives are larger than the final selected photos. A removed/changed remote photo or different rendering can fail checksum verification. Keep a private backup of the original `data/` directory; never silently substitute another image to bypass a failed checksum. After moving a dataset to another computer, rebuild splits because some recorded paths/dataset fingerprints are machine-specific.

### 3. Train, calibrate, evaluate and export

The example below creates a **new run** under `artifacts/run/`. Preserve an existing folder first or choose a fresh name and use it consistently. Training duration and measured accuracy can differ from the recorded baseline.

```powershell
./.venv/Scripts/python.exe train.py --splits data/splits.json --output artifacts/run --epochs 40 --fine-tune-epochs 12 --batch 8
./.venv/Scripts/python.exe tools/prepare_negative_images.py
```

Review the downloaded unsupported/non-fish contact sheets under `data/review/` and their source/license metadata before continuing. This helper can obtain different negative images on a fresh clone; it is not an exact restoration of the earlier baseline's negative evaluation set. For exact baseline reproduction, retain the frozen negative images/manifests and update their local paths deliberately.

Then freeze a threshold from validation, evaluate the held-out test set and export:

```powershell
./.venv/Scripts/python.exe calibrate.py --run artifacts/run --splits data/splits.json --negatives data/validation-negatives.json
$threshold = (Get-Content artifacts/run/calibration.json -Raw | ConvertFrom-Json).threshold
./.venv/Scripts/python.exe evaluate.py --run artifacts/run --splits data/splits.json --threshold $threshold --negatives data/test-negatives.json
./.venv/Scripts/python.exe export.py --run artifacts/run --splits data/splits.json --version 0.2.0 --float16
./.venv/Scripts/python.exe validate.py artifacts/run/model.tflite --manifest artifacts/run/manifest.json
```

`0.2.0` is an example new version; choose an unused version for each release. Do not copy the old `0.977` threshold onto new weights. In activated Git Bash/Linux/macOS, read the run's threshold with:

```bash
threshold=$(python -c 'import json; print(json.load(open("artifacts/run/calibration.json"))["threshold"])')
python evaluate.py --run artifacts/run --splits data/splits.json --threshold "$threshold" --negatives data/test-negatives.json
```

Use that command in place of the PowerShell threshold/evaluate lines; other commands have the same arguments. Do not select/tune a model using held-out test results. Read the generated calibration/evaluation evidence before publishing; a successful export/structural validation is not evidence of production accuracy.

### 4. Output and admin upload

| Path | Purpose |
| --- | --- |
| `artifacts/run/model.keras` | Trained Keras checkpoint |
| `artifacts/run/calibration.json` | Threshold for this exact model |
| `artifacts/run/evaluation.json` | Measured held-out results |
| `artifacts/run/model.tflite` | Offline mobile model to upload |
| `artifacts/run/manifest.json` | Matching labels, preprocessing and evaluation metadata |

In admin **Models & releases**, upload `model.tflite` together with its matching `manifest.json`, run Validate, review evaluation evidence, then explicitly publish with a catalog containing every model label. The current app's native model sync downloads compatible published releases; a new fish record alone does not train a new class. A draft upload is not automatically active. Keep a separate private backup/release artifact store because `.gitignore` excludes the generated weights.

To test the newly exported model locally without app-side:

```powershell
./.venv/Scripts/python.exe predict.py data/raw/bangus/bangus-001.png --model artifacts/run/model.tflite --manifest artifacts/run/manifest.json
./.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The sections below retain the detailed **historical 0.1.0 baseline** commands and evidence. Their `python` commands refer to the configured virtual environment; in an unactivated PowerShell use `./.venv/Scripts/python.exe`. Its fixed output paths and threshold are for that historical run, not a template for a newly trained release.

## Environment

From model-side in PowerShell:

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Use that environment's Python below. Actual training used existing Python 3.12.3 / TensorFlow 2.21.0 on the laptop CPU, batch 8 and limited threads. Two candidate training stages took approximately 50 and 25 seconds respectively, excluding the second process startup. Initial pretrained ImageNet weights download once; keep the Keras cache for offline retraining. Flutter was initially outside PATH; the later [APK task](../app-side/releases/README.md) built and verified this model on an offline Android emulator. Physical phone performance and camera/preprocessing checks remain unverified.

## Data and grouping

See [sources and licenses](DATASET_SOURCES.md), [offline photo gallery](data/review/index.html) and tracked [dataset.csv](dataset.csv). All 250 selected photos remain untouched in data/raw. Groups for training are more conservative than acquisition metadata: repeated Fish-gres backgrounds/setups remain together, and each iNaturalist observer stays together across dates. These are inferred capture groups, not certified specimen identities. Registry review means source and visual screening for experimental training, not expert certification.

The reviewed preparation script writes data/training-reviewed.csv and data/splits.json: **164 train / 46 validation / 40 held-out test**, each containing all five classes. The audit records [split assignments](reports/0.1.0/split-assignment.csv). The generic splitter rejects missing provenance, corrupt/undersized images, exact duplicates and coarse average-hash collisions across groups. Individually reviewed false-positive hash pairs require checksum-bound reasons in the registry; no general duplicate-check bypass exists.

```powershell
python tools/restore_selected_dataset.py --verify-only
python tools/prepare_training_review.py
python train.py --splits data/splits.json --output artifacts/market-v0.1.0 --epochs 40 --fine-tune-epochs 12 --batch 8
python tools/fine_tune_baseline.py
```

The second command overwrites the local split manifest. Preserve existing artifacts/evidence before new experiments. The fine-tuning script uses the first candidate, trains the last 100 backbone layers with BatchNorm frozen, adds augmentation/class weighting and selects its checkpoint by validation loss. The two candidates were compared by validation classification accuracy, selecting tuned-v0.1.0. See selection.json. Training scripts never use test pixels to select weights.

The recorded datasetVersion includes resolved local paths and metadata; relocating the workspace changes that version. Portable split-assignment.csv records original file names, checksums, groups and partitions. Retain original frozen local splits with the run for strict same-version verification.

## Calibrate, evaluate and export

Separate negative photos are licensed unsupported barramundi and house sparrow observations, partitioned by observer: 32 validation / 40 test. The helper downloads them and creates separate manifests/contact sheets. All 72 were visually checked; they are evaluation images, not additional training classes. This narrow negative set does not establish general unknown-image robustness.

```powershell
python tools/prepare_negative_images.py
python calibrate.py --run artifacts/tuned-v0.1.0
```

Calibration uses validation positives/negatives only and freezes a threshold with the exact model checksum. For the recorded run it selected **0.977**, maximizing accepted coverage subject to >=90% accepted accuracy and <=10% negative false accepts on validation. These are experimental score-selection targets, not probability calibration or field guarantees.

Choose the candidate and freeze calibration plus config/quality-gates.json before any final test. Record that choice in selection.json. The completed run's selection record is retained in reports/0.1.0/.

```powershell
python evaluate.py --run artifacts/tuned-v0.1.0 --splits data/splits.json --threshold 0.977 --negatives data/test-negatives.json
python export.py --run artifacts/tuned-v0.1.0 --splits data/splits.json --version 0.1.0 --float16
python tools/package_flutter_baseline.py --run artifacts/tuned-v0.1.0 --experimental
python tools/verify_flutter_bundle.py
```

The threshold above belongs only to this trained version. New weights need fresh validation calibration, not a copied number. This test set has now been inspected: use a new untouched Philippine-market test set when improving recognition quality. Do not tune against the reported test results.

Evaluation checks model/split/calibration provenance and positive-negative checksum separation. Export requires held-out negative evidence, structural validation and Keras/TFLite comparison on all held-out fish photos. Packaging refuses a failed-gate model unless explicitly marked experimental; it does not publish an online release. Evidence, individual negative credits and split assignments are retained outside ignored artifacts/data folders.

## Flutter artifact

Actual model: **1,948,220 bytes (1.86 MiB)**, float16 weights, float32 input/output. Input `[1,224,224,3]`, raw RGB 0-255, orientation normalized and bilinear stretch; normalization is inside the model. Output `[1,5]`, ordered manifest labels. Do not add divide-by-255 normalization. Model and manifest must stay together. The app catalog includes every output ID.

`verify_flutter_bundle.py` checked actual bundled bytes and class IDs, reran TFLite test/negative predictions and measured laptop inference. Tensor checks, all 40 held-out top-label comparisons, and key Keras/TFLite test metrics agree. Float16 altered one validation top-label; see model card. Score differences can exist, so do not treat conversion as automatically lossless.

The app already loads these bundled assets through ModelService; no mandatory online model download is required. After installing Flutter, run bootstrap, resolve dependencies, analyze/build, compare camera preprocessing against Python, and test a fresh install in airplane mode. A catalog-only sync must retain every bundled model class ID; incompatible catalogs are rejected locally. Do not publish the current baseline as a production accurate model.

Quick local prediction:

```powershell
python predict.py data/raw/bangus/bangus-001.png
```

This is an inference demonstration on an existing photo, not a new accuracy test. The CLI reports fishId, confidence, uncertain and all class scores. Scores are not guaranteed real-world correctness probabilities.

## Checks

```powershell
python -m unittest discover -s tests -v
python validate.py ../app-side/assets/models/model.tflite --manifest ../app-side/assets/models/manifest.json
```

Tests cover reviewed taxonomy, grouped split/provenance/tamper handling, class metrics, validation rejection calibration and real TFLite tensor contracts. Synthetic test images/models are structural fixtures only. Tensor validation is not recognition accuracy certification. TensorFlow's pinned Python TFLite interpreter works here but emits a deprecation warning; reassess runtime migration before upgrading.

Training implementation reference: [official Keras transfer-learning guidance](https://keras.io/guides/transfer_learning/).
