# Fish2Dish experimental model 0.1.0

Trained November 12, 2026. Status: experimental development bundle; numeric quality gates failed. No online publication was performed.

Artifact: `app-side/assets/models/model.tflite`, 1,948,220 bytes (1.86 MiB).
SHA-256: `d74da76facdbd5f2c4e6a549c08712b5c803930ecbd533d8ed33866810ab1a6d`.
MobileNetV3-Small pretrained on ImageNet, five-class head, float16 weight conversion with float32 input/output. Inference is fully local.

## Scope and data

Outputs, in order: `bangus` (Chanos chanos), `dalag` (Channa striata), `hito` (Clarias gariepinus), `karpa` (Cyprinus carpio), `tilapia` (Oreochromis niloticus). Local names do not mean every related species is supported. Whole-fish photos were used; fillets, cooked food and mixed-species piles are outside scope. This classifier does not determine freshness, safety or edibility.

250 source-labeled photos, 50 per class. Bangus and tilapia come from Fish-gres V2 in Indonesia; the other classes from licensed research-grade iNaturalist records. This is not a Philippine-market field dataset. Labels were source-checked and visually screened, not expert-certified. Quarantined source-label concerns were excluded. See ../../DATASET_SOURCES.md, ../../dataset.csv and bundled credits.

Conservative grouping keeps repeated Fish-gres backgrounds/setups together and each iNaturalist observer together across dates. Four coarse average-hash collisions were visually reviewed as different taxa/scenes; exact checksum-bound reviews are recorded in config/species.json. No exact photo is shared across partitions. Independent specimen identity still cannot be certified.

| Class | Train | Validation | Held-out test |
| --- | ---: | ---: | ---: |
| Bangus | 34 | 8 | 8 |
| Dalag | 35 | 6 | 9 |
| Hito | 36 | 7 | 7 |
| Karpa | 35 | 10 | 5 |
| Tilapia | 24 | 15 | 11 |
| Total | 164 | 46 | 40 |

Separate licensed negatives: 32 validation and 40 test photos of unsupported Lates calcarifer and non-fish Passer domesticus. Observers remain in one negative partition. This limited negative set does not cover arbitrary phone images. Credits/licenses are retained in negative-image-provenance.json.

## Training and selection

Seed 42, batch 8, TensorFlow 2.21.0, Python 3.12, laptop CPU. First candidate: frozen backbone and optional final-block fine-tuning, minimum validation loss checkpoint. Second candidate: last 100 backbone layers available for tuning, BatchNorm frozen/inference mode, Adam 3e-5, class weighting, horizontal flip, small rotation/zoom/brightness augmentation, dropout and early stopping.

Two candidates were compared using validation only: 67.39% versus 78.26% Keras validation accuracy. The second candidate was selected. No test image pixels were used for training/checkpoint/threshold selection. See selection.json, training.json, calibration.json and split-assignment.csv. Test results must not be used to retune this version; further improvements need a new independent test.

## Held-out results

**31/40 correct = 77.5% top-1 accuracy**, macro F1 **0.7602**, before rejection. Image-level Wilson 95% interval: approximately **62.5-87.7%**. Capture clustering makes effective uncertainty larger; this is not an estimate of Philippine field accuracy. Some class test sets represent only one or a few capture groups.

| Class | Correct / photos | Recall | Precision |
| --- | ---: | ---: | ---: |
| Bangus | 8 / 8 | 100% | 100% |
| Dalag | 4 / 9 | 44.4% | 66.7% |
| Hito | 5 / 7 | 71.4% | 50.0% |
| Karpa | 4 / 5 | 80.0% | 66.7% |
| Tilapia | 10 / 11 | 90.9% | 100% |

Small class counts do not establish reliable per-species accuracy. Four dalag test photos were classified as hito; two hito photos as dalag.

Validation-only rejection chose threshold **0.977**, targeting accepted accuracy at least 90% and negative false accepts at most 10%, maximizing known-image coverage among feasible thresholds. Confidence is not calibrated field accuracy. On the final test, **7/40 known fish were accepted**, all seven correct; **33/40 were uncertain**. Do not present 7/7 as overall 100% accuracy. **3/40 negatives were falsely accepted (7.5%)**: 3/21 sparrow photos and 0/19 unsupported fish.

Provisional gates frozen before final test: top-1 >=85%, macro F1 >=0.8, every-class recall >=70%, accepted accuracy >=90%, accepted coverage >=50%, negative FAR <=10%. Top-1, macro F1, minimum recall and coverage failed. This model is not approved for production publication.

## Conversion and integration

Contract: input float32 `[1,224,224,3]`, raw RGB 0-255, orientation-corrected bilinear stretch; internal model rescaling. Output float32 `[1,5]` probabilities. No external divide-by-255 or -1..1 normalization. Manifest and model must remain together with matching catalog IDs. No extra unknown class exists; rejection uses the threshold and can still fail on unsupported inputs.

TFLite structural checks and all 40 held-out top-label comparisons passed. Maximum Keras/TFLite probability difference was **0.01421**. Actual bundled TFLite was separately measured: same 31/40 correct, 7 accepted/7 correct and 3/40 false accepts. Float16 changed one validation top-label relative to Keras (37/46 versus 36/46 correct); threshold was kept frozen, with the same validation acceptance counts.

Laptop benchmark: two CPU threads/XNNPACK, 5 warmups and 50 iterations, about **1.05 ms median / 1.36 ms p95** inference. Excludes image decode, preprocessing, I/O and Flutter overhead. This is not a phone performance or RAM measurement. See bundle-verification.json for exact recorded results.

Flutter SDK was unavailable: no Flutter analyze/build, cross-language golden preprocessing comparison, device latency or airplane-mode acceptance run. Pillow and Dart image resamplers can differ; phone verification remains required. All five IDs are present in the bundled catalog. Recipes remain the two original starter recipes.

## Next accuracy work

Collect independently labeled Philippine market photos with varied vendors, fish size, lighting, backgrounds and phones, especially dalag/hito and related species. Keep sessions/specimens/observers grouped, add diverse non-fish and unsupported fish negatives, and reserve a new untouched market test set before further training. Obtain species/content review, validate rejection coverage and false accepts, then run actual Flutter camera/device checks before publication.
