# NeuroScan AI Module 0

Module 0 is an input-domain gatekeeper. It predicts whether an uploaded image is a brain MRI before the frozen Module 1 tumor classifier is allowed to run.

Module 0 does not diagnose a tumor. Module 1 does not determine whether an arbitrary image is an MRI. The combined output is an AI-assisted brain MRI classification and tumor category prediction, not a clinical diagnosis.

## Dataset

Keep the dataset exactly under:

```text
dataset/Train/brainmri_train
dataset/Train/nonbrainmri_train
dataset/Test/brainmri_test
dataset/Test/nonbrainmri_test
```

Labels come from directories only:

```text
non_brain_mri = 0
brain_mri = 1
```

The official Test folders are used only by `src.evaluate` after training. Validation is a stratified 20 percent split from `dataset/Train` with seed `42`, saved to `results/split_indices.csv`.

## Model

Module 0 uses torchvision EfficientNet-B0 with ImageNet weights and a two-class classifier head. Training uses ImageNet normalization, deterministic validation/test preprocessing, and mild training augmentation: resize to 224, horizontal flip, small rotation, mild affine shift/scale, and mild brightness/contrast jitter.

The class imbalance is handled with class-weighted cross entropy.

## Train

Run from this directory:

```bash
python -m src.train --config configs/baseline.yaml
```

Outputs:

```text
models/efficientnet_b0_best.pt
results/split_indices.csv
results/training_history.csv
results/training_curves.png
results/threshold_selection.json
configs/inference.yaml
```

The deployment threshold is selected from validation probabilities only. The selector prioritizes non-brain-MRI specificity while maintaining the configured minimum brain-MRI sensitivity when possible.

## Evaluate

```bash
python -m src.evaluate --config configs/baseline.yaml --inference-config configs/inference.yaml
```

Outputs:

```text
results/evaluation/metrics.json
results/evaluation/classification_report.txt
results/evaluation/confusion_matrix.png
results/evaluation/roc_curve.png
results/evaluation/test_predictions.csv
```

The evaluation report includes the number of non-brain-MRI images incorrectly accepted as brain MRI.

## Module 0 Inference

```bash
python -m inference.predictor --image dataset/Test/brainmri_test/<sample>.jpg
```

Python API:

```python
from inference.predictor import MRIGatePredictor

predictor = MRIGatePredictor()
result = predictor.predict("path/to/image.jpg")
```

Accepted file types are `.jpg`, `.jpeg`, and `.png`. The predictor verifies existence, file type, image decoding, dimensions, RGB conversion, checkpoint loading, CPU fallback, and CUDA when available.

## End-To-End Pipeline

Module 1 is treated as a frozen black-box downstream component located at `../module1`. The pipeline imports and calls its existing `TumorClassifier` only after Module 0 accepts the image.

```bash
python -m inference.pipeline --image dataset/Test/brainmri_test/<sample>.jpg
```

Optional overrides:

```bash
python -m inference.pipeline \
  --image <IMAGE_PATH> \
  --module0-config configs/inference.yaml \
  --module0-checkpoint models/efficientnet_b0_best.pt \
  --module1-config ../module1/configs/inference.yaml \
  --module1-checkpoint ../module1/models/efficientnetv2s_best.pt
```

Flask backend contract:

```python
from inference.pipeline import NeuroScanPipeline

pipeline = NeuroScanPipeline()
result = pipeline.predict(temp_path)
return jsonify(result)
```

For a rejected image, Module 1 is not loaded or executed. For an accepted image, the result contains the existing Module 1 JSON prediction under `module1`.

## Deployment Bundle

After training, build the compact Module 0 deployment folder:

```bash
python -m src.package_deployment
```

The bundle is written to:

```text
deployment/module0/
```

It contains Module 0 inference code, configuration, checkpoint when available, `requirements-inference.txt`, and this deployment README. It intentionally excludes training datasets, notebooks, intermediate results, and development artifacts.

## Tests

```bash
pytest -q
```

The test suite covers rejection gating, acceptance gating, invalid/missing/unsupported inputs, JSON serialization, checkpoint loading, CPU inference, and CUDA inference when available.

## Requirements

Install the training stack with:

```bash
pip install -r requirements.txt
```

For backend inference only:

```bash
pip install -r requirements-inference.txt
```

CUDA is used automatically when available. CPU inference is supported.

## Known Limitations

Module 0 detects whether an input resembles the training distribution for brain MRI images. It is not a clinical quality-control system and should not be used as a medical device. Low-quality, corrupted, out-of-distribution, or adversarial images may produce unreliable confidence scores.
