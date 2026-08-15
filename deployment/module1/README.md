# NeuroScan AI Module 1

## 1. Purpose

This package performs brain MRI image classification into:

- Glioma
- Meningioma
- No Tumor
- Pituitary

It uses the validated EfficientNetV2-S Module 1 classifier and is intended as a self-contained inference package for backend integration.

## 2. Model

- Model: EfficientNetV2-S
- Checkpoint: `models/efficientnetv2s_best.pt`
- Version: `module1_v1`
- Input: JPG, JPEG, or PNG image converted to RGB
- Image size: 224 x 224

## 3. Model Evaluation

Reference evaluation results on the official 1,600-image test set:

- Accuracy: 94.06%
- Macro-F1: 93.91%
- ROC-AUC: 0.9896
- Tumor Sensitivity: 96.00%
- Tumor Specificity: 100.00%

These are validation results for the research test set. They do not guarantee production or clinical performance.

## 4. Installation

```bash
pip install -r requirements-inference.txt
```

## 5. Python Usage

```python
from inference import TumorClassifier

classifier = TumorClassifier()

result = classifier.predict("sample.jpg")

print(result)
```

## 6. PIL Usage

```python
from PIL import Image
from inference import TumorClassifier

classifier = TumorClassifier()

image = Image.open("sample.jpg")

result = classifier.predict_pil(image)

print(result)
```

## 7. Output Format

```json
{
  "status": "success",
  "model": {
    "name": "EfficientNetV2-S",
    "version": "module1_v1"
  },
  "prediction": {
    "class": "Meningioma",
    "class_id": 1,
    "confidence": 0.9777439
  },
  "probabilities": {
    "Glioma": 0.0107414,
    "Meningioma": 0.9777439,
    "No Tumor": 0.0019034,
    "Pituitary": 0.0096111
  },
  "uncertainty": {
    "top1_probability": 0.9777439,
    "top2_probability": 0.0107414,
    "prediction_margin": 0.9670025
  }
}
```

All numeric values are normal Python floats and are JSON serializable with Flask `jsonify(result)`.

## 8. Device

If no device is supplied, CUDA is used when available; otherwise CPU is used.

```python
classifier = TumorClassifier(device="cuda")
classifier = TumorClassifier(device="cpu")
print(classifier.device)
```

## 9. Flask Integration

This package does not depend on Flask. A backend can integrate it like this:

```python
from flask import request, jsonify
from PIL import Image
from inference import TumorClassifier

classifier = TumorClassifier()

@app.route("/api/module1/predict", methods=["POST"])
def predict():
    uploaded_file = request.files["image"]

    image = Image.open(uploaded_file).convert("RGB")

    result = classifier.predict_pil(image)

    return jsonify(result)
```

## 10. Model Loading Behavior

Instantiate `TumorClassifier` once per backend worker or process and reuse that instance across requests. The checkpoint is loaded when the classifier object is initialized, not on every prediction.

## 11. Error Handling

The predictor raises clear exceptions for:

- Missing configuration file
- Missing or invalid checkpoint
- Missing image path
- Unsupported file extension
- Invalid or undecodable image file
- Non-PIL input passed to `predict_pil`
- Model loading failures

## 12. Medical Disclaimer

This is a research and decision-support classification component. It is not a standalone clinical diagnosis system and must not replace professional medical interpretation.

## CLI

```bash
python -m inference.predictor --image sample.jpg
```

Optional arguments:

```bash
python -m inference.predictor --image sample.jpg --checkpoint models/efficientnetv2s_best.pt --config configs/inference.yaml --device cpu
```

## Notes

Grad-CAM and human review workflows are separate concerns and are not run by `predict()` or `predict_pil()`. This package does not include reinforcement learning, active learning, retraining, feedback storage, Flask, or FastAPI.
