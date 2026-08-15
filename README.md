# NeuroScan AI — Brain MRI Validation and Tumor Classification Pipeline

NeuroScan AI is a two-stage computer vision inference pipeline designed for brain MRI image validation and brain tumor classification.

The pipeline prevents arbitrary or non-MRI images from reaching the tumor classification model.

## Pipeline

```text
JPG / JPEG IMAGE
       |
       v
+----------------------+
|       MODULE 0       |
|    MRI Gatekeeper    |
|    EfficientNet-B0   |
+----------+-----------+
           |
      Is it Brain MRI?
        /         \
      NO           YES
      |             |
      v             v
   REJECT      +----------------------+
               |       MODULE 1       |
               |  Tumor Classifier    |
               |   EfficientNetV2-S   |
               +----------+-----------+
                          |
                          v
               Glioma / Meningioma /
               No Tumor / Pituitary
```

## Modules

### Module 0 — MRI Input Validation

Classifies an uploaded image as:

- Brain MRI
- Non-Brain MRI

If rejected, Module 1 is not executed.

### Module 1 — Brain Tumor Classification

Classifies accepted brain MRI images into:

- Glioma
- Meningioma
- No Tumor
- Pituitary

## Repository Structure

```text
EffiNetDep/
├── deployment/
│   ├── module0/
│   │   ├── README.md
│   │   ├── configs/
│   │   │   └── inference.yaml
│   │   ├── inference/
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   ├── pipeline.py
│   │   │   ├── predictor.py
│   │   │   └── transforms.py
│   │   ├── models/
│   │   │   └── efficientnet_b0_best.pt
│   │   └── requirements-inference.txt
│   └── module1/
│       ├── README.md
│       ├── configs/
│       │   └── inference.yaml
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── model.py
│       │   ├── predictor.py
│       │   └── transforms.py
│       ├── models/
│       │   └── efficientnetv2s_best.pt
│       └── requirements-inference.txt
├── .gitattributes
└── .gitignore
```

## Models

| Module | Model | Task | Version |
|---|---|---|---|
| Module 0 | EfficientNet-B0 | MRI input validation | `module0_v1` |
| Module 1 | EfficientNetV2-S | Tumor classification | `module1_v1` |

Module 1 supports:

```text
0 → Glioma
1 → Meningioma
2 → No Tumor
3 → Pituitary
```

The Module 1 checkpoint is stored using Git LFS.

## Module 0 Decision Threshold

Current threshold:

```text
0.995301365852356
```

Decision:

```text
brain_mri_probability >= threshold
        |
        +---- YES ---> ACCEPT ---> Module 1
        |
        +---- NO ----> REJECT
```

The threshold was selected during Module 0 validation and is stored in the inference configuration.

## Module 0 Evaluation

Evaluation set:

```text
Non-Brain MRI: 3489
Brain MRI:     1200
Total:         4689
```

| Metric | Result |
|---|---:|
| Accuracy | 100% |
| Precision | 100% |
| Sensitivity / Recall | 100% |
| F1-score | 100% |
| Specificity | 100% |
| ROC-AUC | 1.0000 |

Confusion matrix:

```text
                         Predicted
                     Non-MRI    Brain MRI

Actual Non-MRI          3489        0
Actual Brain MRI           0      1200
```

These results describe performance on the available evaluation dataset and are not a guarantee for every real-world image.

## Module 1 Evaluation

Official test set:

```text
Total images: 1600
```

| Metric | Result |
|---|---:|
| Accuracy | 94.06% |
| Macro Precision | 94.47% |
| Macro Recall / Sensitivity | 94.06% |
| Macro-F1 | 93.91% |
| Multiclass ROC-AUC | 0.9896 |
| Tumor Sensitivity | 96.00% |
| Tumor Specificity | 100.00% |
| Cohen's Kappa | 0.9208 |
| ECE | 0.0313 |

Class-level results:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Glioma | 98.45% | 79.50% | 87.97% |
| Meningioma | 91.12% | 97.50% | 94.20% |
| No Tumor | 89.29% | 100.00% | 94.34% |
| Pituitary | 99.00% | 99.25% | 99.13% |

## Installation

Use the inference requirements, not the training requirements.

### Module 0

```bash
cd deployment/module0
pip install -r requirements-inference.txt
```

### Module 1

```bash
cd deployment/module1
pip install -r requirements-inference.txt
```

Recommended environment:

```text
Python 3.11+
PyTorch
Torchvision
Pillow
NumPy
PyYAML
```

## Module 0 Standalone Inference

From `deployment/module0`:

```bash
PYTHONPATH=. python -m inference.predictor \
    --image /absolute/path/to/image.jpg
```

Example accepted response:

```json
{
  "status": "accepted",
  "module": "module0",
  "model": {
    "name": "EfficientNet-B0",
    "version": "module0_v1"
  },
  "validation": {
    "class": "Brain MRI",
    "class_id": 1,
    "confidence": 0.99995,
    "brain_mri_probability": 0.99995,
    "threshold": 0.99530
  }
}
```

Example rejected response:

```json
{
  "status": "rejected",
  "module": "module0",
  "model": {
    "name": "EfficientNet-B0",
    "version": "module0_v1"
  },
  "validation": {
    "class": "Non-Brain MRI",
    "class_id": 0,
    "confidence": 0.99631,
    "brain_mri_probability": 0.00369,
    "threshold": 0.99530
  },
  "reason": "Input image was classified as non-brain-MRI"
}
```

## Module 1 Standalone Inference

From `deployment/module1`:

```bash
PYTHONPATH=. python -m inference.predictor \
    --image /absolute/path/to/image.jpg
```

Example:

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
    "confidence": 0.97774
  },
  "probabilities": {
    "Glioma": 0.01074,
    "Meningioma": 0.97774,
    "No Tumor": 0.00190,
    "Pituitary": 0.00961
  },
  "uncertainty": {
    "top1_probability": 0.97774,
    "top2_probability": 0.01074,
    "prediction_margin": 0.96700
  }
}
```

## Complete Pipeline

Preferred integration:

```text
Image
  |
  v
Module 0
  |
  +---- Rejected ----> Return rejection JSON
  |
  +---- Accepted ----> Module 1
                          |
                          v
                    Tumor prediction
```

From `deployment/module0`:

```bash
PYTHONPATH=. python -m inference.pipeline \
    --image /absolute/path/to/image.jpg
```

### Accepted MRI

```json
{
  "status": "success",
  "stage": "module1",
  "module0": {
    "status": "accepted",
    "class": "Brain MRI",
    "brain_mri_probability": 0.99995,
    "threshold": 0.99530
  },
  "module1": {
    "status": "success",
    "model": {
      "name": "EfficientNetV2-S",
      "version": "module1_v1"
    },
    "prediction": {
      "class": "Meningioma",
      "class_id": 1,
      "confidence": 0.97774
    },
    "probabilities": {
      "Glioma": 0.01074,
      "Meningioma": 0.97774,
      "No Tumor": 0.00190,
      "Pituitary": 0.00961
    },
    "uncertainty": {
      "top1_probability": 0.97774,
      "top2_probability": 0.01074,
      "prediction_margin": 0.96700
    }
  }
}
```

### Rejected Non-MRI

```json
{
  "status": "rejected",
  "stage": "module0",
  "reason": "Input is not classified as a brain MRI",
  "module0": {
    "status": "rejected",
    "module": "module0",
    "model": {
      "name": "EfficientNet-B0",
      "version": "module0_v1"
    },
    "validation": {
      "class": "Non-Brain MRI",
      "class_id": 0,
      "confidence": 0.99631,
      "brain_mri_probability": 0.00369,
      "threshold": 0.99530
    },
    "reason": "Input image was classified as non-brain-MRI"
  }
}
```

When Module 0 rejects an image, Module 1 is not executed.

## Flask Backend Integration

Recommended architecture:

```text
Frontend
   |
   | POST /predict
   | multipart/form-data
   | image=<file>
   |
   v
Flask Backend
   |
   | Save temporary image
   |
   v
NeuroScan Pipeline
   |
   +----------------------+
   |                      |
   v                      v
Module 0                Reject
   |
   | Accepted
   v
Module 1
   |
   v
JSON Result
   |
   v
Flask Response
   |
   v
Frontend
```

### Backend responsibilities

- Receive the uploaded image.
- Validate file type and upload size.
- Save the image temporarily.
- Pass the image path to the inference pipeline.
- Return the pipeline JSON response.
- Remove the temporary image after inference.
- Handle API authentication, logging, database integration, and frontend integration.

### ML package responsibilities

- Image preprocessing.
- Module 0 inference.
- Module 0 thresholding.
- Module 1 inference.
- Class mapping.
- Probability calculation.
- Prediction output.

The backend should not duplicate the ML preprocessing or Module 0 → Module 1 decision logic.

## Suggested API Contract

```http
POST /predict
Content-Type: multipart/form-data
```

Form field:

```text
image
```

Example:

```text
POST /predict
image=<MRI_IMAGE.jpg>
```

For an accepted MRI, return the complete pipeline JSON.

For a rejected image, return the Module 0 rejection JSON.

## Error Handling

### Invalid upload

Examples:

- Missing image
- Unsupported file type
- Corrupted image
- File too large

These should be handled by Flask before model inference.

### Model rejection

The image is valid but Module 0 determines that it is not a brain MRI.

The API should return a controlled rejection response rather than treating it as a server failure.

### Model/server error

Examples:

- Missing checkpoint
- Incorrect model configuration
- Runtime exception
- CUDA/PyTorch failure

These should be treated as inference/server errors.

## Supported Input

The current inference interface accepts image files such as:

```text
.jpg
.jpeg
.png
```

The models were developed and evaluated using image data corresponding to the project datasets.

Module 0 is an input-domain gate. It is not a formal DICOM validator or a guarantee that an image is clinically acquired MRI data.

## Important Deployment Notes

### Do not modify model preprocessing

Module 0 and Module 1 contain their own preprocessing logic.

The Flask backend should not resize, normalize, crop, or otherwise preprocess the image before passing it to the inference pipeline unless explicitly required by the inference code.

### Do not recreate class mappings

Module 1 mapping:

```text
0 → Glioma
1 → Meningioma
2 → No Tumor
3 → Pituitary
```

### Do not bypass Module 0

The intended public flow is:

```text
Input → Module 0 → Module 1
```

Do not expose Module 1 as the primary public prediction endpoint.

## Confidence and Uncertainty

### Module 0

Returns:

```text
brain_mri_probability
```

This is the predicted probability for the Brain MRI class.

### Module 1

Returns:

```text
confidence
top1_probability
top2_probability
prediction_margin
```

The margin is:

```text
top1_probability - top2_probability
```

These values are model probabilities and uncertainty indicators. They should not be interpreted as clinical certainty.

## Git LFS

The Module 1 checkpoint is larger than GitHub's normal file-size limit and is stored using Git LFS:

```text
deployment/module1/models/efficientnetv2s_best.pt
```

Verify Git LFS:

```bash
git lfs version
```

After cloning:

```bash
git lfs pull
```

Verify that the model file is the actual checkpoint and not an LFS pointer file.

## Verification After Cloning

```bash
git clone <repository-url>
cd EffiNetDep
git lfs pull
```

Then:

```bash
ls -lh deployment/module0/models/
ls -lh deployment/module1/models/
```

## Deployment Validation Checklist

```text
[ ] Module 0 model exists
[ ] Module 1 model exists
[ ] Git LFS model downloaded
[ ] inference.yaml files exist
[ ] Inference dependencies installed
[ ] Module 0 import succeeds
[ ] Module 1 import succeeds
[ ] MRI image is accepted
[ ] Accepted MRI reaches Module 1
[ ] Module 1 returns tumor classification
[ ] Non-MRI image is rejected
[ ] Rejected image does not reach Module 1
[ ] JSON response is returned correctly
```

## Human-in-the-Loop Future Architecture

The deployment is designed to support future model improvement through human review and subsequent retraining.

```text
User Image
    |
    v
Module 0
    |
    v
Module 1
    |
    v
Prediction JSON
    |
    v
Human Review
    |
    v
Curated Data Store
    |
    v
Model Retraining
    |
    v
New Model Version
```

Future model versions should be versioned rather than silently replacing an existing deployment.

## Model Limitations

This system is a machine-learning research prototype intended for the NeuroScan AI project and hackathon demonstration.

It is not a standalone medical diagnostic system.

Predictions should not be treated as a medical diagnosis or as a replacement for assessment by a qualified medical professional.

Reported metrics are based on the datasets and evaluation procedures used during model development.

Performance may vary on images from different scanners, hospitals, acquisition protocols, populations, or image distributions.

Module 0 reduces accidental processing of arbitrary images but is not a formal medical image authenticity detector.

## Current Deployment Versions

| Module | Model | Version |
|---|---|---|
| Module 0 | EfficientNet-B0 | `module0_v1` |
| Module 1 | EfficientNetV2-S | `module1_v1` |

## Quick Start

```bash
cd deployment/module0
pip install -r requirements-inference.txt

PYTHONPATH=. python -m inference.pipeline \
    --image /absolute/path/to/image.jpg
```

Expected flow for an MRI:

```text
MRI image
    ↓
Module 0 ACCEPT
    ↓
Module 1 CLASSIFY
    ↓
JSON RESULT
```

Expected flow for a non-MRI image:

```text
Non-MRI image
    ↓
Module 0 REJECT
    ↓
Pipeline stops
    ↓
JSON REJECTION
```

## Integration Summary

The backend should integrate the complete pipeline:

```text
Client
  |
  | POST /predict
  | image=<file>
  v
Flask
  |
  v
NeuroScan Pipeline
  |
  +--> Module 0: Brain MRI validation
  |
  +--> If accepted
          |
          v
      Module 1: Tumor classification
          |
          v
      JSON response
```

Public behavior:

```text
Brain MRI
    → Accept
    → Classify
    → Return tumor class + confidence + probabilities

Non-Brain MRI
    → Reject
    → Do not run tumor classifier
    → Return rejection reason
```

---

**NeuroScan AI — Deployment Package**
