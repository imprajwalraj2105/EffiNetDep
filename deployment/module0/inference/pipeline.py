from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from .predictor import MRIGatePredictor


MODULE0_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_ROOT = MODULE0_ROOT.parent
MODULE1_ROOT = DEPLOYMENT_ROOT / "module1"


class NeuroScanPipeline:
    """Orchestrates Module 0 validation before calling the frozen Module 1 predictor."""

    def __init__(
        self,
        module0_config: str | Path | None = None,
        module0_checkpoint: str | Path | None = None,
        module1_config: str | Path | None = None,
        module1_checkpoint: str | Path | None = None,
        device: str | None = None,
        module1_predictor: Any | None = None,
        module0_predictor: MRIGatePredictor | None = None,
    ) -> None:
        self.module0 = module0_predictor or MRIGatePredictor(
            checkpoint_path=module0_checkpoint,
            config_path=module0_config,
            device=device,
        )
        self.module1_config = module1_config
        self.module1_checkpoint = module1_checkpoint
        self.device = device
        self._module1_predictor = module1_predictor

    def _load_module1_predictor(self):
        if self._module1_predictor is not None:
            return self._module1_predictor
        if not MODULE1_ROOT.exists():
            raise FileNotFoundError(f"Frozen Module 1 deployment package not found: {MODULE1_ROOT}")
        deployment_root = str(DEPLOYMENT_ROOT)
        if deployment_root not in sys.path:
            sys.path.insert(0, deployment_root)
        predictor_module = importlib.import_module("module1.inference.predictor")
        predictor_cls = getattr(predictor_module, "TumorClassifier")
        self._module1_predictor = predictor_cls(
            checkpoint_path=self.module1_checkpoint,
            config_path=self.module1_config,
            device=self.device,
        )
        return self._module1_predictor

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        module0_result = self.module0.predict(image_path)
        if module0_result["status"] != "accepted":
            return {
                "status": "rejected",
                "stage": "module0",
                "reason": "Input is not classified as a brain MRI",
                "module0": module0_result,
            }
        module1 = self._load_module1_predictor()
        module1_result = module1.predict(image_path)
        return {
            "status": "success",
            "stage": "module1",
            "module0": {
                "status": module0_result["status"],
                "class": module0_result["validation"]["class"],
                "brain_mri_probability": module0_result["validation"]["brain_mri_probability"],
                "threshold": module0_result["validation"]["threshold"],
            },
            "module1": module1_result,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NeuroScan AI Module 0 -> Module 1 pipeline.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--module0-config", default=None)
    parser.add_argument("--module0-checkpoint", default=None)
    parser.add_argument("--module1-config", default=None)
    parser.add_argument("--module1-checkpoint", default=None)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    pipeline = NeuroScanPipeline(
        module0_config=args.module0_config,
        module0_checkpoint=args.module0_checkpoint,
        module1_config=args.module1_config,
        module1_checkpoint=args.module1_checkpoint,
        device=args.device,
    )
    print(json.dumps(pipeline.predict(args.image), indent=2))


if __name__ == "__main__":
    main()
