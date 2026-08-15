"""Deployment inference package for NeuroScan AI Module 0."""

__all__ = ["MRIGatePredictor", "NeuroScanPipeline"]


def __getattr__(name: str):
    if name == "MRIGatePredictor":
        from .predictor import MRIGatePredictor

        return MRIGatePredictor
    if name == "NeuroScanPipeline":
        from .pipeline import NeuroScanPipeline

        return NeuroScanPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
