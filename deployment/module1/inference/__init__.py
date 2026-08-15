"""NeuroScan AI Module 1 deployment inference package."""

__all__ = ["TumorClassifier"]


def __getattr__(name: str):
    if name == "TumorClassifier":
        from .predictor import TumorClassifier

        return TumorClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
