"""Credit-risk model: synthetic SME data, training, and company scoring."""

from .data import generate_synthetic_portfolio
from .model import CreditDecision, CreditRiskModel
from .serve import ARTIFACT_PATH, load_or_train

__all__ = [
    "ARTIFACT_PATH",
    "CreditDecision",
    "CreditRiskModel",
    "generate_synthetic_portfolio",
    "load_or_train",
]
