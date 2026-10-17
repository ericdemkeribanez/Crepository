"""Helpers for serving a fitted model from Streamlit (or any web app)."""

from __future__ import annotations

from pathlib import Path

from .data import generate_synthetic_portfolio
from .model import CreditRiskModel

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = PACKAGE_ROOT / "artifacts" / "credit_risk_model.joblib"

EXAMPLE_COMPANIES = {
    "Crisis Construction AB": dict(
        sector="Bygg",
        revenue_sek=8_000_000,
        age_years=1.5,
        debt_to_equity=3.5,
        current_ratio=0.6,
        profit_margin=-0.08,
        payment_remark=1,
    ),
    "Steady Growth AB": dict(
        sector="IT",
        revenue_sek=120_000_000,
        age_years=18,
        debt_to_equity=0.4,
        current_ratio=2.0,
        profit_margin=0.12,
        payment_remark=0,
    ),
    "Mellanbolaget AB": dict(
        sector="Handel",
        revenue_sek=30_000_000,
        age_years=8,
        debt_to_equity=1.2,
        current_ratio=None,
        profit_margin=0.04,
        payment_remark=0,
    ),
}


def load_or_train(
    path: str | Path | None = None,
    *,
    retrain: bool = False,
) -> CreditRiskModel:
    """Load a saved artifact, or train once and persist it for the app."""
    artifact = Path(path) if path is not None else ARTIFACT_PATH
    if artifact.exists() and not retrain:
        return CreditRiskModel.load(artifact)

    df = generate_synthetic_portfolio()
    model = CreditRiskModel().fit(df, store_metrics=True)
    model.save(artifact)
    return model
