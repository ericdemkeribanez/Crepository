"""Synthetic SME portfolio used to train the credit-risk model."""

from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = ["Bygg", "Transport", "Restaurang", "IT", "Handel", "Fastighet"]
SECTOR_PROBS = [0.20, 0.15, 0.15, 0.20, 0.20, 0.10]
SECTOR_EFFECT = {
    "Bygg": 0.3,
    "Transport": 0.2,
    "Restaurang": 0.5,
    "IT": -0.2,
    "Handel": 0.1,
    "Fastighet": -0.3,
}

ID_COL = "company_id"
TARGET_COL = "default_12m"
FEATURE_COLS = [
    "sector",
    "revenue_sek",
    "age_years",
    "debt_to_equity",
    "current_ratio",
    "profit_margin",
    "payment_remark",
]


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(score: np.ndarray, target_rate: float, n_iter: int = 50) -> float:
    lo, hi = -10.0, 10.0
    intercept = 0.0
    for _ in range(n_iter):
        intercept = (lo + hi) / 2
        if _sigmoid(score + intercept).mean() > target_rate:
            hi = intercept
        else:
            lo = intercept
    return intercept


def generate_synthetic_portfolio(
    n: int = 2000,
    seed: int = 42,
    default_rate: float = 0.06,
    missing_current_ratio: float = 0.03,
) -> pd.DataFrame:
    """Return a labeled portfolio with realistic feature distributions and ~`default_rate` defaults."""
    rng = np.random.default_rng(seed)

    sector = rng.choice(SECTORS, size=n, p=SECTOR_PROBS)
    sector_effect = pd.Series(sector).map(SECTOR_EFFECT).to_numpy()

    revenue_sek = np.clip(rng.lognormal(np.log(30e6), 0.9, n), 2e6, 500e6).round(-4)
    age_years = np.clip(rng.exponential(10, n), 0.5, 40).round(1)
    debt_to_equity = np.clip(rng.lognormal(np.log(1.0), 0.6, n), 0.05, 8).round(2)
    current_ratio = np.clip(rng.lognormal(np.log(1.3), 0.4, n), 0.2, 4).round(2)
    profit_margin = np.clip(
        rng.normal(0.06, 0.08, n) + 0.03 * (current_ratio - 1.3),
        -0.3,
        0.3,
    ).round(3)

    p_remark = _sigmoid(-2.5 + 0.5 * (debt_to_equity - 1) - 0.8 * (current_ratio - 1.3))
    payment_remark = rng.binomial(1, p_remark)

    score = (
        0.5 * _z(np.log(debt_to_equity))
        - 0.6 * _z(np.log(current_ratio))
        - 0.5 * _z(profit_margin)
        - 0.3 * _z(np.log(age_years))
        - 0.2 * _z(np.log(revenue_sek))
        + 0.9 * payment_remark
        + sector_effect
        + rng.normal(0, 0.5, n)
    )
    intercept = _calibrate_intercept(score, default_rate)
    default = rng.binomial(1, _sigmoid(score + intercept))

    df = pd.DataFrame(
        {
            ID_COL: [f"C{i:04d}" for i in range(n)],
            "sector": sector,
            "revenue_sek": revenue_sek,
            "age_years": age_years,
            "debt_to_equity": debt_to_equity,
            "current_ratio": current_ratio,
            "profit_margin": profit_margin,
            "payment_remark": payment_remark,
            TARGET_COL: default,
        }
    )
    df.loc[rng.random(n) < missing_current_ratio, "current_ratio"] = np.nan
    return df
