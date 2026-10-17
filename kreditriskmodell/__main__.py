"""Train on synthetic data and score a few example companies.

Run:
    python -m kreditriskmodell
"""

from __future__ import annotations

import sys

from .data import generate_synthetic_portfolio
from .model import CreditRiskModel
from .serve import ARTIFACT_PATH, EXAMPLE_COMPANIES


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _configure_stdout()

    df = generate_synthetic_portfolio()
    print(f"Default rate: {df['default_12m'].mean():.1%}")
    print(df.head())
    print()

    model = CreditRiskModel().fit(df, store_metrics=True)
    model.save(ARTIFACT_PATH)
    print(f"Saved model to {ARTIFACT_PATH}")
    for cv in model.cv_results:
        print(f"{cv.name}: CV-AUC {cv.mean_auc:.3f} +/- {cv.std_auc:.3f}")

    test = model.test_result
    if test is None:
        test = model.evaluate_test()
    print(f"\nSelected model: {test.model_name}")
    print(f"Test-AUC:    {test.auc:.3f}")
    print(
        f"Test-PR-AUC: {test.pr_auc:.3f} "
        f"(random baseline ~ {test.baseline_pr_auc:.3f})"
    )
    print(f"\nClassification report at threshold {test.threshold}:")
    print(test.report)
    print(test.feature_importance.round(3).to_string())
    print()

    for name, kwargs in EXAMPLE_COMPANIES.items():
        model.predict_company(company_name=name, verbose=True, **kwargs)


if __name__ == "__main__":
    main()
