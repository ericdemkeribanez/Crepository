"""Train, evaluate, and score the credit-risk pipeline."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FEATURE_COLS, ID_COL, TARGET_COL


APPROVE_BELOW = 0.10
REJECT_FROM = 0.25


@dataclass
class CvResult:
    name: str
    mean_auc: float
    std_auc: float


@dataclass
class TestResult:
    model_name: str
    auc: float
    pr_auc: float
    baseline_pr_auc: float
    threshold: float
    report: str
    feature_importance: pd.Series


@dataclass
class CreditDecision:
    company: str
    p_default: float
    decision: str
    band: str
    drivers: list[tuple[str, float]] = field(default_factory=list)


class CreditRiskModel:
    """Stratified train/test split, CV model selection, and company-level scoring."""

    def __init__(
        self,
        approve_below: float = APPROVE_BELOW,
        reject_from: float = REJECT_FROM,
        random_state: int = 42,
    ) -> None:
        if not 0 < approve_below < reject_from < 1:
            raise ValueError("Need 0 < approve_below < reject_from < 1.")
        self.approve_below = approve_below
        self.reject_from = reject_from
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.model_name: str | None = None
        self.features: list[str] = list(FEATURE_COLS)
        self.sectors: list[str] = []
        self.cv_results: list[CvResult] = []
        self.test_result: TestResult | None = None
        self._typical: pd.Series | None = None
        self.train_default_rate: float | None = None
        self._X_test: pd.DataFrame | None = None
        self._y_test: pd.Series | None = None

    def fit(
        self,
        df: pd.DataFrame,
        test_size: float = 0.25,
        store_metrics: bool = True,
    ) -> CreditRiskModel:
        X = df.drop(columns=[ID_COL, TARGET_COL])
        y = df[TARGET_COL]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        self.features = X_train.columns.tolist()
        self.sectors = sorted(X_train["sector"].astype(str).unique())
        self._typical = _typical_values(X_train)
        self.train_default_rate = float(y_train.mean())
        self._X_test = X_test
        self._y_test = y_test

        preprocess = _build_preprocess(X)
        candidates = {
            "Logistisk regression": LogisticRegression(max_iter=1000),
            "Random forest": RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=20,
                random_state=self.random_state,
                n_jobs=-1,
            ),
        }
        pipes = {
            name: Pipeline([("prep", preprocess), ("clf", clf)])
            for name, clf in candidates.items()
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        self.cv_results = []
        cv_auc: dict[str, float] = {}
        for name, pipe in pipes.items():
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
            self.cv_results.append(CvResult(name, float(scores.mean()), float(scores.std())))
            cv_auc[name] = float(scores.mean())

        self.model_name = max(cv_auc, key=cv_auc.get)
        self.pipeline = pipes[self.model_name].fit(X_train, y_train)
        if store_metrics:
            self.test_result = self.evaluate_test()
        return self

    def evaluate_test(self, n_repeats: int = 10) -> TestResult:
        pipeline, X_test, y_test = self._require_fitted_test()
        p_test = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (p_test >= self.approve_below).astype(int)
        importance = permutation_importance(
            pipeline,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=self.random_state,
        )
        return TestResult(
            model_name=self.model_name or "",
            auc=float(roc_auc_score(y_test, p_test)),
            pr_auc=float(average_precision_score(y_test, p_test)),
            baseline_pr_auc=float(y_test.mean()),
            threshold=self.approve_below,
            report=classification_report(y_test, y_pred, zero_division=0),
            feature_importance=pd.Series(
                importance.importances_mean, index=X_test.columns
            ).sort_values(ascending=False),
        )

    def predict_company(
        self,
        company_name: str,
        sector: str,
        revenue_sek: float,
        age_years: float,
        debt_to_equity: float,
        current_ratio: float | None,
        profit_margin: float,
        payment_remark: int,
        verbose: bool = False,
    ) -> CreditDecision:
        pipeline = self._require_pipeline()
        self._validate(
            sector, revenue_sek, age_years, debt_to_equity, profit_margin, payment_remark
        )
        row = pd.DataFrame(
            [
                {
                    "sector": sector,
                    "revenue_sek": revenue_sek,
                    "age_years": age_years,
                    "debt_to_equity": debt_to_equity,
                    "current_ratio": np.nan if current_ratio is None else current_ratio,
                    "profit_margin": profit_margin,
                    "payment_remark": payment_remark,
                }
            ]
        )[self.features]

        p_default = float(pipeline.predict_proba(row)[0, 1])
        band = self._decision_band(p_default)
        decision = self._decision_label(band)
        drivers = self._top_risk_drivers(row)

        result = CreditDecision(company_name, p_default, decision, band, drivers)
        if verbose:
            print(_format_decision(result, self.train_default_rate or 0.0))
        return result

    def save(self, path: str | Path, include_holdout: bool = False) -> Path:
        """Serialize the fitted pipeline and scoring metadata for the Streamlit app."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self if include_holdout else self._without_holdout()
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> CreditRiskModel:
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__} artifact, got {type(model)!r}")
        if model.pipeline is None:
            raise RuntimeError("Loaded artifact is not fitted.")
        return model

    def _without_holdout(self) -> CreditRiskModel:
        clone = copy.copy(self)
        clone._X_test = None
        clone._y_test = None
        return clone

    def _decision_band(self, p_default: float) -> str:
        if p_default < self.approve_below:
            return "approve"
        if p_default < self.reject_from:
            return "review"
        return "reject"

    def _decision_label(self, band: str) -> str:
        labels = {
            "approve": "GODKÄND (låg risk)",
            "review": "MANUELL GRANSKNING (förhöjd risk)",
            "reject": "AVSLAG (hög risk)",
        }
        return labels[band]

    def _top_risk_drivers(self, row: pd.DataFrame, n: int = 3) -> list[tuple[str, float]]:
        pipeline = self._require_pipeline()
        typical = self._typical
        if typical is None:
            return []
        base = float(pipeline.predict_proba(row)[0, 1])
        effects: dict[str, float] = {}
        for col in row.columns:
            alt = row.copy()
            alt[col] = typical[col]
            effects[col] = base - float(pipeline.predict_proba(alt)[0, 1])
        ranked = sorted(effects.items(), key=lambda kv: kv[1], reverse=True)
        return [(col, delta) for col, delta in ranked[:n] if delta > 0.002]

    def _validate(
        self,
        sector: str,
        revenue_sek: float,
        age_years: float,
        debt_to_equity: float,
        profit_margin: float,
        payment_remark: int,
    ) -> None:
        if sector not in self.sectors:
            raise ValueError(f"Unknown sector '{sector}'. Choose among: {self.sectors}")
        if revenue_sek <= 0 or age_years <= 0 or debt_to_equity < 0:
            raise ValueError("Revenue and age must be > 0; debt-to-equity must be >= 0.")
        if not -1 <= profit_margin <= 1:
            raise ValueError("Profit margin is a decimal, e.g. 0.06 for 6%.")
        if payment_remark not in (0, 1):
            raise ValueError("payment_remark must be 0 or 1.")

    def _require_pipeline(self) -> Pipeline:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self.pipeline

    def _require_fitted_test(self) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
        pipeline = self._require_pipeline()
        if self._X_test is None or self._y_test is None:
            raise RuntimeError("No hold-out set stored. Call fit() first.")
        return pipeline, self._X_test, self._y_test


def _build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes("number").columns.tolist()
    return ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["sector"]),
        ]
    )


def _typical_values(X_train: pd.DataFrame) -> pd.Series:
    values: dict[str, object] = {}
    for col in X_train.columns:
        if col == "sector":
            values[col] = X_train[col].mode()[0]
        else:
            values[col] = X_train[col].median()
    return pd.Series(values)


def _format_decision(result: CreditDecision, portfolio_rate: float) -> str:
    lines = [
        f"--- CREDIT ASSESSMENT: {result.company} ---",
        (
            f"12-month default probability: {result.p_default:.1%}  "
            f"(portfolio average ~ {portfolio_rate:.1%})"
        ),
        f"Decision: {result.decision}",
    ]
    if result.drivers:
        lines.append("Largest risk drivers:")
        for col, delta in result.drivers:
            lines.append(
                f"  - {col}: risk falls by {delta * 100:.1f} pp if the value were typical"
            )
    lines.append("-" * 50)
    return "\n".join(lines)
