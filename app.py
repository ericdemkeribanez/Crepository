"""Streamlit UI for the fitted credit-risk model."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from kreditriskmodell import load_or_train
from kreditriskmodell.serve import ARTIFACT_PATH, EXAMPLE_COMPANIES

st.set_page_config(
    page_title="Kreditriskmodell",
    page_icon="🏦",
    layout="wide",
)

FEATURE_LABELS = {
    "sector": "Sector",
    "revenue_sek": "Revenue",
    "age_years": "Company age",
    "debt_to_equity": "Debt-to-equity",
    "current_ratio": "Current ratio",
    "profit_margin": "Profit margin",
    "payment_remark": "Payment remark",
}

BAND_COLORS = {
    "approve": "#1B7A4E",
    "review": "#B86E00",
    "reject": "#B42318",
}


@st.cache_resource(show_spinner="Loading credit-risk model…")
def get_model():
    return load_or_train()


def _apply_example(name: str) -> None:
    payload = EXAMPLE_COMPANIES[name]
    st.session_state.company_name = name
    st.session_state.sector = payload["sector"]
    st.session_state.revenue_msek = payload["revenue_sek"] / 1_000_000
    st.session_state.age_years = float(payload["age_years"])
    st.session_state.debt_to_equity = float(payload["debt_to_equity"])
    missing = payload["current_ratio"] is None
    st.session_state.missing_current_ratio = missing
    st.session_state.current_ratio = 1.3 if missing else float(payload["current_ratio"])
    st.session_state.profit_margin_pct = float(payload["profit_margin"]) * 100
    st.session_state.payment_remark = bool(payload["payment_remark"])
    st.session_state.pending_score = True


def _init_form_state() -> None:
    if "form_ready" in st.session_state:
        return
    _apply_example("Steady Growth AB")
    st.session_state.form_ready = True


def _render_sidebar(model) -> None:
    st.sidebar.header("Model")
    st.sidebar.write(f"**{model.model_name}**")
    st.sidebar.caption(f"Artifact: `{ARTIFACT_PATH.name}`")
    st.sidebar.metric("Approve below", f"{model.approve_below:.0%}")
    st.sidebar.metric("Reject from", f"{model.reject_from:.0%}")
    if model.train_default_rate is not None:
        st.sidebar.metric("Train default rate", f"{model.train_default_rate:.1%}")

    test = model.test_result
    if test is not None:
        st.sidebar.divider()
        st.sidebar.subheader("Hold-out performance")
        st.sidebar.metric("Test AUC", f"{test.auc:.3f}")
        st.sidebar.metric("Test PR-AUC", f"{test.pr_auc:.3f}")
        st.sidebar.caption(f"Random baseline ~ {test.baseline_pr_auc:.3f}")
        with st.sidebar.expander("Feature importance (permutation AUC)"):
            st.dataframe(
                test.feature_importance.round(3).rename("importance"),
                use_container_width=True,
            )
        with st.sidebar.expander("Classification report"):
            st.text(test.report)

    st.sidebar.divider()
    if st.sidebar.button("Retrain and overwrite artifact"):
        with st.spinner("Retraining model…"):
            load_or_train(retrain=True)
        st.cache_resource.clear()
        st.rerun()


def _score(model):
    current_ratio = None if st.session_state.missing_current_ratio else float(
        st.session_state.current_ratio
    )
    return model.predict_company(
        company_name=st.session_state.company_name.strip() or "Unnamed company",
        sector=st.session_state.sector,
        revenue_sek=float(st.session_state.revenue_msek) * 1_000_000,
        age_years=float(st.session_state.age_years),
        debt_to_equity=float(st.session_state.debt_to_equity),
        current_ratio=current_ratio,
        profit_margin=float(st.session_state.profit_margin_pct) / 100,
        payment_remark=int(st.session_state.payment_remark),
    )


def _render_result(model, result) -> None:
    color = BAND_COLORS[result.band]
    portfolio = model.train_default_rate or 0.0
    st.markdown(
        f"<div style='padding:1rem 1.2rem;border-radius:12px;background:{color}14;"
        f"border:1px solid {color}55'>"
        f"<div style='color:{color};font-weight:700;font-size:1.15rem'>{result.decision}</div>"
        f"<div style='margin-top:0.35rem;color:#333'>{result.company}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("12-month PD", f"{result.p_default:.1%}")
    c2.metric("Portfolio average PD", f"{portfolio:.1%}")
    if portfolio:
        c3.metric("Multiple of average", f"{result.p_default / portfolio:.2f}x")
    else:
        c3.metric("Multiple of average", "—")

    gauge = min(max(result.p_default / 0.5, 0.0), 1.0)
    st.progress(gauge, text=f"Default probability {result.p_default:.1%} (bar scaled to 50%)")

    if result.drivers:
        st.subheader("Largest risk drivers")
        st.caption("How much PD would fall if this field were set to a typical training value.")
        rows = [
            {
                "Feature": FEATURE_LABELS.get(col, col),
                "PD reduction (pp)": round(delta * 100, 1),
            }
            for col, delta in result.drivers
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No material risk drivers versus typical values.")


def main() -> None:
    _init_form_state()
    model = get_model()
    _render_sidebar(model)

    st.title("Credit assessment")
    st.write(
        "Score a company with the fitted pipeline. "
        "Missing current ratio is imputed the same way as in training."
    )

    def _on_example_change() -> None:
        name = st.session_state.example_choice
        if name in EXAMPLE_COMPANIES:
            _apply_example(name)

    st.selectbox(
        "Load an example",
        options=["(keep current inputs)", *EXAMPLE_COMPANIES.keys()],
        key="example_choice",
        on_change=_on_example_change,
    )

    left, right = st.columns((1.15, 1))
    with left:
        st.text_input("Company name", key="company_name")
        st.selectbox("Sector", options=model.sectors, key="sector")
        r1, r2 = st.columns(2)
        r1.number_input("Revenue (MSEK)", min_value=0.1, step=1.0, key="revenue_msek")
        r2.number_input("Age (years)", min_value=0.1, step=0.5, key="age_years")
        r3, r4 = st.columns(2)
        r3.number_input("Debt-to-equity", min_value=0.0, step=0.1, key="debt_to_equity")
        r4.number_input("Profit margin (%)", min_value=-100.0, max_value=100.0, step=0.5, key="profit_margin_pct")
        st.checkbox("Current ratio is missing", key="missing_current_ratio")
        st.number_input(
            "Current ratio",
            min_value=0.01,
            step=0.1,
            key="current_ratio",
            disabled=st.session_state.missing_current_ratio,
        )
        st.checkbox("Payment remark (yes = 1)", key="payment_remark")
        assess = st.button("Assess credit risk", type="primary")

    with right:
        if assess:
            try:
                result = _score(model)
            except ValueError as exc:
                st.error(str(exc))
            else:
                _render_result(model, result)
        else:
            st.caption("Fill in the company and click **Assess credit risk**.")


if __name__ == "__main__":
    main()
