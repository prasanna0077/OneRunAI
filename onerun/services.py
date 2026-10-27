"""Service layer - the single place where Excel data, ML engines and the Deal Context meet.

Pure Python (no Streamlit import) so it can be unit-tested and reused by Phase-2/3 modules.
The chain is:  Acquisition -> (chosen target) -> Integration -> (exposure flags) -> Retention.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from onerun.config import ACQ_FILE, ACQUIRER, DATA_FILES, INT_FILE, RET_FILE
from onerun.data_generation.acquirer import build_financials
from onerun.data_generation.generator import ensure_data
from onerun.ml import acquisition as acq_ml
from onerun.ml import integration as int_ml
from onerun.ml import proforma as pf_ml
from onerun.ml import retention as ret_ml


def data_version() -> float:
    """Changes whenever a workbook is regenerated -> invalidates the caches below."""
    return max((f.stat().st_mtime for f in DATA_FILES if f.exists()), default=0.0)


def clear_caches() -> None:
    for fn in (_acq_raw, _int_raw, _ret_raw, _acq_results, _int_results, _ret_results):
        fn.cache_clear()


# ----------------------------------------------------------------------------- raw data
@lru_cache(maxsize=4)
def _acq_raw(_v: float) -> dict:
    ensure_data()
    s = pd.read_excel(ACQ_FILE, sheet_name=None)
    cand = (s["Bank_Financials"].merge(s["Credit_Metrics"], on="bank_id").merge(s["Business_Metrics"], on="bank_id")
            .merge(s["Digital_Metrics"], on="bank_id"))
    return {"candidates": cand, "history": s["Historical_Deals"], "acquirer": s["Acquirer_Profile"],
            "acquirer_financials": s.get("Acquirer_Financials")}


@lru_cache(maxsize=4)
def _int_raw(_v: float) -> dict:
    ensure_data()
    return pd.read_excel(INT_FILE, sheet_name=None)


@lru_cache(maxsize=4)
def _ret_raw(_v: float) -> dict:
    ensure_data()
    return pd.read_excel(RET_FILE, sheet_name=None)


def acquisition_data() -> dict: return _acq_raw(data_version())
def integration_data() -> dict: return _int_raw(data_version())
def retention_data() -> dict: return _ret_raw(data_version())


# ----------------------------------------------------------------------------- client bank
def acquirer() -> dict:
    """Our Bank (Meridian) as one dict: ACQUIRER ratios + rupee amounts.

    Amounts come from the Excel sheet 'Acquirer_Financials' (edit it to change the numbers); if the workbook was
    generated before that sheet existed, they fall back to the values derived from config.py.
    """
    out = {**ACQUIRER, **build_financials()}
    fin = acquisition_data().get("acquirer_financials")
    if fin is not None and {"metric", "value"} <= set(fin.columns):
        for k, v in zip(fin["metric"], fin["value"]):
            if k in out:
                out[k] = v.item() if hasattr(v, "item") else v
    return out


# ----------------------------------------------------------------------------- MVP 1
@lru_cache(maxsize=4)
def _acq_results(_v: float):
    raw = _acq_raw(_v)
    suite = acq_ml.train(raw["history"])
    return suite, acq_ml.score_candidates(suite, raw["candidates"])


def acquisition_results():
    """-> (TrainedSuite, scored candidates DataFrame sorted by score)."""
    return _acq_results(data_version())


def top_target_id() -> str:
    return str(acquisition_results()[1].iloc[0]["bank_id"])


def target_row(bank_id: str) -> pd.Series:
    scored = acquisition_results()[1]
    return scored[scored.bank_id == bank_id].iloc[0]


# ----------------------------------------------------------------------------- MVP 2
@lru_cache(maxsize=64)
def _int_results(_v: float, bank_id: str, kc: int, kp: int, kb: int) -> int_ml.IntegrationResult:
    raw = _int_raw(_v)
    return int_ml.run_integration(bank_id, raw["Customers"], raw["Products"], raw["Acquirer_Products"], raw["Branches"],
                                  target_row(bank_id), kc, kp, kb)


def integration_results(bank_id: str, k_customers: int = 5, k_products: int = 4, k_branches: int = 3):
    return _int_results(data_version(), bank_id, int(k_customers), int(k_products), int(k_branches))


# ----------------------------------------------------------------------------- MVP 3
@lru_cache(maxsize=4)
def _ret_results(_v: float):
    return ret_ml.train(_ret_raw(_v)["Historical_Merger_Customers"])


def retention_suite():
    return _ret_results(data_version())


def retention_scored(bank_id: str, model: str | None = None, medium: float = .15, high: float = .35) -> pd.DataFrame:
    """Score the target's customers; exposure flags come from the Integration plan (MVP-to-MVP hand-off)."""
    integ = integration_results(bank_id)
    return ret_ml.score_customers(retention_suite(), integ.customers, model, medium, high)


# ----------------------------------------------------------------------------- deal context (for Executive / Phase 2-3)
def deal_context(bank_id: str) -> dict:
    """One dict that summarises the deal across all live MVPs. Phase-2/3 modules build on this."""
    suite, scored = acquisition_results()
    t = target_row(bank_id)
    integ = integration_results(bank_id)
    ret = retention_scored(bank_id)
    return {
        "acquirer": ACQUIRER["name"], "target": t, "acquisition_suite": suite, "scored": scored,
        "integration": integ, "retention": ret, "combined": pf_ml.combine(acquirer(), t),
        "kpis": {
            "acquisition_score": float(t.acquisition_score), "tier": t.tier, "rank": int(t["rank"]),
            "risk_adj_synergy_cr": float(t.risk_adj_synergy_cr), "valuation_cr": float(t.valuation_cr),
            "complexity_score": integ.complexity["score"], "complexity_band": integ.complexity["band"],
            "branch_saving_cr": float(integ.branches.annual_saving_lakh.sum() / 100 * integ.scale_branches),
            "expected_churn_pct": float(ret.churn_probability.mean() * 100),
            "high_risk_customers": int((ret.risk_band == "High").sum()),
            "revenue_at_risk_cr": float(ret.revenue_at_risk_inr.sum() / 1e7 * integ.scale_customers),
        },
    }
