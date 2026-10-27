"""MVP 1 - Acquisition Intelligence.

Business question: *Which institution should we acquire?*
Engine: Random Forest + Gradient Boosting + Decision Tree trained on historical deals
(label = synergy realised), then used to score every live candidate (0-100).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from onerun.config import ACQUIRER, ASSUMPTIONS
from onerun.data_generation.banks import BANK_TYPE_LIST
from onerun.ml.common import TrainedSuite, fit_suite

NUMERIC_FEATURES = [
    "roa_pct", "roe_pct", "casa_pct", "gnpa_pct", "nnpa_pct", "pcr_pct", "car_pct", "nim_pct",
    "cost_to_income_pct", "log_branches", "customers_lakh", "app_mau_pct", "digital_adoption_pct",
    "digital_txn_share_pct", "tech_modernity_score", "size_ratio", "geo_overlap_pct",
]
FEATURE_LABELS = {
    "roa_pct": "ROA", "roe_pct": "ROE", "casa_pct": "CASA ratio", "gnpa_pct": "Gross NPA", "nnpa_pct": "Net NPA",
    "pcr_pct": "Provision coverage", "car_pct": "Capital adequacy", "nim_pct": "Net interest margin",
    "cost_to_income_pct": "Cost-to-income", "log_branches": "Branch network", "customers_lakh": "Customer base",
    "app_mau_pct": "App usage (MAU)", "digital_adoption_pct": "Digital adoption",
    "digital_txn_share_pct": "Digital txn share", "tech_modernity_score": "Tech modernity",
    "size_ratio": "Size vs acquirer", "geo_overlap_pct": "Geographic overlap",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X["log_branches"] = np.log1p(X["branches"])
    X = X[NUMERIC_FEATURES].copy()
    for t in BANK_TYPE_LIST:                          # fixed one-hot so train/score columns always match
        X[f"type: {t}"] = (df["bank_type"] == t).astype(int)
    return X


def train(history: pd.DataFrame, seed: int = 42) -> TrainedSuite:
    X, y = build_features(history), history["synergy_realized"]
    estimators = {
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=seed, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=.05, max_depth=3, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=15, random_state=seed),
    }
    return fit_suite(X, y, estimators, seed=seed)


def red_flags(r: pd.Series) -> list[str]:
    flags = []
    if r.gnpa_pct > 8: flags.append(f"High gross NPA ({r.gnpa_pct:.1f}%)")
    if r.pcr_pct < 55: flags.append(f"Thin provision coverage ({r.pcr_pct:.0f}%)")
    if r.car_pct < 12: flags.append(f"Low capital adequacy ({r.car_pct:.1f}%)")
    if r.roa_pct < 0.4: flags.append(f"Weak profitability (ROA {r.roa_pct:.2f}%)")
    if r.cost_to_income_pct > 70: flags.append(f"High cost-to-income ({r.cost_to_income_pct:.0f}%)")
    if r.tech_modernity_score < 2.3: flags.append("Legacy technology stack")
    if r.size_ratio > .5: flags.append("Very large relative to acquirer")
    if r.geo_overlap_pct > 55: flags.append("Heavy branch overlap - attrition risk in consolidation")
    return flags


def tier(score: float) -> str:
    a = ASSUMPTIONS
    return ("Prime Target" if score >= a["tier_prime"] else "Strong Candidate" if score >= a["tier_strong"]
            else "Watchlist" if score >= a["tier_watch"] else "Avoid")


def score_candidates(suite: TrainedSuite, cand: pd.DataFrame) -> pd.DataFrame:
    """Score + rank candidates. Ensemble = AUC-weighted mean of the three model probabilities."""
    X = build_features(cand)
    out = cand.copy()
    w = suite.metrics.set_index("Model")["AUC"]
    w = w / w.sum()
    ens = np.zeros(len(out))
    for name in suite.models:
        p = suite.predict_proba(name, X)
        out[f"p_{name}"] = p
        ens += w[name] * p
    out["success_probability"] = ens
    out["acquisition_score"] = (ens * 100).round(1)
    out["tier"] = out["acquisition_score"].map(tier)
    out["indicative_synergy_cr"] = (out.total_assets_cr * ASSUMPTIONS["synergy_pct_of_assets"]
                                    * (.6 + out.geo_overlap_pct / 100)).round(0)
    out["risk_adj_synergy_cr"] = (out.indicative_synergy_cr * ens).round(0)
    out["synergy_to_valuation_pct"] = (out.risk_adj_synergy_cr / out.valuation_cr * 100).round(1)
    out["red_flags"] = out.apply(lambda r: "; ".join(red_flags(r)) or "None", axis=1)
    out = out.sort_values("acquisition_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def radar_profile(row: pd.Series, scored: pd.DataFrame) -> pd.DataFrame:
    """Percentile-based radar: target vs. acquirer vs. peer median (0-100, higher = better)."""
    dims = {  # label: (column, higher_is_better)
        "Profitability (ROA)": ("roa_pct", True), "Funding (CASA)": ("casa_pct", True),
        "Asset quality": ("gnpa_pct", False), "Provisioning (PCR)": ("pcr_pct", True),
        "Digital adoption": ("digital_adoption_pct", True), "Technology": ("tech_modernity_score", True),
    }
    rows = []
    for label, (col, hib) in dims.items():
        dist = scored[col].to_numpy(dtype=float)
        pct = lambda v: float((dist < v).mean() * 100 if hib else (dist > v).mean() * 100)   # noqa: E731
        rows.append(dict(dimension=label, Target=pct(row[col]), Acquirer=pct(ACQUIRER[col]), **{"Peer median": 50.0}))
    return pd.DataFrame(rows)
