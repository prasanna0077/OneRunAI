"""MVP 3 - Customer Retention Intelligence.

Business question: *Which customers will leave after the merger - and what do we do about it?*
Engine: Logistic Regression + Random Forest + Gradient Boosting trained on past-merger cohorts,
then used to score the target's customers (including exposure flags coming from Integration Intelligence).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from onerun.config import ASSUMPTIONS
from onerun.ml.common import TrainedSuite, fit_suite

FEATURES = [
    "age", "tenure_years", "log_balance", "products_held", "login_frequency_30d", "branch_visits_30d",
    "monthly_txn_count", "complaints_12m", "salary_account", "has_loan", "csat_score", "digital_active",
    "credit_score", "branch_closure_flag", "product_sunset_flag", "fee_increase_flag",
]
FEATURE_LABELS = {
    "age": "Age", "tenure_years": "Tenure", "log_balance": "Balance", "products_held": "Products held",
    "login_frequency_30d": "Digital log-ins", "branch_visits_30d": "Branch visits", "monthly_txn_count": "Transactions",
    "complaints_12m": "Complaints", "salary_account": "Salary account", "has_loan": "Has loan", "csat_score": "Satisfaction",
    "digital_active": "Digital active", "credit_score": "Credit score", "branch_closure_flag": "Branch consolidated",
    "product_sunset_flag": "Product sunset", "fee_increase_flag": "Fee increase",
}

# reason -> (action, channel, cost per customer INR, expected relative churn reduction)
PLAYBOOK = {
    "Branch consolidation": ("Relocation concierge + priority service at the new branch", "RM call + SMS", 450, .30),
    "Product sunset": ("Like-for-like product migration with fee / rate protection", "Branch + app", 300, .28),
    "Fee harmonisation": ("12-month fee waiver or grandfathered pricing", "App + e-mail", 150, .22),
    "Unresolved service issues": ("Service-recovery call within 48 hours and a named owner", "Phone", 250, .25),
    "Low digital engagement": ("Guided digital onboarding with a first-transaction reward", "App / push", 120, .15),
    "Early-tenure relationship": ("Welcome journey with a cross-sell bundle", "E-mail + app", 180, .15),
    "Single-product relationship": ("Cross-sell a second product (card / RD / insurance) with a bonus", "App + branch", 220, .14),
    "Low satisfaction": ("NPS follow-up and personal outreach by the RM", "Phone", 200, .18),
}
REASON_PRIORITY = list(PLAYBOOK)       # order = which reason leads the playbook


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X["log_balance"] = np.log10(X["avg_balance_inr"].clip(lower=1000))
    for c in ("branch_closure_flag", "product_sunset_flag", "fee_increase_flag"):
        if c not in X:
            X[c] = 0
    return X[FEATURES]


def train(history: pd.DataFrame, seed: int = 42) -> TrainedSuite:
    X, y = build_features(history), history["churned_12m"]
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=.5),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=10, min_samples_leaf=10, random_state=seed, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=.06, max_depth=3, random_state=seed),
    }
    return fit_suite(X, y, estimators, scale_for=("Logistic Regression",), seed=seed, threshold=.35)


def _reasons(df: pd.DataFrame) -> pd.DataFrame:
    csat = df.csat_score
    return pd.DataFrame({
        "Branch consolidation": df.branch_closure_flag == 1,
        "Product sunset": df.product_sunset_flag == 1,
        "Fee harmonisation": df.fee_increase_flag == 1,
        "Unresolved service issues": df.complaints_12m >= 2,
        "Low digital engagement": (df.login_frequency_30d <= 2) & (df.age < 45),
        "Early-tenure relationship": df.tenure_years < 2,
        "Single-product relationship": df.products_held == 1,
        "Low satisfaction": csat.notna() & (csat <= 5),
    })


def risk_band(p, medium: float = ASSUMPTIONS["risk_medium"], high: float = ASSUMPTIONS["risk_high"]):
    return np.where(p >= high, "High", np.where(p >= medium, "Medium", "Low"))


def score_customers(suite: TrainedSuite, customers: pd.DataFrame, model: str | None = None,
                    medium: float = ASSUMPTIONS["risk_medium"], high: float = ASSUMPTIONS["risk_high"]) -> pd.DataFrame:
    model = model or suite.best_model
    out = customers.copy()
    out["churn_probability"] = suite.predict_proba(model, build_features(out))
    out["risk_band"] = risk_band(out.churn_probability.to_numpy(), medium, high)
    out["revenue_at_risk_inr"] = (out.churn_probability * out.est_annual_revenue_inr).round(0)
    R = _reasons(out)
    out["top_reasons"] = [", ".join([r for r in REASON_PRIORITY if row[r]][:3]) or "General attrition risk"
                          for _, row in R.iterrows()]
    lead = [next((r for r in REASON_PRIORITY if row[r]), None) for _, row in R.iterrows()]
    out["lead_reason"] = lead
    out["recommended_action"] = [PLAYBOOK[r][0] if r else "Standard relationship management" for r in lead]
    out["channel"] = [PLAYBOOK[r][1] if r else "-" for r in lead]
    out["action_cost_inr"] = [PLAYBOOK[r][2] if r else 0 for r in lead]
    out["expected_uplift"] = [PLAYBOOK[r][3] if r else 0.0 for r in lead]
    out["expected_saving_inr"] = (out.revenue_at_risk_inr * out.expected_uplift).round(0)
    out["net_value_inr"] = out.expected_saving_inr - out.action_cost_inr
    return out


def reason_counts(scored: pd.DataFrame) -> pd.DataFrame:
    R = _reasons(scored)
    rows = []
    for r in REASON_PRIORITY:
        m = R[r]
        rows.append(dict(reason=r, customers=int(m.sum()), avg_churn_probability=float(scored.loc[m, "churn_probability"].mean()) if m.any() else 0.0,
                         revenue_at_risk_cr=float(scored.loc[m, "revenue_at_risk_inr"].sum() / 1e7),
                         action=PLAYBOOK[r][0], channel=PLAYBOOK[r][1], cost_per_customer_inr=PLAYBOOK[r][2],
                         expected_uplift_pct=PLAYBOOK[r][3] * 100))
    return pd.DataFrame(rows)


def simulate_budget(scored: pd.DataFrame, budget_inr: float) -> dict:
    """Pick the highest net-value interventions that fit the budget."""
    cand = scored[(scored.net_value_inr > 0) & (scored.action_cost_inr > 0)].sort_values("net_value_inr", ascending=False)
    chosen = cand[cand.action_cost_inr.cumsum() <= budget_inr]
    spend = float(chosen.action_cost_inr.sum())
    saved = float(chosen.expected_saving_inr.sum())
    return dict(targeted=chosen, n=len(chosen), spend=spend, saved=saved, net=saved - spend,
                roi=(saved / spend) if spend else 0.0, eligible=len(cand),
                full_cost=float(cand.action_cost_inr.sum()), full_saved=float(cand.expected_saving_inr.sum()))
