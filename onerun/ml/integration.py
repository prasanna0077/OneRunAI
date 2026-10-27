"""MVP 2 - Integration Intelligence.

Business question: *How should customers, products and operations be integrated?*
Engine: K-Means (customers, branches) + Hierarchical / Ward clustering (products).
Outputs a migration-wave plan, a product-rationalisation plan and a branch-consolidation plan.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from onerun.config import ACQUIRER, ASSUMPTIONS, THRESHOLDS

CUSTOMER_FEATURES = ["age", "tenure_years", "avg_balance_inr", "products_held", "monthly_txn_count",
                     "login_frequency_30d", "branch_visits_30d"]

SEGMENT_PLAYBOOK = {   # segment -> (wave, approach)
    "Digital Natives": (1, "Self-serve app migration with guided in-app onboarding"),
    "Mass Retail Savers": (2, "Assisted migration: SMS / IVR nudges plus branch help-desk"),
    "Multi-Product Relationship": (3, "RM-led migration that moves all linked products in a single touch"),
    "Active Transactors / MSME": (3, "Dedicated RM, account-number and ERP / API continuity"),
    "Affluent & Wealth": (3, "White-glove migration, no service downtime, wealth-desk hand-holding"),
    "Senior / Branch-Loyal": (4, "Branch-assisted migration with doorstep support and printed guides"),
    "Other / Mixed": (2, "Standard assisted migration"),
}
WAVE_LABEL = {1: "Wave 1 - Digital pilot", 2: "Wave 2 - Mass migration", 3: "Wave 3 - Relationship-led",
              4: "Wave 4 - Assisted / high-touch"}


# ----------------------------------------------------------------------------- customers
def _customer_matrix(df: pd.DataFrame) -> np.ndarray:
    X = df[CUSTOMER_FEATURES].copy()
    X["avg_balance_inr"] = np.log1p(X["avg_balance_inr"])
    return StandardScaler().fit_transform(X.fillna(X.median()))


def silhouette_scan(customers: pd.DataFrame, ks=range(3, 8), seed: int = 42) -> pd.DataFrame:
    X = _customer_matrix(customers)
    rows = []
    for k in ks:
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        rows.append((k, silhouette_score(X, lab, sample_size=min(len(X), 2000), random_state=seed)))
    return pd.DataFrame(rows, columns=["k", "silhouette"])


# desired centroid profile per segment name (weights on the standardised CUSTOMER_FEATURES)
_PROFILE = {
    "Affluent & Wealth":          dict(avg_balance_inr=1.0, products_held=.3),
    "Senior / Branch-Loyal":      dict(age=1.0, tenure_years=.5, login_frequency_30d=-.5, branch_visits_30d=.5),
    "Digital Natives":            dict(login_frequency_30d=1.0, age=-.8, branch_visits_30d=-.5, tenure_years=-.3),
    "Mass Retail Savers":         dict(avg_balance_inr=-1.0, monthly_txn_count=-.5, products_held=-.5, login_frequency_30d=-.3),
    "Active Transactors / MSME":  dict(monthly_txn_count=1.0, branch_visits_30d=.5),
    "Multi-Product Relationship": dict(products_held=1.0, avg_balance_inr=.3),
}


def _name_segments(zcent: pd.DataFrame) -> dict:
    """Name each K-Means cluster from its standardised centroid (optimal one-to-one assignment)."""
    names = list(_PROFILE)
    aff = np.array([[sum(w * zcent.loc[c, f] for f, w in _PROFILE[n].items()) for n in names] for c in zcent.index])
    rows, cols = linear_sum_assignment(-aff)
    out = {zcent.index[r]: names[c] for r, c in zip(rows, cols)}
    for c in zcent.index:                       # k > 6 -> leftover clusters
        out.setdefault(c, "Other / Mixed")
    return out


def segment_customers(customers: pd.DataFrame, k: int = 5, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = customers.copy()
    X = _customer_matrix(df)
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    names = _name_segments(pd.DataFrame(X, columns=CUSTOMER_FEATURES).groupby(lab).mean())
    df["segment"] = pd.Series(lab, index=df.index).map(names)
    df["migration_wave"] = df["segment"].map(lambda s: SEGMENT_PLAYBOOK[s][0])
    df["migration_approach"] = df["segment"].map(lambda s: SEGMENT_PLAYBOOK[s][1])
    summ = df.groupby("segment").agg(
        customers=("customer_id", "count"), avg_age=("age", "mean"), avg_tenure=("tenure_years", "mean"),
        avg_balance_inr=("avg_balance_inr", "mean"), avg_products=("products_held", "mean"),
        avg_logins_30d=("login_frequency_30d", "mean"), avg_branch_visits=("branch_visits_30d", "mean"),
        revenue_cr=("est_annual_revenue_inr", lambda s: s.sum() / 1e7),
    ).reset_index()
    summ["share_pct"] = summ.customers / summ.customers.sum() * 100
    summ["wave"] = summ.segment.map(lambda s: SEGMENT_PLAYBOOK[s][0])
    summ["approach"] = summ.segment.map(lambda s: SEGMENT_PLAYBOOK[s][1])
    return df, summ.sort_values(["wave", "customers"], ascending=[True, False]).reset_index(drop=True)


# ----------------------------------------------------------------------------- products
PRODUCT_ACTIONS = {
    "overlap_core": "Harmonise with acquirer product",
    "unique_driver": "Retain & cross-sell",
    "redundant": "Sunset & migrate customers",
    "niche": "Review selectively",
}


def product_matrix(prod: pd.DataFrame) -> np.ndarray:
    X = pd.DataFrame({
        "log_revenue": np.log1p(prod.annual_revenue_cr * 100),
        "log_accounts": np.log1p(prod.active_accounts),
        "overlap": prod.feature_overlap_pct.fillna(0),
        "growth": prod.yoy_growth_pct.fillna(0),
    })
    return StandardScaler().fit_transform(X)


def cluster_products(products: pd.DataFrame, acq_products: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    df = products.copy().reset_index(drop=True)
    k = max(2, min(k, len(df) - 1))
    X = product_matrix(df)
    lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    rev_z, ov_z = (pd.Series(X[:, 0]).groupby(lab).mean(), pd.Series(X[:, 2]).groupby(lab).mean())
    kind, label = {}, {}
    for c in range(k):
        if rev_z[c] > 0 and ov_z[c] > 0: kind[c], label[c] = "overlap_core", "Overlapping core"
        elif rev_z[c] > 0: kind[c], label[c] = "unique_driver", "Unique revenue drivers"
        elif ov_z[c] > 0: kind[c], label[c] = "redundant", "Redundant tail"
        else: kind[c], label[c] = "niche", "Niche low-revenue"
    df["cluster"] = [f"C{c + 1} - {label[c]}" for c in lab]
    df["recommended_action"] = [PRODUCT_ACTIONS[kind[c]] for c in lab]
    no_equiv = df.acquirer_equivalent_id.isna()      # nothing to merge into -> keep unless it is a low-value niche
    df.loc[no_equiv & (df.recommended_action != PRODUCT_ACTIONS["niche"]), "recommended_action"] = PRODUCT_ACTIONS["unique_driver"]
    eq = acq_products.set_index("product_id")["product_name"]
    df["acquirer_equivalent"] = df.acquirer_equivalent_id.map(eq).fillna("- (no equivalent)")
    df["product_sunset_flag"] = (df.recommended_action == PRODUCT_ACTIONS["redundant"]).astype(int)
    return df


# ----------------------------------------------------------------------------- branches
BRANCH_TIER_NAMES = {2: ["Performing", "Under-utilised"], 3: ["Flagship", "Core", "Under-utilised"],
                     4: ["Flagship", "Core", "Marginal", "Under-utilised"]}


def cluster_branches(branches: pd.DataFrame, k: int = 3, seed: int = 42) -> pd.DataFrame:
    df = branches.copy().reset_index(drop=True)
    k = max(2, min(k, 4, len(df) - 1))
    df["cost_per_txn_inr"] = (df.monthly_cost_lakh * 1e5 / (df.transactions_daily.clip(lower=1) * 26)).round(1)
    df["deposits_per_lakh_cost"] = (df.deposits_cr * 100 / df.monthly_cost_lakh).round(1)
    X = pd.DataFrame({
        "footfall": np.log1p(df.footfall_daily), "txns": np.log1p(df.transactions_daily),
        "deposits": np.log1p(df.deposits_cr), "cost_txn": np.log1p(df.cost_per_txn_inr),
        "dist": np.log1p(df.distance_to_acq_branch_km),
    })
    Z = StandardScaler().fit_transform(X)
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
    perf = pd.Series(Z[:, 0] + Z[:, 1] + Z[:, 2] - Z[:, 3]).groupby(lab).mean().sort_values(ascending=False)
    rank = {c: i for i, c in enumerate(perf.index)}
    names = BRANCH_TIER_NAMES[k]
    df["branch_tier"] = [names[rank[c]] for c in lab]
    df["_rank"] = [rank[c] for c in lab]
    q80 = df.cost_per_txn_inr.quantile(.8)
    act = []
    for _, r in df.iterrows():
        if r._rank == 0: act.append("Retain")
        elif r.distance_to_acq_branch_km <= 3: act.append("Merge into acquirer branch")
        elif r._rank == k - 1 and r.cost_per_txn_inr >= q80: act.append("Shut down")
        elif r._rank == k - 1: act.append("Retain - monitor")
        else: act.append("Retain")
    df["recommended_action"] = act
    a = ASSUMPTIONS
    saving = np.where(df.recommended_action == "Merge into acquirer branch", a["branch_saving_merge"],
                      np.where(df.recommended_action == "Shut down", a["branch_saving_shutdown"], 0.0))
    df["annual_saving_lakh"] = (df.monthly_cost_lakh * 12 * saving).round(1)
    df["branch_closure_flag"] = df.recommended_action.isin(["Merge into acquirer branch", "Shut down"]).astype(int)
    return df.drop(columns="_rank")


# ----------------------------------------------------------------------------- orchestration
@dataclass
class IntegrationResult:
    bank_id: str
    customers: pd.DataFrame        # with segment / wave + branch & product exposure flags
    segments: pd.DataFrame
    silhouette: pd.DataFrame
    products: pd.DataFrame
    branches: pd.DataFrame
    waves: pd.DataFrame
    complexity: dict
    scale_customers: float = 1.0   # sample -> full customer base multiplier
    scale_branches: float = 1.0    # sample -> full branch network multiplier


def complexity_score(customers, products, branches, target_platform: str, target_tech: float) -> dict:
    c1 = float((customers.migration_wave >= 3).mean())
    c2 = float(branches.branch_closure_flag.mean())
    c3 = float((products.recommended_action.isin([PRODUCT_ACTIONS["redundant"], PRODUCT_ACTIONS["overlap_core"]])).mean())
    c4 = 0.0 if target_platform == ACQUIRER["core_banking_platform"] else (0.5 if target_tech >= 3.5 else 1.0)
    score = 100 * (.30 * c1 + .25 * c2 + .20 * c3 + .25 * c4)
    band = "Low" if score < THRESHOLDS["complexity_low"] else "Moderate" if score < THRESHOLDS["complexity_high"] else "High"
    return dict(score=score, band=band, high_touch_customers=c1, branch_changes=c2, product_changes=c3,
                platform_gap=c4, target_platform=target_platform, acquirer_platform=ACQUIRER["core_banking_platform"])


def run_integration(bank_id: str, customers: pd.DataFrame, products: pd.DataFrame, acq_products: pd.DataFrame,
                    branches: pd.DataFrame, bank_row: pd.Series, k_customers: int = 5, k_products: int = 4,
                    k_branches: int = 3, seed: int = 42) -> IntegrationResult:
    cu = customers[customers.bank_id == bank_id]
    pr = products[products.bank_id == bank_id]
    br = branches[branches.bank_id == bank_id]
    cu, seg = segment_customers(cu, k_customers, seed)
    pr = cluster_products(pr, acq_products, k_products)
    br = cluster_branches(br, k_branches, seed)
    # exposure flags consumed by Retention Intelligence (the MVP-to-MVP hand-off)
    cu["branch_closure_flag"] = cu.primary_branch_id.map(br.set_index("branch_id").branch_closure_flag).fillna(0).astype(int)
    cu["product_sunset_flag"] = cu.primary_product_id.map(pr.set_index("product_id").product_sunset_flag).fillna(0).astype(int)
    cu["home_city"] = cu.primary_branch_id.map(br.set_index("branch_id").city)
    waves = (cu.groupby("migration_wave").agg(customers=("customer_id", "count"),
                                              revenue_cr=("est_annual_revenue_inr", lambda s: s.sum() / 1e7))
             .reset_index())
    waves["wave"] = waves.migration_wave.map(WAVE_LABEL)
    waves["share_pct"] = waves.customers / waves.customers.sum() * 100
    cx = complexity_score(cu, pr, br, bank_row["core_banking_platform"], float(bank_row["tech_modernity_score"]))
    return IntegrationResult(bank_id, cu, seg, silhouette_scan(cu, range(3, min(8, len(cu) // 20 + 3)), seed), pr, br, waves, cx,
                             scale_customers=float(bank_row["customers_lakh"]) * 1e5 / max(len(cu), 1),
                             scale_branches=float(bank_row["branches"]) / max(len(br), 1))
