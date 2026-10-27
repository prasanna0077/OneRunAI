"""Integration + Retention data: branches, products, customers and past-merger churn history.

Customers are drawn from six hidden behavioural archetypes (mass savers, digital natives,
affluent, MSME, seniors, multi-product) whose mix depends on the bank type. The archetype is
*not* stored - the Integration module has to rediscover the segments by clustering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from onerun.config import ACQUIRER
from onerun.data_generation.banks import BANK_TYPE_LIST
from onerun.data_generation.common import CITIES, REGIONS, acquirer_region_share, sigmoid, solve_intercept

# --------------------------------------------------------------------------- products
# type, category, variants, share of customers holding, revenue per account (INR / yr)
PRODUCT_CATALOG = [
    ("Savings Account", "Deposit", ["Basic Savings", "Premium Savings", "Salary Plus Savings", "Senior Citizen Savings"], .62, 900),
    ("Current Account", "Deposit", ["Regular Current", "Business Plus Current"], .08, 6500),
    ("Fixed Deposit", "Deposit", ["Regular FD", "Tax-Saver FD"], .22, 3200),
    ("Recurring Deposit", "Deposit", ["Recurring Deposit"], .08, 700),
    ("Home Loan", "Loan", ["Home Loan"], .035, 21000),
    ("Personal Loan", "Loan", ["Personal Loan"], .06, 14000),
    ("Vehicle Loan", "Loan", ["Vehicle Loan"], .03, 10000),
    ("MSME Loan", "Loan", ["MSME Working Capital Loan"], .035, 38000),
    ("Gold Loan", "Loan", ["Gold Loan"], .05, 9000),
    ("Credit Card", "Fee-based", ["Classic Credit Card", "Rewards Credit Card"], .07, 5200),
    ("Insurance", "Fee-based", ["Life Insurance (Bancassurance)", "Health Insurance"], .09, 2200),
    ("Demat & Trading", "Fee-based", ["Demat & Trading Account"], .035, 1800),
]
# availability by bank type, in BANK_TYPE_LIST order (Private, SFB, Co-op, Old-gen)
PRODUCT_AVAIL = {
    "Savings Account": (1, 1, 1, 1), "Current Account": (1, .95, .95, 1), "Fixed Deposit": (1, 1, 1, 1),
    "Recurring Deposit": (.8, .9, 1, .9), "Home Loan": (.95, .7, .6, .85), "Personal Loan": (1, .9, .7, .9),
    "Vehicle Loan": (.9, .8, .6, .8), "MSME Loan": (.8, 1, .7, .7), "Gold Loan": (.5, 1, .5, .9),
    "Credit Card": (1, .5, .1, .6), "Insurance": (.95, .7, .55, .8), "Demat & Trading": (.8, .2, .08, .5),
}
ACQ_MISSING_TYPES = {"Gold Loan"}      # the acquirer does not offer these -> target's version is unique
LOAN_RATE = {"Home Loan": 1.2, "Personal Loan": 4.5, "Vehicle Loan": 2.4, "MSME Loan": 5.0, "Gold Loan": .8}


def make_products(rng, banks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (target_products, acquirer_products)."""
    acq_rows, acq_by_type = [], {}
    for ptype, cat, variants, share, rpa in PRODUCT_CATALOG:
        if ptype in ACQ_MISSING_TYPES:
            continue
        for v in variants:
            pid = f"ACQ-P{len(acq_rows) + 1:02d}"
            accounts = int(ACQUIRER["customers_lakh"] * 1e5 * share / len(variants) * rng.uniform(.8, 1.2))
            acq_rows.append(dict(product_id=pid, bank_id=ACQUIRER["bank_id"], product_name=v, product_type=ptype,
                                 category=cat, active_accounts=accounts,
                                 annual_revenue_cr=round(accounts * rpa / 1e7, 1), revenue_per_account_inr=rpa))
            acq_by_type.setdefault(ptype, []).append(pid)
    acq = pd.DataFrame(acq_rows)

    rows = []
    for _, b in banks.iterrows():
        ti = BANK_TYPE_LIST.index(b.bank_type)
        n_cust = b.customers_lakh * 1e5
        k = 0
        for ptype, cat, variants, share, rpa in PRODUCT_CATALOG:
            if rng.random() > PRODUCT_AVAIL[ptype][ti]:
                continue
            split = rng.dirichlet(np.ones(len(variants)) * 3)
            mult = np.exp(rng.normal(0, .35))
            for v, sp in zip(variants, split):
                k += 1
                accounts = max(50, int(n_cust * share * sp * mult * rng.uniform(.7, 1.3)))
                rev_pa = rpa * np.exp(rng.normal(0, .22))
                equiv = acq_by_type.get(ptype)
                has_eq = equiv is not None
                rows.append(dict(
                    product_id=f"{b.bank_id}-P{k:02d}", bank_id=b.bank_id, product_name=v, product_type=ptype,
                    category=cat, active_accounts=accounts, annual_revenue_cr=round(accounts * rev_pa / 1e7, 2),
                    revenue_per_account_inr=round(rev_pa), yoy_growth_pct=round(rng.normal(8, 10), 1),
                    acquirer_equivalent_id=rng.choice(equiv) if has_eq else np.nan,
                    feature_overlap_pct=round(float(np.clip(rng.beta(5, 2) * 100, 35, 98)), 1) if has_eq else np.nan,
                    npa_rate_pct=round(float(np.clip(rng.normal(LOAN_RATE[ptype] * 2, 1.2), .2, 14)), 2)
                    if ptype in LOAN_RATE else np.nan,
                ))
    return pd.DataFrame(rows), acq


# --------------------------------------------------------------------------- branches
_BRANCH_TYPES = {  # footfall median, cost (lakh / month) median
    "Metro": (420, 14.0), "Urban": (300, 9.0), "Semi-Urban": (190, 5.5), "Rural": (90, 3.2),
}


def make_branches(rng, banks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, b in banks.iterrows():
        n_br = int(min(b.branches, np.clip(round(b.branches * .12), 14, 30)))
        home = CITIES[CITIES.region == b.region]
        for i in range(n_br):
            pool = home if rng.random() < .7 else CITIES
            c = pool.iloc[rng.integers(len(pool))]
            if c.tier == 1:
                btype = rng.choice(["Metro", "Urban"], p=[.55, .45])
            elif c.tier == 2:
                btype = rng.choice(["Urban", "Semi-Urban"], p=[.6, .4])
            else:
                btype = rng.choice(["Semi-Urban", "Rural"], p=[.55, .45])
            foot_m, cost_m = _BRANCH_TYPES[btype]
            footfall = int(np.clip(foot_m * np.exp(rng.normal(0, .38)) * (b.customers_lakh / b.branches * 1e5 / 9000) ** .3, 25, 1500))
            txns = int(footfall * rng.uniform(1.5, 3.2) * (1 - .004 * b.digital_txn_share_pct))
            cost = round(cost_m * np.exp(rng.normal(0, .22)), 2)
            share = acquirer_region_share(c.region)
            dist_base = (1 + 14 * (1 - share) ** 2) * {1: .5, 2: .8, 3: 1.2}[int(c.tier)]
            dist = float(np.clip(rng.gamma(1.5, dist_base / 1.5), .2, 60))
            rows.append(dict(
                branch_id=f"{b.bank_id}-BR{i + 1:02d}", bank_id=b.bank_id, city=c.city, state=c.state,
                region=c.region, city_tier=int(c.tier), branch_type=btype,
                lat=round(c.lat + rng.normal(0, .08), 4), lon=round(c.lon + rng.normal(0, .08), 4),
                footfall_daily=footfall, transactions_daily=txns,
                monthly_cost_lakh=cost, staff_count=int(np.clip(round(cost / .75 + rng.normal(0, 1.5)), 3, 45)),
                deposits_cr=round(float(b.deposits_cr / b.branches * np.exp(rng.normal(0, .5))), 1),
                distance_to_acq_branch_km=round(dist, 1),
            ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- customers
SEGMENTS = {   # hidden archetypes: age(m,sd) tenure(m,sd) balance(median,sigma) products txn login visits loan salary
    "mass": dict(age=(38, 11), ten=(8, 4), bal=(18_000, .8), prod=1.6, txn=14, login=3, vis=2.2, loan=.12, sal=.18),
    "digital": dict(age=(29, 5), ten=(3, 2), bal=(32_000, .7), prod=2.0, txn=38, login=24, vis=.4, loan=.22, sal=.60),
    "affluent": dict(age=(49, 8), ten=(12, 5), bal=(650_000, .6), prod=3.9, txn=28, login=12, vis=1.5, loan=.35, sal=.50),
    "msme": dict(age=(42, 8), ten=(7, 4), bal=(240_000, .7), prod=2.8, txn=85, login=14, vis=5, loan=.55, sal=.02),
    "senior": dict(age=(66, 6), ten=(18, 6), bal=(140_000, .7), prod=2.1, txn=8, login=.6, vis=4.5, loan=.03, sal=.0),
    "multi": dict(age=(44, 9), ten=(10, 4), bal=(220_000, .6), prod=4.8, txn=30, login=11, vis=1.8, loan=.60, sal=.40),
}
SEG_ORDER = list(SEGMENTS)
SEGMENT_MIX = {   # by bank type, in SEG_ORDER order
    "Private Sector Bank": [.24, .26, .16, .10, .08, .16],
    "Small Finance Bank": [.38, .18, .04, .20, .06, .14],
    "Urban Co-operative Bank": [.32, .04, .06, .18, .26, .14],
    "Old-Generation Private Bank": [.30, .10, .10, .14, .20, .16],
}
SEG_PRIMARY_PRODUCT = {   # archetype -> weights over product types
    "mass": {"Savings Account": .7, "Fixed Deposit": .1, "Recurring Deposit": .1, "Personal Loan": .04, "Gold Loan": .06},
    "digital": {"Savings Account": .7, "Personal Loan": .1, "Credit Card": .1, "Insurance": .05, "Recurring Deposit": .05},
    "affluent": {"Savings Account": .2, "Fixed Deposit": .25, "Home Loan": .15, "Demat & Trading": .15, "Insurance": .1, "Credit Card": .15},
    "msme": {"Current Account": .5, "MSME Loan": .3, "Savings Account": .1, "Fixed Deposit": .1},
    "senior": {"Fixed Deposit": .5, "Savings Account": .35, "Recurring Deposit": .1, "Insurance": .05},
    "multi": {"Home Loan": .25, "Savings Account": .2, "Vehicle Loan": .1, "Credit Card": .15, "Fixed Deposit": .15, "Personal Loan": .15},
}


def _draw_customers(rng, n: int, mix, login_mult: float = 1.0) -> pd.DataFrame:
    """Core behavioural attributes for n customers drawn from the six archetypes."""
    seg = rng.choice(SEG_ORDER, size=n, p=np.asarray(mix) / np.sum(mix))
    S = pd.DataFrame([SEGMENTS[s] for s in seg])
    age = np.clip(np.round(rng.normal(S.age.str[0], S.age.str[1])), 18, 90).astype(int)
    tenure = np.clip(rng.normal(S.ten.str[0], S.ten.str[1]), .2, 40)
    tenure = np.minimum(tenure, np.maximum(age - 18, .3)).round(1)
    balance = np.exp(np.log(S.bal.str[0].astype(float)) + S.bal.str[1].astype(float) * rng.normal(size=n)).round(-1)
    products = np.clip(np.round(rng.normal(S["prod"], .8)), 1, 7).astype(int)
    txn = rng.poisson(S.txn.astype(float))
    login = rng.poisson(S.login.astype(float) * login_mult)
    visits = rng.poisson(S.vis.astype(float))
    has_loan = (rng.random(n) < S.loan.astype(float)).astype(int)
    salary = (rng.random(n) < S.sal.astype(float)).astype(int)
    has_cc = (rng.random(n) < np.clip(.04 + .12 * products, 0, .7)).astype(int)
    has_ins = (rng.random(n) < np.clip(.05 + .07 * products, 0, .6)).astype(int)
    complaints = rng.poisson(.35, n)
    score = np.clip(rng.normal(700, 55, n) + np.where(np.isin(seg, ["affluent", "multi"]), 25, 0)
                    - np.where(seg == "mass", 20, 0), 350, 900).round()
    score = np.where(rng.random(n) < .04, np.nan, score)                         # ~4% missing bureau score
    csat = np.clip(np.round(rng.normal(7.0 - .5 * complaints, 1.8)), 1, 10)
    csat = np.where(rng.random(n) < .15, np.nan, csat)                           # ~15% survey non-response
    revenue = (balance * .021 + products * 700 + txn * 30 + has_loan * 4500 + has_cc * 2200).round(-1)
    df = pd.DataFrame({
        "age": age, "tenure_years": tenure, "avg_balance_inr": balance, "products_held": products,
        "has_loan": has_loan, "has_credit_card": has_cc, "has_insurance": has_ins,
        "salary_account": salary, "monthly_txn_count": txn, "login_frequency_30d": login,
        "branch_visits_30d": visits, "digital_active": (login >= 2).astype(int), "complaints_12m": complaints,
        "credit_score": score, "csat_score": csat, "est_annual_revenue_inr": revenue,
    })
    df["_segment"] = seg
    return df


def make_customers(rng, banks, branches, products, n_per_bank: int) -> pd.DataFrame:
    parts = []
    for _, b in banks.iterrows():
        df = _draw_customers(rng, n_per_bank, SEGMENT_MIX[b.bank_type], np.clip(b.app_mau_pct / 35, .4, 1.6))
        br = branches[branches.bank_id == b.bank_id]
        w = br.footfall_daily.to_numpy(float)
        df["primary_branch_id"] = rng.choice(br.branch_id.to_numpy(), size=n_per_bank, p=w / w.sum())
        pr = products[products.bank_id == b.bank_id]
        types_avail = set(pr.product_type)
        pids = []
        for seg in df["_segment"]:
            wts = {t: x for t, x in SEG_PRIMARY_PRODUCT[seg].items() if t in types_avail} or {"Savings Account": 1}
            t = rng.choice(list(wts), p=np.array(list(wts.values())) / sum(wts.values()))
            cand = pr[pr.product_type == t].product_id.to_numpy()
            pids.append(rng.choice(cand) if len(cand) else pr.product_id.iloc[0])
        df["primary_product_id"] = pids
        df["fee_increase_flag"] = (rng.random(n_per_bank) < .22).astype(int)     # fee-schedule harmonisation
        df.insert(0, "bank_id", b.bank_id)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True).drop(columns="_segment")
    out.insert(0, "customer_id", [f"CUST-{i + 1:06d}" for i in range(len(out))])
    return out


# --------------------------------------------------------------------------- past-merger churn history
def make_merger_history(rng, n: int, n_mergers: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Customers of 6 completed mergers with a 12-month churn outcome (training data for Retention)."""
    mergers = pd.DataFrame({
        "merger_id": [f"MRG-{2018 + i}-{chr(65 + i)}" for i in range(n_mergers)],
        "merger_year": [2018 + i for i in range(n_mergers)],
        "target_bank_type": rng.choice(BANK_TYPE_LIST, n_mergers),
        "branch_closure_rate": rng.uniform(.05, .28, n_mergers).round(3),
        "product_sunset_rate": rng.uniform(.04, .22, n_mergers).round(3),
        "fee_change_rate": rng.uniform(.10, .40, n_mergers).round(3),
        "integration_severity": rng.uniform(-.3, .5, n_mergers).round(2),
    })
    sizes = np.full(n_mergers, n // n_mergers)
    sizes[: n - sizes.sum()] += 1
    parts = []
    for (_, m), size in zip(mergers.iterrows(), sizes):
        df = _draw_customers(rng, int(size), SEGMENT_MIX[m.target_bank_type], rng.uniform(.6, 1.3))
        closure_p = np.clip(m.branch_closure_rate * (1 + .6 * (df.branch_visits_30d > 3)), 0, .9)
        df["branch_closure_flag"] = (rng.random(len(df)) < closure_p).astype(int)
        df["product_sunset_flag"] = (rng.random(len(df)) < m.product_sunset_rate).astype(int)
        df["fee_increase_flag"] = (rng.random(len(df)) < m.fee_change_rate).astype(int)
        df["_sev"] = m.integration_severity
        df.insert(0, "merger_id", m.merger_id)
        parts.append(df)
    h = pd.concat(parts, ignore_index=True)

    csat = h.csat_score.fillna(7)
    logit = (-.09 * np.minimum(h.tenure_years, 20) - .35 * (h.products_held - 2) - .035 * np.minimum(h.login_frequency_30d, 25)
             + .25 * (np.log10(h.avg_balance_inr.clip(lower=1000)) - 5) + .30 * np.minimum(h.complaints_12m, 4)
             - .12 * (csat - 7) + 1.05 * h.branch_closure_flag + .85 * h.product_sunset_flag + .55 * h.fee_increase_flag
             - .70 * h.salary_account - .50 * h.has_loan
             + np.where(h.age < 35, .025 * (35 - h.age), -.010 * (h.age - 35))
             + .12 * h.branch_closure_flag * np.minimum(h.branch_visits_30d, 8) + h._sev
             + rng.normal(0, .30, len(h))).to_numpy() * 1.25
    prob = sigmoid(logit + solve_intercept(logit, .18))
    h["churned_12m"] = (rng.random(len(h)) < prob).astype(int)
    h = h.drop(columns=["_segment", "_sev"])
    h.insert(0, "customer_id", [f"HIST-{i + 1:06d}" for i in range(len(h))])
    return h, mergers
