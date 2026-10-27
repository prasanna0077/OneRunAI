"""Acquisition Intelligence data: candidate target institutions + historical deals.

Realism comes from:
* a hidden "quality" factor per bank that correlates ROA, CASA, NPA, PCR, digital ...
* institution-type profiles (private / small-finance / urban co-op / old-gen private)
* geography relative to the acquirer's own branch footprint
* a historical-deal table whose outcome (synergy realised or not) is driven by those
  same fundamentals + an inverted-U effect of size and geographic overlap, plus noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from onerun.config import ACQUIRER
from onerun.data_generation.common import (
    REGIONS, acquirer_region_share, sigmoid, solve_intercept, zscore,
)

BANK_TYPES = {
    "Private Sector Bank": dict(w=.22, assets_med=65_000, assets_sig=.70, roa=1.05, casa=40, gnpa=2.2, nim=3.5,
                                ci=52, mau=42, dig=62, apb=110, cpb=9_000, eqm=9, tech=3.6, pb=1.8),
    "Small Finance Bank": dict(w=.30, assets_med=17_000, assets_sig=.60, roa=1.20, casa=26, gnpa=3.4, nim=6.0,
                               ci=64, mau=30, dig=48, apb=28, cpb=11_000, eqm=7, tech=3.2, pb=1.5),
    "Urban Co-operative Bank": dict(w=.28, assets_med=5_500, assets_sig=.60, roa=.55, casa=34, gnpa=6.0, nim=3.2,
                                    ci=68, mau=12, dig=26, apb=32, cpb=8_000, eqm=10, tech=2.2, pb=.9),
    "Old-Generation Private Bank": dict(w=.20, assets_med=38_000, assets_sig=.60, roa=.75, casa=35, gnpa=3.8,
                                        nim=3.0, ci=60, mau=24, dig=40, apb=80, cpb=8_500, eqm=10, tech=2.9, pb=1.0),
}
BANK_TYPE_LIST = list(BANK_TYPES)
PLATFORMS = ["Finacle", "Oracle Flexcube", "TCS BaNCS", "In-house / Legacy"]

_NAME_PREFIX = [
    "Cauvery", "Sahyadri", "Konkan", "Vindhya", "Ganga", "Himgiri", "Deccan", "Malabar", "Utkal", "Brahmaputra",
    "Aravali", "Narmada", "Godavari", "Coromandel", "Nilgiri", "Kosi", "Tapti", "Saurashtra", "Rajputana", "Chambal",
    "Bhagirathi", "Satpura", "Kalinga", "Pennar", "Sutlej", "Yamuna", "Vaigai", "Krishna", "Mahanadi", "Thar",
    "Western Ghats", "Shivalik", "Marwar", "Awadh", "Magadh", "Bundelkhand", "Kutch", "Mithila", "Tulu Nadu",
    "Rayalaseema", "Konkan Coast", "Malwa", "Doab", "Ahom", "Vidarbha", "Telangana Heritage", "Chola", "Pallava",
]
_SUFFIX = {
    "Private Sector Bank": "Bank Ltd",
    "Small Finance Bank": "Small Finance Bank",
    "Urban Co-operative Bank": "Urban Co-operative Bank",
    "Old-Generation Private Bank": "Bank",
}


def make_bank_features(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Generate n banks with correlated, type-specific financial / digital metrics."""
    w = np.array([BANK_TYPES[t]["w"] for t in BANK_TYPE_LIST])
    bank_type = rng.choice(BANK_TYPE_LIST, size=n, p=w / w.sum())
    P = pd.DataFrame([BANK_TYPES[t] for t in bank_type])
    g = lambda c: P[c].to_numpy(dtype=float)           # noqa: E731
    noise = lambda s: rng.normal(0, s, n)               # noqa: E731
    q = rng.normal(size=n)                              # hidden quality factor

    assets = np.clip(np.exp(np.log(g("assets_med")) + g("assets_sig") * rng.normal(size=n)), 1_500, 450_000)
    branches = np.maximum(12, np.round(assets / g("apb") * np.exp(noise(.25)))).astype(int)
    customers_lakh = np.round(branches * g("cpb") * np.exp(noise(.2)) / 1e5, 2)
    employees = (branches * np.clip(rng.normal(11, 2.5, n), 6, 20)).round().astype(int)

    roa = np.clip(g("roa") + .32 * q + noise(.15), -.6, 2.5)
    eqm = g("eqm") * np.exp(noise(.08))
    roe = np.clip(roa * eqm + noise(.8), -9, 25)
    casa = np.clip(g("casa") + 5 * q + noise(3.5), 8, 62)
    gnpa = np.clip(g("gnpa") * np.exp(-.28 * q + noise(.25)), .4, 18)
    pcr = np.clip(64 + 7 * q + noise(4), 35, 96)
    nnpa = np.clip(gnpa * (1 - pcr / 100) * rng.uniform(.9, 1.1, n), .05, 12)
    car = np.clip(15.2 + 1.6 * q + noise(1.2), 9.5, 26)
    nim = np.clip(g("nim") + .25 * q + noise(.25), 1.8, 9)
    cti = np.clip(g("ci") - 3.5 * q + noise(3), 35, 90)
    cdr = np.clip(rng.normal(78, 7, n), 55, 98)

    tech = np.clip(g("tech") + .35 * q + noise(.35), 1, 5)
    mau = np.clip(g("mau") + 6 * q + 4 * (tech - 3) + noise(3), 3, 78)
    dig = np.clip(np.maximum(g("dig") + 7 * q + 8 * (tech - 3) + noise(5), mau + 5), 10, 97)
    dtxn = np.clip(mau * 1.35 + noise(6), 8, 92)
    app_rating = np.clip(3.4 + .25 * (tech - 3) + .15 * q + noise(.15), 2.6, 4.8)
    app_rating = np.where(mau < 8, np.nan, app_rating)          # no meaningful app -> missing

    deposits = assets * np.clip(.78 + noise(.03), .65, .88)
    advances = deposits * cdr / 100
    networth = assets / eqm * np.exp(noise(.05))
    pb = np.clip(g("pb") * np.exp(.18 * q + .02 * (roe - 10) + noise(.08)), .4, 4.5)

    region = rng.choice(REGIONS, size=n, p=[.20, .28, .14, .22, .16])
    share = np.array([acquirer_region_share(r) for r in region])
    overlap = np.clip(share * 100 * 1.4 + noise(5), 2, 80)

    platform = []
    for t in tech:
        p = [.2, .2, .1, .5] if t < 2.6 else ([.35, .3, .2, .15] if t < 3.6 else [.4, .3, .28, .02])
        platform.append(rng.choice(PLATFORMS, p=p))

    return pd.DataFrame({
        "bank_type": bank_type, "region": region,
        "total_assets_cr": assets.round(0), "deposits_cr": deposits.round(0), "advances_cr": advances.round(0),
        "networth_cr": networth.round(0), "roa_pct": roa.round(2), "roe_pct": roe.round(2),
        "nim_pct": nim.round(2), "cost_to_income_pct": cti.round(1), "casa_pct": casa.round(1),
        "price_to_book": pb.round(2), "valuation_cr": (networth * pb).round(0),
        "gnpa_pct": gnpa.round(2), "nnpa_pct": nnpa.round(2), "pcr_pct": pcr.round(1), "car_pct": car.round(1),
        "credit_deposit_ratio_pct": cdr.round(1),
        "branches": branches, "customers_lakh": customers_lakh, "employees": employees,
        "geo_overlap_pct": overlap.round(1),
        "app_mau_pct": mau.round(1), "digital_adoption_pct": dig.round(1),
        "digital_txn_share_pct": dtxn.round(1), "app_rating": np.round(app_rating, 1),
        "tech_modernity_score": tech.round(2), "core_banking_platform": platform,
    })


def make_candidates(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """The live pipeline of potential target institutions the client is screening."""
    df = make_bank_features(rng, n)
    prefixes = list(rng.permutation(_NAME_PREFIX))
    names = []
    for i, t in enumerate(df["bank_type"]):
        base = prefixes[i % len(prefixes)]
        base = base if i < len(prefixes) else f"{base} II"
        names.append(f"{base} {_SUFFIX[t]}")
    df.insert(0, "bank_id", [f"BNK-{i + 1:03d}" for i in range(n)])
    df.insert(1, "bank_name", names)
    df["size_ratio"] = (df["total_assets_cr"] / ACQUIRER["total_assets_cr"]).round(4)
    return df


def make_history(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Historical financial-sector deals with a known outcome (synergy realised within 36 months)."""
    df = make_bank_features(rng, n)
    acq_assets = np.clip(np.exp(np.log(150_000) + .7 * rng.normal(size=n)), 20_000, 900_000)
    df["geo_overlap_pct"] = (rng.beta(2, 3, n) * 85).round(1)       # other acquirers, other footprints
    df["size_ratio"] = np.clip(df["total_assets_cr"] / acq_assets, .002, 1.5).round(4)
    df.insert(0, "deal_id", [f"DEAL-{i + 1:04d}" for i in range(n)])
    df.insert(1, "deal_year", rng.integers(2012, 2026, n))
    df.insert(2, "acquirer_assets_cr", acq_assets.round(0))

    z = zscore
    logit = (.75 * z(df.roa_pct) + .45 * z(df.casa_pct) - .70 * z(np.log(df.gnpa_pct)) + .35 * z(df.pcr_pct)
             + .55 * z(df.digital_adoption_pct) + .30 * z(df.tech_modernity_score) - .35 * z(df.cost_to_income_pct)
             - .90 * ((df.geo_overlap_pct - 35) / 25) ** 2                       # sweet-spot overlap ~35%
             - 1.10 * ((np.log(df.size_ratio) - np.log(.12)) / 1.2) ** 2)        # sweet-spot size ~12%
    logit = .55 * logit.to_numpy() + rng.normal(0, .35, n)      # damped signal + unobserved factors
    prob = sigmoid(logit + solve_intercept(logit, .52))
    success = (rng.random(n) < prob).astype(int)
    achieved = np.where(success == 1, rng.normal(104, 12, n), rng.normal(48, 18, n)).clip(5, 135)
    df["synergy_achieved_pct"] = achieved.round(1)
    df["synergy_realized"] = success
    return df


def acquirer_profile_df() -> pd.DataFrame:
    rows = []
    for k, v in ACQUIRER.items():
        if k == "region_branch_share":
            for r, s in v.items():
                rows.append((f"branch_share_{r.lower()}", s))
        else:
            rows.append((k, v))
    return pd.DataFrame(rows, columns=["attribute", "value"])
