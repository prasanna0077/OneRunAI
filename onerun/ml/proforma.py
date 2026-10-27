"""Combined-bank (pro-forma) view: Our Bank + target, Day-1 simple merge.

Day-1 means the two balance sheets are added together with no synergies, no customer run-off and no
purchase-accounting adjustments. Totals are summed; ratios are re-computed from the summed amounts
(i.e. weighted by the right base), never averaged.

Proxies (documented, fictional): operating income = assets x (NIM + non-interest income %), net profit = ROA x assets,
net worth of the target comes from the candidate data, CAR is weighted by assets (risk-weighted assets are not modelled).
"""
from __future__ import annotations

import pandas as pd

from onerun.config import ACQUIRER_ASSUMPTIONS

# metric key -> (label, unit, higher_is_better or None for neutral)
METRICS = {
    "total_assets_cr": ("Total assets", "₹ Cr", True),
    "total_deposits_cr": ("Total deposits", "₹ Cr", True),
    "casa_deposits_cr": ("CASA deposits", "₹ Cr", True),
    "total_advances_cr": ("Total advances (loans)", "₹ Cr", True),
    "networth_cr": ("Net worth", "₹ Cr", True),
    "gross_npa_cr": ("Gross NPA", "₹ Cr", False),
    "provisions_cr": ("Provisions held", "₹ Cr", True),
    "net_npa_cr": ("Net NPA", "₹ Cr", False),
    "net_profit_cr": ("Net profit", "₹ Cr", True),
    "branches": ("Branches", "#", None),
    "customers_lakh": ("Customers", "lakh", True),
    "employees": ("Employees", "#", None),
    "casa_pct": ("CASA ratio", "%", True),
    "gnpa_pct": ("Gross NPA ratio", "%", False),
    "nnpa_pct": ("Net NPA ratio", "%", False),
    "pcr_pct": ("Provision coverage", "%", True),
    "credit_deposit_pct": ("Credit-deposit ratio", "%", None),
    "roa_pct": ("Return on assets", "%", True),
    "roe_pct": ("Return on equity", "%", True),
    "car_pct": ("Capital adequacy", "%", True),
    "nim_pct": ("Net interest margin", "%", True),
    "cost_to_income_pct": ("Cost-to-income", "%", False),
    "digital_adoption_pct": ("Digital adoption", "%", True),
}
AMOUNT_KEYS = [k for k, v in METRICS.items() if v[1] in ("₹ Cr", "#", "lakh")]
RATIO_KEYS = [k for k in METRICS if k not in AMOUNT_KEYS]


def _bank_base(d) -> dict:
    """Rupee amounts and ratios for one bank. `d` is a dict (Our Bank) or a candidate row (target)."""
    g = lambda k: float(d[k])            # noqa: E731
    assets, adv, dep = g("total_assets_cr"), g("total_advances_cr") if "total_advances_cr" in d else g("advances_cr"), \
        g("total_deposits_cr") if "total_deposits_cr" in d else g("deposits_cr")
    gnpa_cr = adv * g("gnpa_pct") / 100
    income = assets * (g("nim_pct") + ACQUIRER_ASSUMPTIONS["non_interest_income_pct_assets"]) / 100
    return dict(
        total_assets_cr=assets, total_deposits_cr=dep, casa_deposits_cr=dep * g("casa_pct") / 100, total_advances_cr=adv,
        networth_cr=g("networth_cr"), gross_npa_cr=gnpa_cr, net_npa_cr=adv * g("nnpa_pct") / 100,
        provisions_cr=gnpa_cr * g("pcr_pct") / 100, net_profit_cr=assets * g("roa_pct") / 100,
        operating_income_cr=income, operating_cost_cr=income * g("cost_to_income_pct") / 100,
        branches=g("branches"), customers_lakh=g("customers_lakh"),
        employees=g("employees") if "employees" in d else 0.0,
        digital_customers=g("customers_lakh") * g("digital_adoption_pct") / 100,
        car_assets=assets * g("car_pct"), nim_assets=assets * g("nim_pct"),
    )


def _ratios(b: dict) -> dict:
    return dict(
        casa_pct=b["casa_deposits_cr"] / b["total_deposits_cr"] * 100,
        gnpa_pct=b["gross_npa_cr"] / b["total_advances_cr"] * 100,
        nnpa_pct=b["net_npa_cr"] / b["total_advances_cr"] * 100,
        pcr_pct=b["provisions_cr"] / b["gross_npa_cr"] * 100 if b["gross_npa_cr"] else 0.0,
        credit_deposit_pct=b["total_advances_cr"] / b["total_deposits_cr"] * 100,
        roa_pct=b["net_profit_cr"] / b["total_assets_cr"] * 100,
        roe_pct=b["net_profit_cr"] / b["networth_cr"] * 100,
        car_pct=b["car_assets"] / b["total_assets_cr"],
        nim_pct=b["nim_assets"] / b["total_assets_cr"],
        cost_to_income_pct=b["operating_cost_cr"] / b["operating_income_cr"] * 100,
        digital_adoption_pct=b["digital_customers"] / b["customers_lakh"] * 100,
    )


def combine(our_bank: dict, target) -> dict:
    """-> dict(table=DataFrame, base={'Our Bank','Target','Combined'}) for the Day-1 simple merge.

    `target` is one row of the scored candidates table. The target's employees are estimated from its branches
    if the column is missing (the candidate data has it).
    """
    ours = _bank_base(our_bank)
    tgt = _bank_base(target)
    both = {k: ours[k] + tgt[k] for k in ours}
    bases = {"Our Bank": ours, "Target": tgt, "Combined": both}
    vals = {name: {**{k: b[k] for k in AMOUNT_KEYS}, **_ratios(b)} for name, b in bases.items()}

    rows = []
    for key, (label, unit, hib) in METRICS.items():
        o, t, c = (vals[n][key] for n in ("Our Bank", "Target", "Combined"))
        delta = c - o
        if hib is None or abs(delta) < 1e-9:
            verdict = "Neutral" if hib is None else "No change"
        else:
            verdict = "Better" if (delta > 0) == hib else "Worse"
        rows.append(dict(key=key, Metric=label, Unit=unit, **{"Our Bank": o, "Target": t, "Combined": c},
                         Change=delta, **{"Change %": delta / o * 100 if o else 0.0}, Impact=verdict))
    table = pd.DataFrame(rows)
    return dict(table=table, base=bases, values=vals)


def deposit_split(base: dict) -> pd.DataFrame:
    return pd.DataFrame({"Type": ["CASA", "Term deposits"],
                         "₹ Cr": [base["casa_deposits_cr"], base["total_deposits_cr"] - base["casa_deposits_cr"]]})


def loan_quality(base: dict) -> pd.DataFrame:
    perf = base["total_advances_cr"] - base["gross_npa_cr"]
    return pd.DataFrame({"Bucket": ["Performing loans", "Net NPA", "Provisions held"],
                         "₹ Cr": [perf, base["net_npa_cr"], base["provisions_cr"]]})


def summary_lines(res: dict, our_name: str, target_name: str) -> list[str]:
    """Plain-English takeaways for the combined bank."""
    t = res["table"].set_index("key")
    v = lambda k, col: float(t.loc[k, col])    # noqa: E731
    lines = [
        f"Combined balance sheet: **₹{v('total_assets_cr', 'Combined'):,.0f} Cr** of assets "
        f"(+{v('total_assets_cr', 'Change %'):.0f}% vs {our_name} today), "
        f"**₹{v('total_deposits_cr', 'Combined'):,.0f} Cr** deposits and **₹{v('total_advances_cr', 'Combined'):,.0f} Cr** loans.",
        f"Reach grows to **{v('branches', 'Combined'):,.0f} branches** and **{v('customers_lakh', 'Combined'):,.1f} lakh customers**.",
    ]
    for key, name in (("gnpa_pct", "Gross NPA"), ("casa_pct", "CASA"), ("roa_pct", "ROA"), ("car_pct", "Capital adequacy")):
        imp = t.loc[key, "Impact"]
        word = {"Better": "improves", "Worse": "weakens"}.get(imp, "is unchanged")
        lines.append(f"{name} {word}: {v(key, 'Our Bank'):.2f}% → **{v(key, 'Combined'):.2f}%** "
                     f"(target stand-alone {v(key, 'Target'):.2f}%).")
    return lines
