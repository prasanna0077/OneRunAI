"""Rupee-amount balance sheet for the client bank (Meridian), derived from its ratios in config.py.

Used (1) to write the 'Acquirer_Financials' sheet and (2) as a fallback when the sheet is missing,
so older generated workbooks keep working.
"""
from __future__ import annotations

import pandas as pd

from onerun.config import ACQUIRER, ACQUIRER_ASSUMPTIONS as A

# metric -> (unit, description)
META = {
    "name": ("-", "Bank name"), "bank_type": ("-", "Institution category"), "hq_city": ("-", "Headquarters city"),
    "hq_state": ("-", "Headquarters state"), "established": ("year", "Year established"),
    "total_assets_cr": ("INR Cr", "Total assets"), "total_deposits_cr": ("INR Cr", "Total deposits"),
    "casa_deposits_cr": ("INR Cr", "Current + savings deposits"), "term_deposits_cr": ("INR Cr", "Term deposits"),
    "total_advances_cr": ("INR Cr", "Gross advances (loan book)"),
    "loans_retail_cr": ("INR Cr", "Retail loans"), "loans_msme_cr": ("INR Cr", "MSME loans"),
    "loans_corporate_cr": ("INR Cr", "Corporate loans"), "loans_agri_gold_cr": ("INR Cr", "Agri and gold loans"),
    "networth_cr": ("INR Cr", "Net worth"), "gross_npa_cr": ("INR Cr", "Gross NPA amount"),
    "net_npa_cr": ("INR Cr", "Net NPA amount"), "provisions_cr": ("INR Cr", "Provisions held against gross NPA"),
    "net_profit_cr": ("INR Cr", "Net profit (ROA x assets)"), "operating_income_cr": ("INR Cr", "Operating income proxy"),
    "operating_cost_cr": ("INR Cr", "Operating cost (cost-to-income x income)"),
    "employees": ("#", "Headcount"), "branches": ("#", "Branch count"), "customers_lakh": ("Lakh", "Customer base"),
    "roa_pct": ("%", "Return on assets"), "roe_pct": ("%", "Return on equity"), "casa_pct": ("%", "CASA ratio"),
    "gnpa_pct": ("%", "Gross NPA ratio"), "nnpa_pct": ("%", "Net NPA ratio"), "pcr_pct": ("%", "Provision coverage"),
    "car_pct": ("%", "Capital adequacy ratio"), "nim_pct": ("%", "Net interest margin"),
    "cost_to_income_pct": ("%", "Cost-to-income ratio"), "digital_adoption_pct": ("%", "Digital adoption"),
    "app_mau_pct": ("%", "Monthly active app users"), "tech_modernity_score": ("1-5", "Technology modernity"),
    "core_banking_platform": ("-", "Core banking system"),
}


def build_financials() -> dict:
    a = ACQUIRER
    assets = float(a["total_assets_cr"])
    deposits = assets * A["deposits_to_assets"]
    advances = deposits * A["credit_deposit_ratio"]
    income = assets * (a["nim_pct"] + A["non_interest_income_pct_assets"]) / 100
    mix = A["loan_mix"]
    out = {
        "name": a["name"], "bank_type": a["bank_type"], "hq_city": a["hq_city"], "hq_state": a["hq_state"],
        "established": a["established"], "total_assets_cr": assets, "total_deposits_cr": deposits,
        "casa_deposits_cr": deposits * a["casa_pct"] / 100, "term_deposits_cr": deposits * (1 - a["casa_pct"] / 100),
        "total_advances_cr": advances,
        "loans_retail_cr": advances * mix["Retail"], "loans_msme_cr": advances * mix["MSME"],
        "loans_corporate_cr": advances * mix["Corporate"], "loans_agri_gold_cr": advances * mix["Agri & Gold"],
        "networth_cr": assets * a["roa_pct"] / a["roe_pct"],
        "gross_npa_cr": advances * a["gnpa_pct"] / 100, "net_npa_cr": advances * a["nnpa_pct"] / 100,
        "provisions_cr": advances * a["gnpa_pct"] / 100 * a["pcr_pct"] / 100,
        "net_profit_cr": assets * a["roa_pct"] / 100, "operating_income_cr": income,
        "operating_cost_cr": income * a["cost_to_income_pct"] / 100,
        "employees": A["employees"], "branches": a["branches"], "customers_lakh": a["customers_lakh"],
    }
    for k in ("roa_pct", "roe_pct", "casa_pct", "gnpa_pct", "nnpa_pct", "pcr_pct", "car_pct", "nim_pct",
              "cost_to_income_pct", "digital_adoption_pct", "app_mau_pct", "tech_modernity_score", "core_banking_platform"):
        out[k] = a[k]
    # round rupee amounts only - ratios keep full precision so they match config.py exactly
    return {k: (round(v, 1) if isinstance(v, float) and k.endswith("_cr") else v) for k, v in out.items()}


def financials_df() -> pd.DataFrame:
    rows = [(k, v, *META.get(k, ("", ""))) for k, v in build_financials().items()]
    return pd.DataFrame(rows, columns=["metric", "value", "unit", "description"])
