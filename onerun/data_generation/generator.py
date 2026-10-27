"""Generate all three MVP workbooks. Used by generate_data.py (CLI) and the Streamlit app."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from onerun.config import ACQ_FILE, DEFAULT_SEED, INT_FILE, RET_FILE
from onerun.data_generation.acquirer import financials_df
from onerun.data_generation.banks import acquirer_profile_df, make_candidates, make_history
from onerun.data_generation.common import make_rng
from onerun.data_generation.customers import (
    make_branches, make_customers, make_merger_history, make_products,
)
from onerun.data_generation.excel_writer import write_workbook


def generate_all(seed: int = DEFAULT_SEED, n_banks: int = 40, customers_per_bank: int = 250,
                 n_deals: int = 700, n_churn_history: int = 8000,
                 out_files: Optional[dict] = None,
                 progress: Optional[Callable[[str], None]] = None) -> dict:
    """Create the Excel datasets. Returns {file name: {sheet: (rows, cols)}}."""
    say = progress or (lambda m: None)
    files = {"acq": ACQ_FILE, "int": INT_FILE, "ret": RET_FILE, **(out_files or {})}
    manifest = {}

    # ---------------------------------------------------------------- MVP 1: Acquisition
    say("MVP 1 - Acquisition Intelligence: candidate banks + historical deals ...")
    cand = make_candidates(make_rng(seed, 1), n_banks)
    hist = make_history(make_rng(seed, 2), n_deals)
    id_cols = ["bank_id", "bank_name", "bank_type", "region"]
    sheets = {
        "Bank_Financials": cand[id_cols + ["total_assets_cr", "deposits_cr", "advances_cr", "networth_cr", "roa_pct",
                                           "roe_pct", "nim_pct", "cost_to_income_pct", "casa_pct", "price_to_book",
                                           "valuation_cr"]],
        "Credit_Metrics": cand[["bank_id", "gnpa_pct", "nnpa_pct", "pcr_pct", "car_pct", "credit_deposit_ratio_pct"]],
        "Business_Metrics": cand[["bank_id", "branches", "customers_lakh", "employees", "geo_overlap_pct", "size_ratio"]],
        "Digital_Metrics": cand[["bank_id", "app_mau_pct", "digital_adoption_pct", "digital_txn_share_pct", "app_rating",
                                 "tech_modernity_score", "core_banking_platform"]],
        "Historical_Deals": hist,
        "Acquirer_Profile": acquirer_profile_df(),
        "Acquirer_Financials": financials_df(),
    }
    manifest[Path(files["acq"]).name] = write_workbook(
        Path(files["acq"]), sheets, "Acquisition Intelligence dataset",
        [f"{n_banks} candidate target institutions screened for the client acquirer.",
         f"{n_deals} historical deals with a known outcome (synergy_realized) used to train the models."])

    # ---------------------------------------------------------------- MVP 2: Integration
    say("MVP 2 - Integration Intelligence: branches, products, customers ...")
    branches = make_branches(make_rng(seed, 3), cand)
    products, acq_products = make_products(make_rng(seed, 4), cand)
    customers = make_customers(make_rng(seed, 5), cand, branches, products, customers_per_bank)
    manifest[Path(files["int"]).name] = write_workbook(
        Path(files["int"]), {"Customers": customers, "Products": products, "Acquirer_Products": acq_products,
                             "Branches": branches},
        "Integration & Retention scoring dataset",
        [f"Representative samples: {customers_per_bank} customers and up to 30 branches per candidate institution.",
         "Customer segments are NOT labelled - the platform discovers them with clustering."])

    # ---------------------------------------------------------------- MVP 3: Retention
    say("MVP 3 - Customer Retention Intelligence: past-merger churn history ...")
    hist_c, mergers = make_merger_history(make_rng(seed, 6), n_churn_history)
    manifest[Path(files["ret"]).name] = write_workbook(
        Path(files["ret"]), {"Historical_Merger_Customers": hist_c, "Merger_Programmes": mergers},
        "Customer Retention training dataset",
        ["Customers of 6 completed mergers with a 12-month churn outcome (churned_12m).",
         "Used to train churn models that then score the target institution's customers."])

    say("Done.")
    return manifest


def data_ready() -> bool:
    return all(Path(f).exists() for f in (ACQ_FILE, INT_FILE, RET_FILE))


def ensure_data(progress: Optional[Callable[[str], None]] = None) -> None:
    if not data_ready():
        generate_all(progress=progress)


def read_sheets(path: Path) -> dict:
    return pd.read_excel(path, sheet_name=None)
