"""OneRun AI - central configuration.

Everything that is a *setting* (paths, brand colours, the client/acquirer profile,
business assumptions, credits) lives here so the rest of the code stays clean.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

ACQ_FILE = DATA_DIR / "OneRun_Acquisition_Data.xlsx"
INT_FILE = DATA_DIR / "OneRun_Integration_Data.xlsx"
RET_FILE = DATA_DIR / "OneRun_Retention_Data.xlsx"
DATA_FILES = [ACQ_FILE, INT_FILE, RET_FILE]

DEFAULT_SEED = 2026

# --------------------------------------------------------------------------- brand
PRODUCT_NAME = "OneRun AI"
PRODUCT_SUBTITLE = "M&A Intelligence Platform"
SLOGAN = "One Run. Every M&A Decision."
VALUE_PROPOSITION = "De-Risk, Accelerate, and Maximize M&A Value - On a Single AI Platform."


class C:
    """Brand colours sampled from the OneRun AI pitch deck."""

    BG = "#262626"          # slide background
    BG_DEEP = "#0D0D0D"     # title / closing slide black
    CARD = "#2F2F2F"
    CARD_ALT = "#363636"
    BORDER = "#464646"
    TEXT = "#FFFFFF"
    MUTED = "#A9A9A9"

    PRIMARY = "#00AEEF"     # headings / "M&A Intelligence Platform" cyan
    BLUE = "#1485D8"        # title-slide blue band
    LOGO_BORDER = "#6DA8DC"
    TABLE_HEAD = "#DDEBF7"  # light-blue table header
    GREEN = "#00B050"       # "Solution" / outcomes
    RED = "#FF3B30"         # "Problem"
    AMBER = "#FFB000"

    BEFORE = "#9B8DD6"      # Before the Deal  (purple)
    DURING = "#D472D8"      # During the Deal  (orchid)
    AFTER = "#E45A85"       # After the Deal   (pink)

    SEQ = [PRIMARY, GREEN, DURING, AFTER, BEFORE, AMBER, BLUE, "#7FD6FF"]


STAGE_COLOURS = {
    "Before the Deal": C.BEFORE,
    "During the Deal": C.DURING,
    "After the Deal": C.AFTER,
    "Across the Lifecycle": C.PRIMARY,
}

# --------------------------------------------------------------------------- client (acquirer)
# The institution OneRun AI is advising in this demo. All values are fictional.
ACQUIRER = {
    "bank_id": "ACQ-000",
    "name": "Meridian Bank",
    "bank_type": "Private Sector Bank",
    "hq_region": "South",
    "hq_city": "Chennai",
    "hq_state": "Tamil Nadu",
    "established": 1994,
    "total_assets_cr": 240_000,
    "branches": 1_900,
    "customers_lakh": 145.0,
    "roa_pct": 1.08,
    "roe_pct": 13.4,
    "casa_pct": 41.5,
    "gnpa_pct": 2.1,
    "nnpa_pct": 0.55,
    "pcr_pct": 74.0,
    "car_pct": 16.4,
    "nim_pct": 3.5,
    "cost_to_income_pct": 48.0,
    "digital_adoption_pct": 68.0,
    "app_mau_pct": 52.0,
    "tech_modernity_score": 4.1,
    "core_banking_platform": "Finacle",
    # share of the acquirer's branch network by region (sums to 1.0)
    "region_branch_share": {"North": 0.15, "South": 0.45, "East": 0.07, "West": 0.28, "Central": 0.05},
}

# Balance-sheet assumptions used to turn Meridian's ratios into rupee amounts (fictional, edit freely).
# They feed the "Acquirer_Financials" Excel sheet, the Our Bank page and the combined-bank view.
ACQUIRER_ASSUMPTIONS = {
    "deposits_to_assets": 0.80,        # deposits as share of total assets
    "credit_deposit_ratio": 0.80,      # advances / deposits
    "employees": 21_500,
    "loan_mix": {"Retail": 0.38, "MSME": 0.22, "Corporate": 0.30, "Agri & Gold": 0.10},   # shares of advances
    "non_interest_income_pct_assets": 1.0,   # fee income as % of assets (used for the cost-to-income proxy)
}

# --------------------------------------------------------------------------- decision thresholds
# One place for every cut-off shown in the Executive Intelligence legend and used by the verdict logic.
THRESHOLDS = {
    "complexity_low": 35,              # integration complexity: Low < 35, Moderate 35-59, High >= 60
    "complexity_high": 60,
    "churn_limit_pct": 25.0,           # verdict "Proceed" needs expected churn below this
    "churn_reference_pct": 18.0,       # average 12-month churn in past mergers (training base rate)
}

# --------------------------------------------------------------------------- business assumptions
ASSUMPTIONS = {
    "synergy_pct_of_assets": 0.010,   # indicative annual synergy pool as % of target assets
    "branch_saving_merge": 0.65,      # share of branch cost saved when merged into an acquirer branch
    "branch_saving_shutdown": 0.85,   # share of branch cost saved when shut down
    "tier_prime": 70,                 # acquisition score cut-offs
    "tier_strong": 55,
    "tier_watch": 40,
    "risk_medium": 0.15,              # default churn-probability cut-offs
    "risk_high": 0.35,
}

# --------------------------------------------------------------------------- credits (edit freely)
CREDITS = {
    "presented_by": "Prasanna Kumar Rajasekaran",
    "programme_id": "iimt_ftai_2601056",
    "date": "31 - Oct - 2026",
    "contact": "",        # e.g. phone / e-mail - leave blank to hide
    "linkedin": "",       # e.g. https://www.linkedin.com/in/<handle>
    "faculty": [
        "Prof. Shalini Velappan",
        "Prof. Venkataraghavan Krishnaswamy",
        "Prof. J.B. Singh",
        "Prof. Gopinath K.",
        "Prof. Vinu C.T.",
    ],
    "partners": ["Indian Institute of Management Tiruchirappalli (IIM Trichy)", "Masai School"],
    "peers": ["Fellow Program Participants and Team Members"],
    "thanks": "Thank you for your time, feedback, and support throughout this journey.",
}
