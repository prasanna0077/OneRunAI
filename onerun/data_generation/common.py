"""Shared lookups + maths helpers for the synthetic data generators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from onerun.config import ACQUIRER

REGIONS = ["North", "South", "East", "West", "Central"]

# city, state, region, tier, latitude, longitude
_CITY_ROWS = [
    ("Chennai", "Tamil Nadu", "South", 1, 13.08, 80.27),
    ("Coimbatore", "Tamil Nadu", "South", 2, 11.02, 76.96),
    ("Madurai", "Tamil Nadu", "South", 2, 9.92, 78.12),
    ("Bengaluru", "Karnataka", "South", 1, 12.97, 77.59),
    ("Mysuru", "Karnataka", "South", 2, 12.30, 76.64),
    ("Hyderabad", "Telangana", "South", 1, 17.39, 78.49),
    ("Visakhapatnam", "Andhra Pradesh", "South", 2, 17.69, 83.22),
    ("Kochi", "Kerala", "South", 2, 9.93, 76.27),
    ("Mumbai", "Maharashtra", "West", 1, 19.08, 72.88),
    ("Pune", "Maharashtra", "West", 1, 18.52, 73.86),
    ("Ahmedabad", "Gujarat", "West", 1, 23.02, 72.57),
    ("Surat", "Gujarat", "West", 2, 21.17, 72.83),
    ("Rajkot", "Gujarat", "West", 3, 22.30, 70.80),
    ("Delhi", "Delhi", "North", 1, 28.61, 77.21),
    ("Gurugram", "Haryana", "North", 2, 28.46, 77.03),
    ("Jaipur", "Rajasthan", "North", 2, 26.91, 75.79),
    ("Lucknow", "Uttar Pradesh", "North", 2, 26.85, 80.95),
    ("Kanpur", "Uttar Pradesh", "North", 2, 26.45, 80.33),
    ("Chandigarh", "Chandigarh", "North", 2, 30.73, 76.78),
    ("Ludhiana", "Punjab", "North", 2, 30.90, 75.86),
    ("Kolkata", "West Bengal", "East", 1, 22.57, 88.36),
    ("Bhubaneswar", "Odisha", "East", 2, 20.30, 85.82),
    ("Patna", "Bihar", "East", 2, 25.59, 85.14),
    ("Guwahati", "Assam", "East", 2, 26.14, 91.74),
    ("Ranchi", "Jharkhand", "East", 3, 23.34, 85.31),
    ("Nagpur", "Maharashtra", "Central", 2, 21.15, 79.09),
    ("Bhopal", "Madhya Pradesh", "Central", 2, 23.26, 77.41),
    ("Indore", "Madhya Pradesh", "Central", 2, 22.72, 75.86),
    ("Raipur", "Chhattisgarh", "Central", 3, 21.25, 81.63),
    ("Jabalpur", "Madhya Pradesh", "Central", 3, 23.18, 79.99),
]
CITIES = pd.DataFrame(_CITY_ROWS, columns=["city", "state", "region", "tier", "lat", "lon"])


def make_rng(seed: int, salt: int = 0) -> np.random.Generator:
    """Deterministic, independent random stream per (seed, salt)."""
    return np.random.default_rng(int(seed) * 1000 + int(salt))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def solve_intercept(logit: np.ndarray, target_rate: float) -> float:
    """Find b so that mean(sigmoid(logit + b)) == target_rate (bisection)."""
    lo, hi = -12.0, 12.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoid(logit + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def acquirer_region_share(region: str) -> float:
    return float(ACQUIRER["region_branch_share"][region])


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / (sd if sd else 1.0)
