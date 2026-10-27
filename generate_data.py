"""Generate the OneRun AI Excel datasets.

PyCharm:  right-click this file -> Run 'generate_data'   (writes into ./data)
Terminal: python generate_data.py --seed 7 --banks 60 --customers-per-bank 300
"""
import argparse

from onerun.config import DATA_DIR, DEFAULT_SEED
from onerun.data_generation.generator import generate_all


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic OneRun AI datasets (Excel).")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (same seed = same data)")
    ap.add_argument("--banks", type=int, default=40, help="candidate target institutions")
    ap.add_argument("--customers-per-bank", type=int, default=250, help="customer sample per institution")
    ap.add_argument("--deals", type=int, default=700, help="historical deals for the acquisition model")
    ap.add_argument("--churn-history", type=int, default=8000, help="past-merger customers for the churn model")
    a = ap.parse_args()

    manifest = generate_all(a.seed, a.banks, a.customers_per_bank, a.deals, a.churn_history, progress=print)
    print(f"\nDatasets written to: {DATA_DIR}")
    for fname, sheets in manifest.items():
        print(f"\n  {fname}")
        for sheet, (rows, cols) in sheets.items():
            print(f"     - {sheet:<30} {rows:>7,} rows x {cols} cols")


if __name__ == "__main__":
    main()
