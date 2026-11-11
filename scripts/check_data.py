"""Step 3: quality report on the downloaded prices, for all 50 ETFs.

Run from the project root, after scripts/update_prices.py:
    python scripts/check_data.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qdc.data import load_prices, load_universe  # noqa: E402
from qdc.quality import quality_report  # noqa: E402

if __name__ == "__main__":
    prices = load_prices()
    report = quality_report(prices, load_universe())
    out = ROOT / "outputs" / f"data_quality_{prices.index[-1].date()}.csv"
    report.to_csv(out)

    print(report["status"].value_counts().to_string(), "\n")
    problems = report[report["status"] != "ok"]
    for ticker, row in problems.iterrows():
        print(f"{ticker:<18} {row['status']:<14} {'; '.join(row['issues'])}")
    print(f"\nFull report saved to {out}")
