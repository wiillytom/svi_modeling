"""
CLI runner for the systematic roll-strategy backtest (Lucic & Sepp 2024).

Examples
--------
# ALL strategies (single + multi leg, long & short) in one pass, paper-style table:
PYTHONPATH="/Users/macbookair/Internship Natixis" \
  /opt/anaconda3/envs/natixis_internship/bin/python3 \
  volatility_surface/backtest/run_roll.py \
    --options volatility_surface/notebooks/clean_chunks_full \
    --perp "2 - Data/parquets/eth_perp_1min_jan-jun2025.parquet" \
    --all --frequency weekly --coin ETH --csv results_weekly.csv

# a single structure:
... run_roll.py --options ... --perp ... --structure "Short Straddle" --frequency weekly

# quick smoke run on a slice of the data:
... run_roll.py --options ... --perp ... --all --max-snaps 5000
"""

from __future__ import annotations

import argparse

from volatility_surface.backtest import roll_engine as R
from volatility_surface.backtest import roll_results as RR


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--options", required=True,
                   help="cleaned option parquet OR a directory of chunk_*.parquet")
    p.add_argument("--perp", required=True, help="perp OHLCV parquet (timestamp_ms, close)")
    p.add_argument("--all", action="store_true",
                   help="run the whole catalog and print the paper-style table")
    p.add_argument("--structure", default="Short Straddle",
                   help="single structure name (ignored with --all); see --list")
    p.add_argument("--frequency", default="weekly",
                   help="weekly | monthly | quarterly, or comma-separated for --all")
    p.add_argument("--coin", default="ETH", help="benchmark row label")
    p.add_argument("--initial-coin", type=float, default=1.0)
    p.add_argument("--size-multiple", type=float, default=1.0)
    p.add_argument("--funding-annual-rate", type=float, default=0.0,
                   help="constant annualised funding proxy (Eq 29); 0 = omitted")
    p.add_argument("--max-snaps", type=int, default=None, help="cap snapshots (smoke run)")
    p.add_argument("--csv", default=None, help="write the results table to this CSV")
    p.add_argument("--list", action="store_true", help="list catalog names and exit")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.list:
        print("\n".join(R.catalog_names()))
        return

    if args.all:
        freqs = tuple(f.strip() for f in args.frequency.split(","))
        results = R.run_all_strategies(
            args.options, args.perp, frequencies=freqs,
            initial_coin=args.initial_coin, size_multiple=args.size_multiple,
            funding_annual_rate=args.funding_annual_rate,
            max_snaps=args.max_snaps, verbose=not args.quiet)
        table = RR.results_table(results, benchmark_name=args.coin)
        print()
        print(RR.format_table(table))
        if args.csv:
            table.to_csv(args.csv)
            print(f"\nsaved table -> {args.csv}")
        return

    res = R.run_roll_backtest(
        args.options, args.perp, structure=args.structure, frequency=args.frequency,
        initial_coin=args.initial_coin, size_multiple=args.size_multiple,
        funding_annual_rate=args.funding_annual_rate,
        max_snaps=args.max_snaps, verbose=not args.quiet)
    print()
    RR.print_summary(RR.summarize_rolls(res))


if __name__ == "__main__":
    main()
