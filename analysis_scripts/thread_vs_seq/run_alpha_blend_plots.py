#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alpha_blend_plots import (
    aggregate,
    find_csv_files,
    load_results,
    normalize,
    plot_best_of,
    plot_best_threads,
    plot_ns_per_pixel_best_of,
    plot_speedup_grid,
    plot_stage_boxplots,
    select_best_threads,
    write_tables,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate alpha-blend plots and summary tables.")
    parser.add_argument("out_dir", help="Output directory for plots and tables")
    parser.add_argument(
        "--csv-root",
        default="results_release/csv",
        help="Root directory to search for CSV files (default: results_release/csv)",
    )
    args = parser.parse_args()

    default_root = "results_release/csv"
    csv_root = Path(args.csv_root)
    if not csv_root.exists():
        if args.csv_root == default_root:
            candidate = Path("results_release")
            if candidate.exists():
                csv_root = candidate
                print("Default csv root not found; using results_release instead.")
            else:
                raise FileNotFoundError(
                    f"Default CSV root not found: {csv_root}. Provide --csv-root explicitly."
                )
        else:
            raise FileNotFoundError(f"CSV root does not exist: {csv_root}")

    csv_files = find_csv_files(csv_root)
    df_raw = load_results(csv_root)
    df = normalize(df_raw)
    df_agg = aggregate(df)
    df_best, best_table = select_best_threads(df_agg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "mode" in df.columns:
        mode_counts = df["mode"].dropna().value_counts()
        modes = list(mode_counts.index)
    else:
        modes = []
    mode_used = modes[0] if modes else None

    plot_best_of(df_agg, df_best, out_dir, mode=mode_used)
    plot_speedup_grid(df_agg, out_dir, mode=mode_used)
    plot_ns_per_pixel_best_of(df_agg, df_best, out_dir, mode=mode_used)
    plot_stage_boxplots(df, df_best, out_dir, mode=mode_used)
    plot_best_threads(df_best, out_dir, mode=mode_used)
    write_tables(df, df_agg, df_best, out_dir, mode=mode_used)

    _print_summary(df, df_agg, df_best, csv_files, out_dir, mode_used, modes)
    _print_best_table(best_table)
    return 0


def _print_summary(
    df: pd.DataFrame,
    df_agg: pd.DataFrame,
    df_best: pd.DataFrame,
    csv_files: list[str],
    out_dir: Path,
    mode_used: str | None,
    modes: list[str],
) -> None:
    mpix_vals = sorted(df["mpix"].dropna().unique())
    print(f"Loaded {len(df)} rows from {len(csv_files)} CSV files.")
    if modes:
        print(f"Modes found: {modes}")
    if mode_used is not None and len(modes) > 1:
        print(f"Mode used for plots/tables: {mode_used}")
    print(f"MP values: {mpix_vals}")
    print(f"Aggregated rows: {len(df_agg)}")
    print(f"Best-par rows: {len(df_best)}")
    print(f"Output directory: {out_dir}")


def _print_best_table(best_table: pd.DataFrame) -> None:
    if best_table.empty:
        return
    print("Best threads per mpix (summary):")
    print(best_table.to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
